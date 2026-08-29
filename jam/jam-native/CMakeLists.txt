cmake_minimum_required(VERSION 3.16)
include(CheckCCompilerFlag)
project(jam C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # compile_commands.json for clangd

# A kernel library built at -O0 is always a mistake, and an unset build type silently is one  -
# the POST_BUILD staging would then ship the slow lib. Explicit -DCMAKE_BUILD_TYPE still wins.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "build type (defaulted; Release unless overridden)" FORCE)
endif()

# Every object ends up inside libjam.so, but the per-ISA OBJECT libraries do NOT inherit the shared
# library's implicit -fPIC: on ELF that breaks the link (or lets PIE code sneak in on default-PIE
# distros). Harmless on Mach-O (always PIC) and PE (no PIC concept).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The flag spellings below are GCC/Clang (-m..., -march=...); Windows builds use MinGW-w64
# (cmake/toolchains/windows-*.cmake). Fail fast instead of erroring on the first -m flag.
if (MSVC)
    message(FATAL_ERROR "jam builds with GCC/Clang, incl. MinGW-w64 for Windows "
                        "(cmake/toolchains/windows-*.cmake); MSVC flag spellings are not maintained")
endif()

# macOS is Apple-silicon only: the x86 branch below is for Linux/Windows, and an Intel-Mac (or
# Rosetta-targeted) build would ship a combination nothing tests. Fail fast, not silently.
if (APPLE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(FATAL_ERROR "jam on macOS supports Apple silicon (aarch64) only")
endif()

option(JAM_JNI "Build the JNI binding into libjam" ON)
option(JAM_METAL "Build the Apple Metal GPU backend (macOS only)" ON)
option(JAM_STRIP "Strip debug symbols from the shared library (Release builds)" OFF)

# AddressSanitizer + UBSan for the test build: instruments the whole lib (all ISA TUs) + the tests, so
# OOB / use-after-free / misalignment / int-overflow in any kernel aborts loudly. Applied globally and
# BEFORE the targets so every object inherits it. Build: cmake -B build-asan -DJAM_SANITIZE=ON -DJAM_JNI=OFF
option(JAM_SANITIZE "Build lib + tests with AddressSanitizer and UBSan" OFF)
if (JAM_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fsanitize=address,undefined)
endif()

# Core engine + thread pool + the portable-C kernel floor (always built, no -m flags).
add_library(jam SHARED
    src/jam.c
    src/jam_pool.c
    src/jam_cpu.c
    src/kernels/jam_kernels_generic.c
)
target_include_directories(jam PUBLIC include PRIVATE src)
set_target_properties(jam PROPERTIES VERSION 1.0.0 SOVERSION 1)   # libjam.so.1 -> ABI guard via soname

# JAM_BUILD marks this as the library's own compilation so jam.h stamps the public API with
# __declspec(dllexport) on Windows (PE hides everything else once jam_jni.c's JNIEXPORT appears - that
# would drop jam_mm/jam_ctx_* from jam.dll and break the Panama binding). No effect on ELF/Mach-O.
target_compile_definitions(jam PRIVATE JAM_BUILD=1)

set(THREADS_PREFER_PTHREAD_FLAG ON)   # -pthread over -lpthread where it matters (GCC/ELF)
find_package(Threads REQUIRED)
target_link_libraries(jam PRIVATE Threads::Threads m)   # libm: lrintf/fabsf in the requant floor
                                                        # (a no-op alias on MinGW, part of libSystem on macOS)

# One per-ISA kernel TU: compiled with its OWN flags and bound at create; the dispatcher (src/jam.c)
# stays at BASELINE flags. `def`=1 is defined for the TU and, when dispatch=ON, for the dispatcher too
# (so jam.c binds the TU's symbols). Extra per-TU defines can be added after the call.
function(jam_isa_tu name def dispatch flags)
    add_library(${name} OBJECT ${ARGN})
    target_include_directories(${name} PRIVATE include src)
    separate_arguments(_f UNIX_COMMAND "${flags}")
    target_compile_options(${name} PRIVATE ${_f})
    target_compile_definitions(${name} PRIVATE ${def}=1)
    target_sources(jam PRIVATE $<TARGET_OBJECTS:${name}>)
    if (dispatch)
        target_compile_definitions(jam PRIVATE ${def}=1)
    endif()
endfunction()

# Per-ISA kernel TUs: each compiled with its OWN flags and bound at create. The dispatcher (src/jam.c)
# stays at BASELINE flags - never -mavx512 - so it can't emit AVX-512 before the CPU check.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    # SSE3 (Prescott 2004+, true SSE3 - no SSSE3 maddubs, no F16C): 128-bit int8 Q8_0/Q4_0/MXFP4 + K-quants
    # via sign-extend+madd with a software fp16 scale. The pre-AVX2 x86 floor, replacing dequant-to-float.
    jam_isa_tu(jam_sse3 JAM_HAVE_SSE3 ON "-msse3"
        src/kernels/jam_kernels_sse3.c src/kernels/jam_kernels_kquant_sse3.c)

    # SSSE3 (Core 2 2006+, still pre-AVX2): Q8_0/Q4_0/MXFP4 via the 128-bit maddubs sign-trick (faster than
    # the true-SSE3 sign-extend floor). Software fp16 (no F16C). K-quants stay on the SSE3 path (already winning).
    jam_isa_tu(jam_ssse3 JAM_HAVE_SSSE3 ON "-mssse3"
        src/kernels/jam_kernels_ssse3.c)

    # AVX2 (Haswell 2013+): F32 mnpack + dense F16/BF16 + every quant via maddubs (no VNNI) - 32-block
    # quants, the 8-row K-quant repack bands, NVFP4, Q1_0. Covers consumer/older CPUs.
    jam_isa_tu(jam_avx2 JAM_HAVE_AVX2 ON "-mavx2 -mfma -mf16c"
        src/kernels/jam_kernels_avx2.c src/kernels/jam_kernels_band8_avx2.c
        src/kernels/jam_kernels_nvfp4_avx2.c src/kernels/jam_kernels_q1_0_avx2.c)

    # AVX-VNNI (Alder/Raptor Lake): 256-bit vpdpbusd without AVX-512 - the Q8_0/Q4_0 8-row repack bands
    # and the pure-256 requant (F32 reuses AVX2; K-quants use the avx2 8-row bands). The flag needs
    # GCC 11+/Clang 12+ (Ubuntu LTS MinGW cross-GCC is older): probe like avx512bf16 below, and an old
    # compiler just skips the TU - dispatch falls back to AVX2 on those builds.
    check_c_compiler_flag(-mavxvnni JAM_CC_AVXVNNI)
    if (JAM_CC_AVXVNNI)
        jam_isa_tu(jam_avxvnni JAM_HAVE_AVXVNNI ON "-mavxvnni -mavx2 -mfma -mf16c"
            src/kernels/jam_kernels_avxvnni.c)
    endif()

    # AVX-512BW without VNNI (Skylake-SP/X): 512-bit maddubs Q8_0 + NVFP4. Built WITHOUT -mavx512vnni so it
    # provably emits no VNNI instructions, hence safe to bind on AVX-512 CPUs that lack VNNI.
    jam_isa_tu(jam_avx512bw JAM_HAVE_AVX512BW ON "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -mf16c"
        src/kernels/jam_kernels_avx512bw.c)

    # AVX-512: F32 mnpack + packed-panel + dense F16/BF16 + Q8_0 VNNI (avx512vnni + avx512vl + f16c).
    jam_isa_tu(jam_avx512 JAM_HAVE_AVX512 ON "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mfma -mf16c"
        src/kernels/jam_kernels_avx512.c)
    target_compile_definitions(jam_avx512 PRIVATE
        $<$<BOOL:${JAM_Q8_RM}>:JAM_Q8_RM=${JAM_Q8_RM}>
        $<$<BOOL:${JAM_Q8_RN}>:JAM_Q8_RN=${JAM_Q8_RN}>)

    # AVX512-BF16 (Zen 4+/Cooper Lake): vdpbf16ps dense BF16 tile + packed-panel. Own TU so the flag can't
    # leak; bound at create only when the CPU reports avx512bf16.
    check_c_compiler_flag(-mavx512bf16 JAM_CC_AVX512BF16)
    if (JAM_CC_AVX512BF16)
        jam_isa_tu(jam_bf16z JAM_HAVE_AVX512BF16 ON "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512bf16 -mfma"
            src/kernels/jam_kernels_bf16_avx512bf16.c)
    endif()

    # The AVX-512-VNNI 16-row repack bands: K-quants (Q4_K/Q5_K/Q6_K) + the 32-block quants (Q8_0/Q4_0/
    # MXFP4) + Q1_0, ported from jinferjni.c. Dispatcher symbols come via JAM_HAVE_AVX512 (already set).
    jam_isa_tu(jam_q4k JAM_HAVE_AVX512 OFF "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mfma -mf16c"
        src/kernels/jam_kernels_q4k_avx512.c)
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    # ARM Q8_0/Q4_0/MXFP4 + K-quant + NVFP4 + Q1_0 kernels: NEON baseline (vmull+vpadal), DOTPROD (sdot),
    # I8MM (smmla). Each its own -march.
    jam_isa_tu(jam_neon JAM_HAVE_NEON ON "-march=armv8-a"
        src/kernels/jam_kernels_neon.c src/kernels/jam_kernels_kquant_neon.c
        src/kernels/jam_kernels_nvfp4_neon.c src/kernels/jam_kernels_q1_0_neon.c)

    jam_isa_tu(jam_dotprod JAM_HAVE_DOTPROD ON "-march=armv8.2-a+dotprod"
        src/kernels/jam_kernels_dotprod.c src/kernels/jam_kernels_kquant_dotprod.c
        src/kernels/jam_kernels_nvfp4_dotprod.c src/kernels/jam_kernels_q1_0_dotprod.c)

    jam_isa_tu(jam_i8mm JAM_HAVE_I8MM ON "-march=armv8.6-a+i8mm"
        src/kernels/jam_kernels_i8mm.c)
endif()

if (JAM_JNI)
    # Native build: find_package locates the host JDK. Cross build: find_package(JNI) can't detect a
    # foreign target JVM, so the caller passes -DJNI_INCLUDE_DIRS (jni.h + the target's jni_md.h) and we
    # honor it verbatim (the shipped fat-jar lib MUST carry the JNI binding on every platform).
    if (NOT JNI_INCLUDE_DIRS)
        find_package(JNI REQUIRED)
    endif()
    target_sources(jam PRIVATE src/jam_jni.c)
    target_include_directories(jam PRIVATE ${JNI_INCLUDE_DIRS})
endif()

# Apple GPU backend (hand-tuned MSL GEMM for prefill, zero-copy over unified memory). Distinct from
# jota-metal, which GENERATES/fuses kernels - jam-metal is the single absolute-fastest matmul. Wraps
# host pointers via newBufferWithBytesNoCopy (needs 16 KB page-aligned buffers), big-GEMM-gated.
if (APPLE AND JAM_METAL)
    # Metal GPU backend (default-on for Apple Silicon AUTO; JAM_ISA=<cpu rung> opts out). MSL compiled at runtime; no .metallib step.
    enable_language(OBJCXX)
    target_sources(jam PRIVATE src/jam_metal.mm)
    set_source_files_properties(src/jam_metal.mm PROPERTIES COMPILE_OPTIONS "-fno-objc-arc")
    target_link_libraries(jam PRIVATE "-framework Metal" "-framework Foundation")
    target_compile_definitions(jam PRIVATE JAM_HAVE_METAL=1)
endif()

# ---- strip debug symbols (opt-in, for distribution builds) ----
if(JAM_STRIP)
    if(APPLE)
        add_custom_command(TARGET jam POST_BUILD
            COMMAND strip -x "$<TARGET_FILE:jam>"
            COMMENT "jam: stripped debug symbols")
    elseif(WIN32)
        add_custom_command(TARGET jam POST_BUILD
            COMMAND ${CMAKE_STRIP} "$<TARGET_FILE:jam>"
            COMMENT "jam: stripped debug symbols")
    else()
        add_custom_command(TARGET jam POST_BUILD
            COMMAND ${CMAKE_STRIP} --strip-unneeded "$<TARGET_FILE:jam>"
            COMMENT "jam: stripped debug symbols")
    endif()
endif()

# ---- stage the native lib into the fat-jar layout: dist/native/com/qxotic/jam/native/<os>-<arch>/<lib> ----
# Each platform's build (native or via cmake/toolchains/) stages its own subdir; the subdirs are
# disjoint, so builds ACCUMULATE here. Everything is built/tested/packaged locally (a Linux VM covers
# linux native + the windows cross targets; the Mac covers darwin-aarch64) - merge the dist/native/
# trees into one checkout and run scripts/build-jar.sh. No CI by design.
# The os/arch tokens MUST match NativeLoader.java.
string(TOLOWER "${CMAKE_SYSTEM_NAME}" JAM_OS)               # linux / darwin / windows
set(JAM_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
if (JAM_ARCH MATCHES "x86_64|AMD64|amd64")
    set(JAM_ARCH "x86-64")
elseif (JAM_ARCH MATCHES "aarch64|arm64|ARM64")
    set(JAM_ARCH "aarch64")
endif()
if (JAM_OS STREQUAL "windows")
    set(JAM_LIBFILE "jam.dll")
elseif (JAM_OS STREQUAL "darwin")
    set(JAM_LIBFILE "libjam.dylib")
else()
    set(JAM_LIBFILE "libjam.so")
endif()
set(JAM_STAGE "${CMAKE_SOURCE_DIR}/dist/native/com/qxotic/jam/native/${JAM_OS}-${JAM_ARCH}")
add_custom_command(TARGET jam POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${JAM_STAGE}"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:jam>" "${JAM_STAGE}/${JAM_LIBFILE}"
    COMMENT "jam: staged ${JAM_LIBFILE} for jam.jar (${JAM_OS}-${JAM_ARCH})")

# ---- tests + benchmark ----
option(JAM_TESTS "Build tests and benchmarks" ON)
if (JAM_TESTS)
    enable_testing()
    add_executable(jam_test  tests/jam_test.c)
    add_executable(jam_bench tests/jam_bench.c)
    foreach(t jam_test jam_bench)
        target_include_directories(${t} PRIVATE include tests)
        target_link_libraries(${t} PRIVATE jam m Threads::Threads)
    endforeach()
    # jam_test internally covers EVERY supported ISA level (capped contexts) at 1 and 3 threads, so one
    # run exercises all kernels. The env-capped run additionally validates JAM_ISA parsing on the global ctx.
    add_test(NAME correctness         COMMAND jam_test)
    add_test(NAME correctness_env_cap COMMAND jam_test)
    set_tests_properties(correctness_env_cap PROPERTIES ENVIRONMENT "JAM_ISA=avx2;JAM_THREADS=2")
endif()
