cmake_minimum_required(VERSION 3.16)
project(jam C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # compile_commands.json for clangd

option(JAM_JNI "Build the JNI binding into libjam" ON)

# Core engine + thread pool + the portable-C kernel floor (always built, no -m flags).
add_library(jam SHARED
    src/jam.c
    src/jam_pool.c
    src/jam_kernels_generic.c
)
target_include_directories(jam PUBLIC include)
set_target_properties(jam PROPERTIES VERSION 1.0.0 SOVERSION 1)   # libjam.so.1 -> ABI guard via soname

find_package(Threads REQUIRED)
target_link_libraries(jam PRIVATE Threads::Threads m)   # libm: lrintf/fabsf in the requant floor

# Per-ISA kernel TUs: each compiled with its OWN flags and bound at create. The dispatcher (src/jam.c)
# stays at BASELINE flags — never -mavx512 — so it can't emit AVX-512 before the CPU check.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    # SSE3 (Prescott 2004+, true SSE3 — no SSSE3 maddubs, no F16C): 128-bit int8 Q8_0 via sign-extend+madd
    # with a software fp16 scale. The pre-AVX2 x86 floor, replacing the generic dequant-to-float path.
    add_library(jam_sse3 OBJECT src/jam_kernels_sse3.c)
    target_include_directories(jam_sse3 PRIVATE include)
    target_compile_options(jam_sse3 PRIVATE -msse3)
    target_compile_definitions(jam_sse3 PRIVATE JAM_HAVE_SSE3=1)
    target_sources(jam PRIVATE $<TARGET_OBJECTS:jam_sse3>)
    target_compile_definitions(jam PRIVATE JAM_HAVE_SSE3=1)

    # AVX2 (Haswell 2013+): F32 + Q8_0 via maddubs (no VNNI). Covers consumer/older CPUs.
    add_library(jam_avx2 OBJECT src/jam_kernels_avx2.c src/jam_kernels_kquant_avx2.c)
    target_include_directories(jam_avx2 PRIVATE include)
    target_compile_options(jam_avx2 PRIVATE -mavx2 -mfma -mf16c)
    target_compile_definitions(jam_avx2 PRIVATE JAM_HAVE_AVX2=1)
    target_sources(jam PRIVATE $<TARGET_OBJECTS:jam_avx2>)
    target_compile_definitions(jam PRIVATE JAM_HAVE_AVX2=1)

    # AVX-VNNI (Alder/Raptor Lake): 256-bit vpdpbusd without AVX-512. Q8_0 only (F32 reuses AVX2).
    add_library(jam_avxvnni OBJECT src/jam_kernels_avxvnni.c)
    target_include_directories(jam_avxvnni PRIVATE include)
    target_compile_options(jam_avxvnni PRIVATE -mavxvnni -mavx2 -mfma -mf16c)
    target_compile_definitions(jam_avxvnni PRIVATE JAM_HAVE_AVXVNNI=1)
    target_sources(jam PRIVATE $<TARGET_OBJECTS:jam_avxvnni>)
    target_compile_definitions(jam PRIVATE JAM_HAVE_AVXVNNI=1)

    # AVX-512BW without VNNI (Skylake-SP/X): 512-bit maddubs Q8_0. Built WITHOUT -mavx512vnni so it
    # provably emits no VNNI instructions, hence safe to bind on AVX-512 CPUs that lack VNNI.
    add_library(jam_avx512bw OBJECT src/jam_kernels_avx512bw.c)
    target_include_directories(jam_avx512bw PRIVATE include)
    target_compile_options(jam_avx512bw PRIVATE -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -mf16c)
    target_compile_definitions(jam_avx512bw PRIVATE JAM_HAVE_AVX512BW=1)
    target_sources(jam PRIVATE $<TARGET_OBJECTS:jam_avx512bw>)
    target_compile_definitions(jam PRIVATE JAM_HAVE_AVX512BW=1)

    # AVX-512: F32 (avx512f/bw/dq) + Q8_0 VNNI (avx512vnni + avx512vl 256-bit dpbusd + f16c).
    add_library(jam_avx512 OBJECT src/jam_kernels_avx512.c)
    target_include_directories(jam_avx512 PRIVATE include)
    target_compile_options(jam_avx512 PRIVATE
        -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mfma -mf16c)
    target_compile_definitions(jam_avx512 PRIVATE JAM_HAVE_AVX512=1
        $<$<BOOL:${JAM_Q8_RM}>:JAM_Q8_RM=${JAM_Q8_RM}>
        $<$<BOOL:${JAM_Q8_RN}>:JAM_Q8_RN=${JAM_Q8_RN}>)
    target_sources(jam PRIVATE $<TARGET_OBJECTS:jam_avx512>)
    target_compile_definitions(jam PRIVATE JAM_HAVE_AVX512=1)   # so the dispatcher binds the symbols

    # Q4_K (and future K-quants) — AVX-512-VNNI, ported from jinferjni.c (16-row repack scheme).
    add_library(jam_q4k OBJECT src/jam_kernels_q4k_avx512.c)
    target_include_directories(jam_q4k PRIVATE include)
    target_compile_options(jam_q4k PRIVATE
        -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mfma -mf16c)
    target_compile_definitions(jam_q4k PRIVATE JAM_HAVE_AVX512=1)
    target_sources(jam PRIVATE $<TARGET_OBJECTS:jam_q4k>)
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    # ARM Q8_0 + K-quant kernels: NEON baseline (vmull+vpadal), DOTPROD (sdot), I8MM (smmla). Each its own -march.
    add_library(jam_neon OBJECT src/jam_kernels_neon.c src/jam_kernels_kquant_neon.c)
    target_include_directories(jam_neon PRIVATE include)
    target_compile_options(jam_neon PRIVATE -march=armv8-a)
    target_compile_definitions(jam_neon PRIVATE JAM_HAVE_NEON=1)
    target_sources(jam PRIVATE $<TARGET_OBJECTS:jam_neon>)
    target_compile_definitions(jam PRIVATE JAM_HAVE_NEON=1)

    add_library(jam_dotprod OBJECT src/jam_kernels_dotprod.c src/jam_kernels_kquant_dotprod.c)
    target_include_directories(jam_dotprod PRIVATE include)
    target_compile_options(jam_dotprod PRIVATE -march=armv8.2-a+dotprod)
    target_compile_definitions(jam_dotprod PRIVATE JAM_HAVE_DOTPROD=1)
    target_sources(jam PRIVATE $<TARGET_OBJECTS:jam_dotprod>)
    target_compile_definitions(jam PRIVATE JAM_HAVE_DOTPROD=1)

    add_library(jam_i8mm OBJECT src/jam_kernels_i8mm.c)
    target_include_directories(jam_i8mm PRIVATE include)
    target_compile_options(jam_i8mm PRIVATE -march=armv8.6-a+i8mm)
    target_compile_definitions(jam_i8mm PRIVATE JAM_HAVE_I8MM=1)
    target_sources(jam PRIVATE $<TARGET_OBJECTS:jam_i8mm>)
    target_compile_definitions(jam PRIVATE JAM_HAVE_I8MM=1)
endif()

if (JAM_JNI)
    find_package(JNI REQUIRED)
    target_sources(jam PRIVATE src/jam_jni.c)
    target_include_directories(jam PRIVATE ${JNI_INCLUDE_DIRS})
endif()

# Apple GPU backend (hand-tuned MSL GEMM for prefill, zero-copy over unified memory). Distinct from
# jota-metal, which GENERATES/fuses kernels — jam-metal is the single absolute-fastest matmul. Wraps
# host pointers via newBufferWithBytesNoCopy (needs 16 KB page-aligned buffers), big-GEMM-gated.
if (APPLE)
    # Metal GPU backend (opt-in via JAM_ISA=metal). MSL compiled at runtime; no .metallib step.
    enable_language(OBJCXX)
    target_sources(jam PRIVATE src/jam_metal.mm)
    set_source_files_properties(src/jam_metal.mm PROPERTIES COMPILE_OPTIONS "-fno-objc-arc")
    target_link_libraries(jam PRIVATE "-framework Metal" "-framework Foundation")
    target_compile_definitions(jam PRIVATE JAM_HAVE_METAL=1)
endif()

# ---- stage the native lib into the fat-jar layout: dist/native/com/qxotic/jam/native/<os>-<arch>/<lib> ----
# CI builds this on each platform and uploads dist/native/; the assemble job merges them + runs build-jar.sh.
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
    set_tests_properties(correctness_env_cap PROPERTIES ENVIRONMENT "JAM_ISA=avx2;JAM_NUM_THREADS=2")
endif()
