# The shipped linux-x86-64 library: built with zig's bundled clang against glibc 2.17, so the
# artifact runs on any x86-64 distribution still in use (RHEL 7+, Ubuntu 14.04+) regardless of the
# release host's own glibc. zig ships the glibc stubs for every version, so no sysroot or container
# is needed - only a `zig` on PATH (scripts/natives.sh pins the version it expects).
#   cmake -S . -B build-linux-x86-64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-x86-64-glibc2.17.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# zig needs its `cc` subcommand, which CMake cannot pass; the wrappers add it (and honor $ZIG).
set(CMAKE_C_COMPILER   ${CMAKE_CURRENT_LIST_DIR}/zig-cc)
set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_LIST_DIR}/zig-c++)
set(CMAKE_C_COMPILER_TARGET   x86_64-linux-gnu.2.17)
set(CMAKE_CXX_COMPILER_TARGET x86_64-linux-gnu.2.17)

# zig 0.16's linker driver crashes on CMake's `-Xlinker --dependency-file=`; let CMake track link
# dependencies itself. (Its compiler-rt also lacks __cpu_model, which is why jam.c probes CPUID
# directly instead of calling __builtin_cpu_supports.)
set(CMAKE_LINK_DEPENDS_USE_LINKER OFF)
