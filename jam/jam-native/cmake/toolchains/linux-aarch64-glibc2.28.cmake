# The shipped linux-aarch64 library: zig's bundled clang against glibc 2.28, the floor of every Linux
# artifact we ship. The twin of linux-x86-64-glibc2.28.cmake, built the same way on an aarch64 Linux
# host (the manylinux container on a Mac, where the tests run on the CPU itself): zig ships the glibc
# stubs for every version, so the host's own glibc does not matter.
#   cmake -S . -B build-linux-aarch64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-aarch64-glibc2.28.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   ${CMAKE_CURRENT_LIST_DIR}/zig-cc)
set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_LIST_DIR}/zig-c++)
set(CMAKE_C_COMPILER_TARGET   aarch64-linux-gnu.2.28)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu.2.28)

# As for x86-64: zig 0.16's linker driver crashes on CMake's `-Xlinker --dependency-file=`.
set(CMAKE_LINK_DEPENDS_USE_LINKER OFF)
