# Cross-compile jam for linux-aarch64 from an x86-64 Linux host (GNU cross toolchain).
#   cmake -S . -B build-linux-arm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-aarch64.cmake ...
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Only search the target sysroot for libraries/headers; use host programs (cmake, make).
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
