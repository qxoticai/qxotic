# Cross-compile jam for windows-aarch64 from a Linux host (MinGW-w64 ARM64 toolchain).
# Requires the aarch64-w64-mingw32 toolchain (llvm-mingw); not in Arch repos. If unavailable this
# target is built on native Windows/ARM CI instead.
#   cmake -S . -B build-win-arm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-aarch64.cmake ...
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER aarch64-w64-mingw32-g++)
set(CMAKE_RC_COMPILER  aarch64-w64-mingw32-windres)

set(CMAKE_C_STANDARD_LIBRARIES "-static-libgcc -static -lpthread" CACHE STRING "" FORCE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
