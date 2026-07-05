# Cross-compile jam for windows-x86-64 from a Linux host (MinGW-w64 toolchain).
#   cmake -S . -B build-win-x64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-x86-64.cmake ...
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER   x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_RC_COMPILER  x86_64-w64-mingw32-windres)

# Ship a self-contained DLL: fold the GCC/pthread runtimes in so no MinGW DLLs must travel with jam.dll.
set(CMAKE_C_STANDARD_LIBRARIES "-static-libgcc -static -lpthread" CACHE STRING "" FORCE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
