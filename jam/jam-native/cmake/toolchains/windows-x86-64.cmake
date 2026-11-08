# The shipped windows-x86-64 library, cross-compiled on Linux with zig's bundled clang and MinGW-w64
# runtime, which it links in: jam.dll needs only KERNEL32 and the Universal C Runtime of Windows 10+.
# Like the Linux toolchains, it needs only a `zig` on PATH (scripts/natives.sh pins the version).
#   cmake -S . -B build-windows-x86-64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-x86-64.cmake \
#         -DJNI_INCLUDE_DIRS="$JDK/include;$PWD/cmake/cross-jni/win32"
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
# zig needs its `cc` subcommand, which CMake cannot pass; the wrappers add it (and honor $ZIG).
set(CMAKE_C_COMPILER   ${CMAKE_CURRENT_LIST_DIR}/zig-cc)
set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_LIST_DIR}/zig-c++)
set(CMAKE_C_COMPILER_TARGET   x86_64-windows-gnu)
set(CMAKE_CXX_COMPILER_TARGET x86_64-windows-gnu)
# zig 0.16's linker driver crashes on CMake's `-Xlinker --dependency-file=`; let CMake track link
# dependencies itself.
set(CMAKE_LINK_DEPENDS_USE_LINKER OFF)
