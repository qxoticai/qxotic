# Building jam

Everything is built, tested and packaged locally, inside VMs. There is no CI. A build needs
**CMake ≥ 3.16**, a **C11 compiler** (GCC or Clang; MSVC is rejected at configure), **Maven**, and
a **JDK ≥ 25**. On macOS, Apple silicon only, `xcode-select --install` covers clang and the
Metal frameworks; install CMake with `brew install cmake`. The Windows library is cross-compiled from Linux with zig (below).

**Maven** runs cmake, javac and the tests in one step, from the **repository root**. jam is not
dependency-closed (`jam-core` builds on `jota-core`), so `mvn` inside this directory only works
after a root `mvn install`:

```sh
mvn -pl jam/jam-vector -am package -DskipTests   # jars under jam/*/target (the native lib builds first)
mvn -pl jam/jam-vector -am verify                # configure + build + JUnit incl. the parity suite
```

**Or build just the native library with cmake**, with no JVM, for the C API or to pre-stage
`dist/native/`:

```sh
cmake -B build                # Release by default
cmake --build build           # -> build/libjam.so, staged into dist/native/<os>-<arch>/
```

Flags: `-DJAM_METAL=OFF` (no Metal), `-DJAM_JNI=OFF` (C only, drops the JDK requirement),
`-DJAM_TESTS=OFF`, `-DJAM_STRIP=ON`, `-DJAM_SANITIZE=ON` (ASan+UBSan test build).

A build carries **every kernel tier of its target architecture**, since any x86-64 compiler emits
the AVX-512 TUs, and the library binds the best tier the CPU supports at runtime, so one artifact
runs on any CPU of its arch. Compilers older than GCC 11 or Clang 12 skip the AVX-VNNI tier
(probed, never fatal).

## The fat jar (every platform)

`jam.jar` bundles one native library per platform under `com/qxotic/jam/native/<os>-<arch>/`.
`NativeLoader` extracts and loads the matching one at first use, falling back to the pure-Java
backends when none matches. The shipped set, and where each library is built:

| artifact | built on | toolchain |
|---|---|---|
| `linux-x86-64/libjam.so` | the Linux release host | `zig cc` targeting glibc 2.17 + `cmake/toolchains/linux-x86-64-glibc2.17.cmake` |
| `linux-aarch64/libjam.so` | the Mac's manylinux container | `zig cc` targeting glibc 2.17 + `cmake/toolchains/linux-aarch64-glibc2.17.cmake` |
| `windows-x86-64/jam.dll` | the Linux release host | `zig cc` targeting `x86_64-windows-gnu` + `cmake/toolchains/windows-x86-64.cmake` |
| `darwin-aarch64/libjam.dylib` | an Apple-silicon Mac | Xcode CLT (clang, ObjC++, Metal); Intel Macs unsupported |

Other platforms get the Java backends.

One script builds, on each host, the libraries that host ships into `dist/release` (apart from
`dist/native`, where every ordinary build stages this host's library), stamps each with the digest of
the native sources it came from and its own checksum, and imports the other host's:

```sh
make jam-natives    # on each host, at the release commit: natives.sh build, and on a Mac the container leg too
jam-native/scripts/natives.sh import <dir>/darwin-aarch64 <dir>/linux-aarch64   # on the Linux host: the Mac's staged directories, carried over
```

The Linux host builds its own library and the Windows one; the Mac builds its own and, in the
manylinux aarch64 container (docker through Colima), the Linux aarch64 one, on the CPU itself. Both
Linux libraries are compiled with [zig](https://ziglang.org) as the C compiler, targeting
`x86_64-linux-gnu.2.17` and `aarch64-linux-gnu.2.17`: zig bundles the glibc stubs for every
version, so the artifacts run on RHEL 7 or Ubuntu 14.04 class systems whatever the build box runs,
and the script fails if a link ever needs a newer glibc symbol. The
Windows library is zig's too, targeting `x86_64-windows-gnu`, with zig's MinGW-w64 runtime linked in,
so `jam.dll` needs only `KERNEL32.dll` and the Universal C Runtime of Windows 10 and later (the
checked-in `cmake/cross-jni/win32/` shim supplies the target `jni_md.h`; any JDK's `jni.h` works). The Mac builds its library with Metal on and runs the
C test suite on the real GPU. `import` checks a library as `verify` does, its stamp naming these
native sources included, before staging it. `JAM_TARGETS="linux-x86-64"` builds a subset of this
host's targets while iterating on one leg.

The release build (`-Prelease`) never runs cmake: it packages the staged set after
`scripts/natives.sh verify` has checked that every shipped library is present, unchanged since its
build, built from the current native sources, and exports every symbol the Java side binds.
Libraries used to accumulate across builds until a pre-rename `jam.dll` shipped and crashed on
load; the stamps make that impossible. `NativeJAMProvider` also binds every symbol before it
reports the backend available, so an unusable bundle degrades to the Java backends with a warning
instead of failing on the first matmul.

Host requirements: cmake, a JDK, zig 0.16 and binutils on the Linux host; Xcode Command Line
Tools, cmake and any JDK on the Mac. The script checks all of that first and only then replaces a
staged library, so a misconfigured host fails in seconds with the previous set intact.

## Tests

```sh
mvn -f jam/jam-native/pom.xml test     # JNI + FFM on AUTO; host-pool bridge on generic CPU
cd build && ctest --output-on-failure   # every kernel, 1 & 3 threads, vs a double-precision reference
./jam_bench [M N K] [iters]             # GMAC/s (compute) and GB/s (bandwidth)
```

On Apple Silicon, the AUTO Java passes exercise Metal. Maven runs `HostPoolTest` separately with
`JAM_NATIVE_ISA=generic` because a GPU matmul has no CPU work and therefore never calls the host pool.
