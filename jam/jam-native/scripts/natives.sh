#!/usr/bin/env bash
# The shipped libjam set: each host builds the platform libraries it can, stages them under
# dist/release and stamps them; the libraries of another host are imported; the staged set is
# verified before the release build packages it. (dist/native is the dev tree: every plain
# cmake/Maven build stages this host's library there. The release tree is separate precisely so a
# dev build can never overwrite a release library.)
#
#   scripts/natives.sh build          # build, stage and stamp this host's shipped targets
#   scripts/natives.sh import DIR...  # stage another host's stamped targets, checked as verify checks
#   scripts/natives.sh verify         # the release gate: every shipped target staged, current, and sane
#
# Shipped targets and the host that builds each:
#   linux-x86-64     Linux x86-64: zig cc against glibc 2.28 (cmake/toolchains/linux-x86-64-glibc2.28.cmake)
#   linux-aarch64    Linux aarch64, the manylinux container on an Apple-silicon Mac (`make jam-natives` there
#                    runs this script inside it): zig cc against glibc 2.28, the tests on the CPU itself
#   windows-x86-64   Linux x86-64: zig cc cross, MinGW-w64 ABI (cmake/toolchains/windows-x86-64.cmake)
#   darwin-aarch64   an Apple-silicon Mac: Xcode's clang, Metal on, its ctest suite run on the GPU
#
# A release takes both hosts, at the same commit: `build` on each, then the Mac's
# dist/release/com/qxotic/jam/native/darwin-aarch64 and linux-aarch64 directories, carried over by
# any means, go to `import` on the Linux host, where `verify` and the release build run.
#
# Environment:
#   JDK           a JDK root for jni.h (default: JAVA_HOME, else java_home on a Mac, else derived from javac)
#   ZIG           the zig binary for the Linux legs (default: `zig` on PATH; 0.16 expected)
#   JAM_TARGETS   space-separated subset of this host's targets, for iterating on one leg
#
# Why stamps: builds used to accumulate in dist/native across months, and a library built before a
# JNI rename shipped and crashed on load. Every staged library now carries the digest of the native
# sources it was built from and its own SHA-256; `verify` (run by the release profile) refuses
# anything else.
set -euo pipefail
shopt -s nullglob

SELF=$(readlink -f "$0")
CALLER=$PWD                                   # where import's relative directories are
cd "$(dirname "$SELF")/.."                    # -> jam-native
NATIVE=dist/release/com/qxotic/jam/native
SHIPPED="linux-x86-64 linux-aarch64 windows-x86-64 darwin-aarch64"
STAMP=jam-native.stamp
GLIBC_FLOOR=2.28   # every Linux artifact's floor: the wheels and archives are manylinux_2_28
ZIG_EXPECTED=0.16
# Every exported symbol: jam.h's functions and the JNI entry points jam_jni.c defines (the same lists
# ctest's exports check derives). Mach-O prefixes an underscore, so match bare names.
SYMBOLS="$(sed -nE 's/^JAM_API .*[ *](jam_[a-z0-9_]+)\(.*/\1/p' include/jam.h | tr '\n' ' ')\
$(sed -nE 's/^((Java|JNI_OnLoad)_[A-Za-z0-9_]+)\(.*/\1/p' src/jam_jni.c | tr '\n' ' ')"

die() { echo "natives: $*" >&2; exit 1; }

libfile() {
    case "$1" in
        windows-*) echo jam.dll ;;
        darwin-*)  echo libjam.dylib ;;
        *)         echo libjam.so ;;
    esac
}

# Digest of everything that shapes the native library: the C/ObjC++ sources, headers, the CMake
# build and the cross toolchains. Java sources and tests are deliberately excluded, and so are
# dotfiles (a Mac's .DS_Store), so two hosts at one commit agree.
digest() {
    find CMakeLists.txt cmake include src -type f -not -name '.*' \
        -not -path 'src/main/*' -not -path 'src/test/*' -print0 \
        | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
}

pack_abi() { sed -n 's/^#define JAM_PACK_ABI[[:space:]]*\([0-9]*\).*/\1/p' include/jam.h; }

sha256() { sha256sum "$1" | cut -d' ' -f1; }

shipped() { case " $SHIPPED " in *" $1 "*) ;; *) return 1 ;; esac; }

# This host's targets: the ones it can build.
host_targets() {
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)  echo "linux-x86-64 windows-x86-64" ;;
        Linux-aarch64) echo "linux-aarch64" ;;
        Darwin-arm64)  echo "darwin-aarch64" ;;
        *) die "this host builds no shipped target ($(uname -s) $(uname -m)): linux x86-64, linux aarch64 and an Apple-silicon Mac do" ;;
    esac
}

check_symbols() {  # <lib> - every symbol must appear in the file's symbol tables; returns 1 listing the missing
    local lib=$1 sym ok=0
    for sym in $SYMBOLS; do
        LC_ALL=C grep -aq "$sym" "$lib" || { echo "natives: $lib: exported symbol '$sym' missing (stale or partial build)" >&2; ok=1; }
    done
    return $ok
}

write_stamp() {  # <target>
    local commit
    commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
    [ -z "$(git status --porcelain -- . 2>/dev/null)" ] || commit="$commit+dirty"
    printf 'digest %s\nabi %s\ncommit %s\nsha256 %s\n' "$(digest)" "$(pack_abi)" "$commit" \
        "$(sha256 "$NATIVE/$1/$(libfile "$1")")" > "$NATIVE/$1/$STAMP"
}

# Copy the built library into the release tree ourselves: CMake's jam_stage target stages into the
# dev tree (dist/native) only, and the release set must come from exactly the build just made.
stage() {  # <target> <built file>
    rm -rf "${NATIVE:?}/$1" && mkdir -p "$NATIVE/$1"
    cp -L "$2" "$NATIVE/$1/$(libfile "$1")"
}

jdk_root() {
    if [ -n "${JDK:-}" ]; then echo "$JDK"; return; fi
    if [ -n "${JAVA_HOME:-}" ]; then echo "$JAVA_HOME"; return; fi
    if [ -x /usr/libexec/java_home ]; then /usr/libexec/java_home; return; fi   # a Mac's /usr/bin/javac is a stub
    local javac
    javac=$(command -v javac) || die "no JDK: set JDK or JAVA_HOME (jni.h is needed for the cross builds)"
    dirname "$(dirname "$(readlink -f "$javac")")"
}

# Every leg configures --fresh: CMake reads a toolchain file only on a build directory's first
# configure, so a reused cache would build with the toolchain it was first given.

# The zig every zig leg builds with ($ZIG, else zig on PATH), warning when it is not $ZIG_EXPECTED.x.
zig_bin() {
    local zig=${ZIG:-zig} v
    v=$("$zig" version)
    case "$v" in "$ZIG_EXPECTED".*) ;; *) echo "natives: WARNING: zig $v, expected $ZIG_EXPECTED.x" >&2 ;; esac
    echo "$zig"
}

# This host's Linux library: zig cc against glibc $GLIBC_FLOOR, the host JDK's headers for JNI, the
# tests run here (binaries linked for the floor run on any newer glibc).
build_linux() {  # <target>
    local t=$1 b=build-$1 zig jdk
    zig=$(zig_bin)
    jdk=$(jdk_root)
    # The toolchain's zig-cc wrapper reads $ZIG, so the chosen binary reaches every compile.
    ZIG=$zig cmake --fresh -S . -B "$b" -DCMAKE_BUILD_TYPE=Release -DJAM_STRIP=ON \
        -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/$t-glibc$GLIBC_FLOOR.cmake \
        -DJNI_INCLUDE_DIRS="$jdk/include;$jdk/include/linux"
    ZIG=$zig cmake --build "$b" -j
    ctest --test-dir "$b" --output-on-failure
    echo "natives: $t contexts: $("$b/jam_test" 2>/dev/null | sed -n 2p)"
    stage "$t" "$b/libjam.so"
    # The floor is a release promise: fail if the link pulled in a newer glibc symbol.
    local floor
    floor=$(objdump -T "$NATIVE/$t/libjam.so" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
    echo "natives: $t requires ${floor:-no versioned glibc symbols}"
    [ "$(printf '%s\n' "${floor#GLIBC_}" "$GLIBC_FLOOR" | sort -V | tail -1)" = "$GLIBC_FLOOR" ] \
        || die "$t needs $floor, above the $GLIBC_FLOOR floor"
}

# The Windows library, cross-compiled with zig cc; the JNI binding pairs the JDK's jni.h with the
# checked-in Windows jni_md.h. Its tests cannot run here.
build_windows_x86_64() {
    local b=build-windows-x86-64 zig jdk
    zig=$(zig_bin)
    jdk=$(jdk_root)
    ZIG=$zig cmake --fresh -S . -B "$b" -DCMAKE_BUILD_TYPE=Release -DJAM_STRIP=ON -DJAM_TESTS=OFF \
        -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-x86-64.cmake \
        -DJNI_INCLUDE_DIRS="$jdk/include;$PWD/cmake/cross-jni/win32"
    ZIG=$zig cmake --build "$b" -j
    stage windows-x86-64 "$b/jam.dll"
}

# Everything a leg needs, checked BEFORE its staged library is replaced, so a misconfigured host
# fails in seconds with the previous set intact instead of after minutes of building.
preflight() {  # <targets>
    local t
    for t in $1; do
        case "$t" in
            linux-* | windows-x86-64)
                command -v "${ZIG:-zig}" >/dev/null || die "zig not found (set ZIG=/path/to/zig, or install zig $ZIG_EXPECTED)"
                command -v objdump >/dev/null || die "objdump not found (binutils)"
                command -v strip >/dev/null || die "strip not found (binutils)"
                [ -f "$(jdk_root)/include/jni.h" ] || die "$(jdk_root)/include/jni.h not found (set JDK)" ;;
            darwin-aarch64)
                xcrun --find clang >/dev/null 2>&1 || die "no clang (xcode-select --install)"
                [ -f "$(jdk_root)/include/jni.h" ] || die "$(jdk_root)/include/jni.h not found (set JDK)" ;;
        esac
        command -v cmake >/dev/null || die "cmake not found"
    done
}

# The Mac's ctest is the real gate for jam_share and the Metal paths: everywhere else a share is
# declined (EUNSUPPORTED), so only this leg exercises the share table on a device.
build_darwin_aarch64() {
    local b=build-darwin-aarch64
    JAVA_HOME=$(jdk_root) cmake --fresh -S . -B "$b" -DCMAKE_BUILD_TYPE=Release -DJAM_STRIP=ON
    cmake --build "$b" -j
    ctest --test-dir "$b" --output-on-failure
    echo "natives: darwin-aarch64 contexts: $("$b/jam_test" 2>/dev/null | sed -n 2p)"
    stage darwin-aarch64 "$b/libjam.dylib"
}

cmd_build() {
    local here targets t lib
    here=$(host_targets); targets=${JAM_TARGETS:-$here}
    for t in $targets; do
        case " $here " in *" $t "*) ;; *) die "target '$t' is not built on this host (it builds: $here)" ;; esac
    done
    preflight "$targets"
    for t in $targets; do
        case "$t" in
            linux-*)        build_linux "$t" ;;
            windows-x86-64) build_windows_x86_64 ;;
            darwin-aarch64) build_darwin_aarch64 ;;
        esac
        lib=$NATIVE/$t/$(libfile "$t")
        [ -f "$lib" ] || die "$t: build finished but $lib was not staged"
        check_symbols "$lib" || die "$t: not releasable"
        write_stamp "$t"
        echo "natives: staged $lib ($(du -h "$lib" | cut -f1))"
    done
    echo "natives: staged $(ls "$NATIVE" | tr '\n' ' ')- verify needs all of $SHIPPED"
}

# A staged target, in <dir>: present, stamped, built from these native sources, unchanged since,
# and exporting every symbol. Returns 1 listing what is wrong.
check_target() {  # <target> <dir>
    local t=$1 lib=$2/$(libfile "$1") f=$2/$STAMP ok=0
    [ -f "$lib" ] || { echo "natives: $t: $lib missing" >&2; return 1; }
    [ -f "$f" ] || { echo "natives: $t: no $STAMP (built outside scripts/natives.sh)" >&2; return 1; }
    if [ "$(sed -n 's/^digest //p' "$f")" != "$(digest)" ]; then
        echo "natives: $t: built from other native sources (stamp $(sed -n 's/^commit //p' "$f")) - rebuild" >&2; ok=1
    fi
    if [ "$(sed -n 's/^abi //p' "$f")" != "$(pack_abi)" ]; then
        echo "natives: $t: pack ABI $(sed -n 's/^abi //p' "$f") != $(pack_abi)" >&2; ok=1
    fi
    if [ "$(sed -n 's/^sha256 //p' "$f")" != "$(sha256 "$lib")" ]; then
        echo "natives: $t: $lib is not the file that was stamped (replaced after the build)" >&2; ok=1
    fi
    check_symbols "$lib" || ok=1
    return $ok
}

cmd_import() {  # <target dir>...
    [ $# -gt 0 ] || die "import: name the directories to import (each <target>/, as build stages it)"
    local d t
    for d in "$@"; do
        case "$d" in /*) ;; *) d=$CALLER/$d ;; esac
        t=$(basename "$d")
        shipped "$t" || die "import: $d names no shipped target ($SHIPPED)"
        check_target "$t" "$d" || die "import: $d is not releasable"
        rm -rf "${NATIVE:?}/$t" && mkdir -p "$NATIVE/$t"
        cp "$d/$(libfile "$t")" "$d/$STAMP" "$NATIVE/$t/"
        echo "natives: imported $t (commit $(sed -n 's/^commit //p' "$d/$STAMP"))"
    done
}

cmd_verify() {
    local t failed=0
    [ -d "$NATIVE" ] || die "nothing staged under $NATIVE - run scripts/natives.sh build"
    for t in $SHIPPED; do
        check_target "$t" "$NATIVE/$t" || failed=1
    done
    for t in "$NATIVE"/*/; do
        t=$(basename "$t")
        shipped "$t" || { echo "natives: unexpected staged target '$t' (not shipped)" >&2; failed=1; }
    done
    [ "$failed" = 0 ] || die "staged set is not releasable - scripts/natives.sh build on each host, then import"
    local want
    want=$(digest)
    echo "natives: verified $SHIPPED at digest ${want:0:12}, pack ABI $(pack_abi)"
}

case "${1:-}" in
    build)  cmd_build ;;
    import) shift; cmd_import "$@" ;;
    verify) cmd_verify ;;
    *) sed -n '2,/^set -/{/^set -/!p}' "$SELF"; exit 2 ;;
esac
