#!/usr/bin/env bash
# The shipped libjam set: build every platform library from ONE source tree, stage them under
# dist/release, stamp them, and verify the staged set before the release build packages it.
# (dist/native is the dev tree: every plain cmake/Maven build stages this host's library there.
# The release tree is separate precisely so a dev build can never overwrite a release library.)
#
#   scripts/natives.sh build     # wipe dist/release, build + stage + stamp every shipped target
#   scripts/natives.sh verify    # the release gate: every shipped target staged, current, and sane
#
# Shipped targets and where each is built:
#   linux-x86-64     this host, zig cc against glibc 2.17 (cmake/toolchains/linux-x86-64-glibc2.17.cmake)
#   windows-x86-64   this host, MinGW-w64 cross (cmake/toolchains/windows-x86-64.cmake)
#   darwin-aarch64   a Mac over ssh: the source tree is rsynced there, built with Xcode's clang
#                    (Metal on), its ctest suite run there, and the dylib fetched back
#
# Environment:
#   JAM_MAC       user@host of the Apple-silicon Mac (required for darwin-aarch64; needs cmake + a JDK)
#   JAM_MAC_DIR   scratch directory on the Mac (default ~/.cache/jam-native-build)
#   JDK           a JDK root for the cross builds' jni.h (default: JAVA_HOME, else derived from javac)
#   ZIG           the zig binary for the Linux leg (default: `zig` on PATH; 0.16 expected)
#   JAM_TARGETS   space-separated subset of the shipped targets, for iterating on one leg. A subset
#                 never passes `verify`; the release needs the whole set.
#
# Why stamps: builds used to accumulate in dist/native across months, and a library built before a
# JNI rename shipped and crashed on load. Every staged library now carries the digest of the native
# sources it was built from and its own SHA-256; `verify` (run by the release profile) refuses
# anything else.
set -euo pipefail
shopt -s nullglob

SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.."                    # -> jam-native
NATIVE=dist/release/com/qxotic/jam/native
SHIPPED="linux-x86-64 windows-x86-64 darwin-aarch64"
STAMP=jam-native.stamp
GLIBC_FLOOR=2.17
ZIG_EXPECTED=0.16
# Every exported symbol the Java side binds. Mach-O prefixes an underscore, so match bare names.
SYMBOLS="jam_mm jam_pack_abi jam_pack_size Java_com_qxotic_jam_libjam_NativeJAM_mmJni Java_com_qxotic_jam_libjam_NativeJAM_createPfJni Java_com_qxotic_jam_libjam_NativeJAM_destroyJni"

die() { echo "natives: $*" >&2; exit 1; }

libfile() {
    case "$1" in
        windows-*) echo jam.dll ;;
        darwin-*)  echo libjam.dylib ;;
        *)         echo libjam.so ;;
    esac
}

# Digest of everything that shapes the native library: the C/ObjC++ sources, headers, the CMake
# build and the cross toolchains. Java sources and tests are deliberately excluded.
digest() {
    find CMakeLists.txt cmake include src -type f \
        -not -path 'src/main/*' -not -path 'src/test/*' -print0 \
        | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
}

pack_abi() { sed -n 's/^#define JAM_PACK_ABI[[:space:]]*\([0-9]*\).*/\1/p' include/jam.h; }

sha256() { sha256sum "$1" | cut -d' ' -f1; }

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

# Copy the built library into the staging tree ourselves. CMake's POST_BUILD staging only runs when
# the target relinks, so an up-to-date build after `rm -rf dist/native` would stage nothing.
stage() {  # <target> <built file>
    mkdir -p "$NATIVE/$1"
    cp -L "$2" "$NATIVE/$1/$(libfile "$1")"
}

jdk_root() {
    if [ -n "${JDK:-}" ]; then echo "$JDK"; return; fi
    if [ -n "${JAVA_HOME:-}" ]; then echo "$JAVA_HOME"; return; fi
    local javac
    javac=$(command -v javac) || die "no JDK: set JDK or JAVA_HOME (jni.h is needed for the cross builds)"
    dirname "$(dirname "$(readlink -f "$javac")")"
}

build_linux_x86_64() {
    local b=build-linux-x86-64 zig=${ZIG:-zig} v
    v=$("$zig" version)
    case "$v" in "$ZIG_EXPECTED".*) ;; *) echo "natives: WARNING: zig $v, expected $ZIG_EXPECTED.x" >&2 ;; esac
    # The toolchain's zig-cc wrapper reads $ZIG, so the chosen binary reaches every compile.
    ZIG=$zig cmake -S . -B "$b" -DCMAKE_BUILD_TYPE=Release -DJAM_STRIP=ON \
        -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-x86-64-glibc$GLIBC_FLOOR.cmake
    ZIG=$zig cmake --build "$b" -j
    ctest --test-dir "$b" --output-on-failure    # the 2.17-targeted binaries run fine on this host
    echo "natives: linux-x86-64 contexts: $("$b/jam_test" 2>/dev/null | sed -n 2p)"
    stage linux-x86-64 "$b/libjam.so"
    # The floor is a release promise: fail if the link pulled in a newer glibc symbol.
    local floor
    floor=$(objdump -T "$NATIVE/linux-x86-64/libjam.so" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
    echo "natives: linux-x86-64 requires ${floor:-no versioned glibc symbols}"
    [ "$(printf '%s\n' "${floor#GLIBC_}" "$GLIBC_FLOOR" | sort -V | tail -1)" = "$GLIBC_FLOOR" ] \
        || die "linux-x86-64 needs $floor, above the $GLIBC_FLOOR floor"
}

build_windows_x86_64() {
    local b=build-windows-x86-64 jdk
    jdk=$(jdk_root)
    cmake -S . -B "$b" -DCMAKE_BUILD_TYPE=Release -DJAM_STRIP=ON -DJAM_TESTS=OFF \
        -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-x86-64.cmake \
        -DJNI_INCLUDE_DIRS="$jdk/include;$PWD/cmake/cross-jni/win32"
    cmake --build "$b" -j
    stage windows-x86-64 "$b/libjam.dll"    # MinGW names it libjam.dll; the loader wants jam.dll
}

# A non-interactive ssh shell on macOS has the bare system PATH: add Homebrew/MacPorts for cmake, and
# name the JDK for find_package(JNI) the way macOS does. Prefixed to every remote command.
MAC_ENV='export PATH=/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:$PATH; export JAVA_HOME=$(/usr/libexec/java_home);'
MAC_DIR=${JAM_MAC_DIR:-.cache/jam-native-build}   # relative to the Mac's home

mac() { ssh -o BatchMode=yes "$JAM_MAC" "$MAC_ENV $*"; }

# Everything a leg needs, checked BEFORE dist/release is wiped, so a misconfigured host fails in
# seconds with the previous set intact instead of after minutes of building.
preflight() {  # <targets>
    local t
    for t in $1; do
        case "$t" in
            linux-x86-64)
                command -v "${ZIG:-zig}" >/dev/null || die "zig not found (set ZIG=/path/to/zig, or install zig $ZIG_EXPECTED)"
                command -v objdump >/dev/null || die "objdump not found (binutils)" ;;
            windows-x86-64)
                command -v x86_64-w64-mingw32-gcc >/dev/null || die "MinGW-w64 (x86_64-w64-mingw32-gcc) not installed"
                [ -f "$(jdk_root)/include/jni.h" ] || die "$(jdk_root)/include/jni.h not found (set JDK)" ;;
            darwin-aarch64)
                [ -n "${JAM_MAC:-}" ] || die "darwin-aarch64 needs JAM_MAC=user@host (an Apple-silicon Mac)"
                command -v rsync >/dev/null || die "rsync not found"
                local report
                report=$(mac 'echo "arch=$(uname -m)"; echo "cmake=$(command -v cmake || echo MISSING)"; echo "jdk=${JAVA_HOME:-MISSING}"; echo "clang=$(xcrun --find clang 2>/dev/null || echo MISSING)"' 2>&1) \
                    || die "cannot reach $JAM_MAC over ssh (BatchMode; is the key loaded?): $report"
                case "$report" in *"arch=arm64"*) ;; *) die "$JAM_MAC is not Apple silicon: $report" ;; esac
                case "$report" in *MISSING*) die "$JAM_MAC lacks a tool (needs Homebrew cmake, a JDK, Xcode CLT): $report" ;; esac
                echo "natives: $JAM_MAC ok - $(echo "$report" | tr '\n' ' ')" ;;
        esac
    done
}

build_darwin_aarch64() {
    local target=darwin-aarch64
    echo "natives: building $target on $JAM_MAC:$MAC_DIR"
    mac "mkdir -p '$MAC_DIR/src'"
    # Only what CMake needs. --delete keeps the remote tree equal to this one (no accumulation there either).
    rsync -a --delete \
        --exclude 'build*/' --exclude 'dist/' --exclude 'target/' --exclude 'src/main/' --exclude 'src/test/' \
        ./ "$JAM_MAC:$MAC_DIR/src/"
    mac "cd '$MAC_DIR/src' \
        && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJAM_STRIP=ON \
        && cmake --build build -j \
        && ctest --test-dir build --output-on-failure \
        && echo \"natives: $target contexts: \$(./build/jam_test 2>/dev/null | sed -n 2p)\""
    mkdir -p "$NATIVE/$target"
    rsync -aL "$JAM_MAC:$MAC_DIR/src/build/libjam.dylib" "$NATIVE/$target/$(libfile $target)"
}

cmd_build() {
    local targets=${JAM_TARGETS:-$SHIPPED} t lib
    for t in $targets; do
        case " $SHIPPED " in *" $t "*) ;; *) die "unknown target '$t' (shipped: $SHIPPED)" ;; esac
    done
    preflight "$targets"
    rm -rf dist/release
    for t in $targets; do
        case "$t" in
            linux-x86-64)   build_linux_x86_64 ;;
            windows-x86-64) build_windows_x86_64 ;;
            darwin-aarch64) build_darwin_aarch64 ;;
        esac
        lib=$NATIVE/$t/$(libfile "$t")
        [ -f "$lib" ] || die "$t: build finished but $lib was not staged"
        check_symbols "$lib" || die "$t: not releasable"
        write_stamp "$t"
        echo "natives: staged $lib ($(du -h "$lib" | cut -f1))"
    done
    echo "natives: done - $(ls "$NATIVE" | tr '\n' ' ')"
    [ "$targets" = "$SHIPPED" ] || echo "natives: NOTE: subset build; 'verify' needs every shipped target"
}

cmd_verify() {
    local want abi t lib f failed=0
    want=$(digest); abi=$(pack_abi)
    [ -d "$NATIVE" ] || die "nothing staged under $NATIVE - run scripts/natives.sh build"
    for t in $SHIPPED; do
        lib=$NATIVE/$t/$(libfile "$t"); f=$NATIVE/$t/$STAMP
        if [ ! -f "$lib" ]; then echo "natives: $t: $lib missing" >&2; failed=1; continue; fi
        if [ ! -f "$f" ]; then echo "natives: $t: no $STAMP (built outside scripts/natives.sh)" >&2; failed=1; continue; fi
        if [ "$(sed -n 's/^digest //p' "$f")" != "$want" ]; then
            echo "natives: $t: built from other native sources (stamp $(sed -n 's/^commit //p' "$f")) - rebuild" >&2; failed=1
        fi
        if [ "$(sed -n 's/^abi //p' "$f")" != "$abi" ]; then
            echo "natives: $t: pack ABI $(sed -n 's/^abi //p' "$f") != $abi" >&2; failed=1
        fi
        if [ "$(sed -n 's/^sha256 //p' "$f")" != "$(sha256 "$lib")" ]; then
            echo "natives: $t: $lib is not the file that was stamped (replaced after the build)" >&2; failed=1
        fi
        check_symbols "$lib" || failed=1
    done
    for t in "$NATIVE"/*/; do
        [ -d "$t" ] || continue   # an empty tree leaves the glob unexpanded
        t=$(basename "$t")
        case " $SHIPPED " in *" $t "*) ;; *) echo "natives: unexpected staged target '$t' (not shipped)" >&2; failed=1 ;; esac
    done
    [ "$failed" = 0 ] || die "staged set is not releasable - run scripts/natives.sh build (JAM_MAC=user@mac)"
    echo "natives: verified $SHIPPED at digest ${want:0:12}, pack ABI $abi"
}

case "${1:-}" in
    build)  cmd_build ;;
    verify) cmd_verify ;;
    *) sed -n '2,/^set -/{/^set -/!p}' "$SELF"; exit 2 ;;
esac
