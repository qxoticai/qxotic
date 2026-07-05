#!/usr/bin/env bash
# Assemble the fat jam.jar = compiled Java from ALL jam modules + every native lib staged under
# jam-native/dist/native/.
#
# Java is compiled by Maven (so the jam-vector Vector-API flags + the graalvm-annotations compile-only
# dep are handled correctly), reusing the pre-staged dist/native (-Djam.native.skip=true) so a
# multi-platform jar keeps whatever libs you cross-built. To make it TRULY cross-platform, populate
# dist/native/ from every target first (each platform's `cmake --build build` stages its own lib; the
# cross-build toolchains under cmake/toolchains/ stage linux-aarch64 / windows-x86-64 from Linux).
set -euo pipefail
cd "$(dirname "$0")/.."          # -> jam-native
ROOT=..                          # jam reactor root (siblings: jam-core, jam-scalar, jam-vector)
OUT=dist
CLASSES="$OUT/classes"

echo "==> mvn package (all jam modules; reuse pre-staged dist/native)"
( cd "$ROOT" && mvn -q -DskipTests -Djam.native.skip=true package )

echo "==> gathering classes + native libs"
rm -rf "$CLASSES"; mkdir -p "$CLASSES"
# jam-native/target/classes already contains the native libs (dist/native is a resource dir).
for m in "$ROOT/jam-core" "$ROOT/jam-scalar" "$ROOT/jam-vector" .; do
    [ -d "$m/target/classes" ] && cp -a "$m/target/classes/." "$CLASSES/"
done
if ! find "$CLASSES/com/qxotic/jam/native" -type f 2>/dev/null | grep -q .; then
    echo "   WARNING: no native libs bundled - run a cmake build first so dist/native/ is populated"
fi

VER="${JAM_VERSION:-0.1.0}"
printf 'Implementation-Title: jam\nImplementation-Version: %s\n' "$VER" > "$OUT/MANIFEST.MF"
jar --create --file "$OUT/jam.jar" --manifest "$OUT/MANIFEST.MF" -C "$CLASSES" .

echo "==> $OUT/jam.jar ($(du -h "$OUT/jam.jar" | cut -f1))"
jar --list --file "$OUT/jam.jar" | grep -E 'native/.*(so|dylib|dll)$|\.class$' | sed 's/^/   /'
