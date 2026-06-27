#!/usr/bin/env bash
# Assemble the fat jam.jar = compiled Java classes + every native lib staged under dist/native/.
#
# Each platform stages its own lib via the CMake POST_BUILD step (cmake --build build). To produce a
# TRULY cross-platform jar, gather dist/native/ from every target platform first (CI does this by
# downloading the per-platform build artifacts into dist/native/ before running this script). Locally
# this yields a jar that's fat for whatever platforms you've built.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=dist
CLASSES="$OUT/classes"
rm -rf "$CLASSES"; mkdir -p "$CLASSES"

echo "==> javac"
javac -d "$CLASSES" java/com/qxotic/jam/*.java

echo "==> bundling native libraries"
if [ -d "$OUT/native" ] && [ -n "$(find "$OUT/native" -type f 2>/dev/null)" ]; then
    cp -r "$OUT"/native/* "$CLASSES"/
    find "$CLASSES/com/qxotic/jam/native" -type f | sed 's#.*/native/##; s/^/   /'
else
    echo "   WARNING: no native libs staged in $OUT/native — run 'cmake --build build' first"
    echo "   (the jar will be class-only and fail to load at runtime)"
fi

VER="${JAM_VERSION:-0.1.0}"
printf 'Implementation-Title: jam\nImplementation-Version: %s\n' "$VER" > "$OUT/MANIFEST.MF"
jar --create --file "$OUT/jam.jar" --manifest "$OUT/MANIFEST.MF" -C "$CLASSES" .

echo "==> $OUT/jam.jar"
jar --list --file "$OUT/jam.jar" | grep -E 'native/.*(so|dylib|dll)$|\.class$' | sed 's/^/   /'
