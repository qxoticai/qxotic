#!/usr/bin/env bash
# Assemble the standalone fat jam.jar = compiled Java from ALL jam modules + the shipped native
# libraries staged under jam-native/dist/release/ by `scripts/natives.sh build`.
#
# Java is compiled by Maven under the release profile, the same shape the Central artifacts get:
# no cmake for this host, `natives.sh verify` on the staged set, and jam-native's target/classes
# holding exactly dist/release. The four modules' classes are then merged into one jar.
set -euo pipefail
cd "$(dirname "$0")/.."          # -> jam-native
ROOT=..                          # jam reactor root (siblings: jam-core, jam-scalar, jam-vector)
OUT=dist
CLASSES="$OUT/classes"

echo "==> mvn -Prelease package (all jam modules; verifies + packages the staged release set)"
( cd "$ROOT" && mvn -q -Prelease -DskipTests -Dgpg.skip=true package )

echo "==> gathering classes + native libs"
rm -rf "$CLASSES"; mkdir -p "$CLASSES"
for m in "$ROOT/jam-core" "$ROOT/jam-scalar" "$ROOT/jam-vector" .; do
    cp -a "$m/target/classes/." "$CLASSES/"
done

VER="${JAM_VERSION:-0.2.0}"
printf 'Implementation-Title: jam\nImplementation-Version: %s\n' "$VER" > "$OUT/MANIFEST.MF"
jar --create --file "$OUT/jam.jar" --manifest "$OUT/MANIFEST.MF" -C "$CLASSES" .

echo "==> $OUT/jam.jar ($(du -h "$OUT/jam.jar" | cut -f1))"
jar --list --file "$OUT/jam.jar" | grep -E 'native/.*(so|dylib|dll)$|\.class$' | sed 's/^/   /'
