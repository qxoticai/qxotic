#!/bin/sh
# One leg of `make wheels`: the ref goes to the leg's host as a tarball, the tree's own targets build
# and check both wheels there (make wheels-linux inside the manylinux container, make wheels-native
# on the Mac, the two build.ps1 on Windows), and the wheels come back into target/wheels/.
#   HOST=<ssh host|localhost> [SETUP='<sh line run first>'] wheels.sh <leg> <ref>
set -eu
leg=$1 ref=$2 dir=qxotic-wheels
[ -n "$HOST" ] || { echo "wheels: $leg has no host in hosts.mk, skipped"; exit 0; }
echo "== $leg on $HOST ($ref)"
mkdir -p target/wheels target/legs
git archive --format=tar.gz --prefix="$dir/" "$ref" > "target/legs/$leg.tgz"

# what the host runs, in its login shell: sh, or PowerShell on Windows
case $leg in
  linux-*)   build="rm -rf $dir && tar -xzf $dir.tgz && cd $dir && make wheels-linux" ;;
  macos-*)   build="rm -rf $dir && tar -xzf $dir.tgz && cd $dir && ${SETUP:-true} && make wheels-native" ;;
  windows-*) build="rm -r -fo -ea 0 $dir; tar -xzf $dir.tgz; & ~\\$dir\\toknroll\\toknroll-native\\build.ps1 library check-wheel; & ~\\$dir\\jinfer\\jinfer-native\\build.ps1 library check-wheel" ;;
  *) echo "wheels: no such leg $leg"; exit 2 ;;
esac
wheels="$dir/toknroll/toknroll-native/target/wheel/*.whl $dir/jinfer/jinfer-native/target/wheel/*.whl"

if [ "$HOST" = localhost ]; then
  (cd target/legs && cp "$leg.tgz" "$dir.tgz" && sh -c "$build" && cp $wheels ../wheels/)
else
  scp -q "target/legs/$leg.tgz" "$HOST:$dir.tgz"
  ssh "$HOST" "$build"
  for w in $wheels; do scp -q "$HOST:$w" target/wheels/; done
fi
find target/wheels -name '*.whl' -newer "target/legs/$leg.tgz" -exec ls -la {} + | awk '{printf "%.1f MB %s\n", $5/1e6, $9}'
