#!/bin/sh
# libtoknroll.a: the link native-image just did, replayed as one relocatable object. The same image
# object and runtime archives, partially linked (only the archive members the image needs come in),
# then every symbol but the header's made local, so the archive exports the API and nothing of the
# runtime behind it, the toknroll__* entry points included. Consumers add the system libraries the
# image needs: -lz -ldl -lpthread -lm, then -lrt on Linux or -framework Foundation on macOS.
#   static.sh <toknroll.h> <cc log the wrapper wrote> <libtoknroll.a to write>
set -e
header=$1; log=$2; out=$3; work=$(dirname "$out")/toknroll_static.o
darwin=; case $(uname -s) in Darwin) darwin=yes ;; esac
grep -E '^TOKNROLL_API' "$header" | grep -oE 'toknroll_[a-z_]+\(' | tr -d '(' | sed "s/^/${darwin:+_}/" > "$work.keep"   # Mach-O names carry an underscore
line=$(grep -- ' -shared ' "$log" | tail -n 1)
[ -n "$line" ] || { echo "static.sh: no link line in $log" >&2; exit 1; }
tmp=""; inputs=""
for arg in $line; do
  case "$arg" in
    -L*SVM-*) tmp=${arg#-L} ;;                    # native-image's temp dir, where the image object is
  esac
done
for arg in $line; do
  case "$arg" in
    /*.o|/*.a) inputs="$inputs $arg" ;;
    *.o) inputs="$inputs $tmp/$arg" ;;           # the image object, named relative to the temp dir
  esac
done
if [ -n "$darwin" ]; then
  # ld-classic -d turns the JDK libraries' common symbols into definitions nmedit can then make local; the
  # new linker has no -d. ponytail: when Apple drops ld-classic, those commons (38 data names) stay global.
  ld=$(xcrun -f ld-classic 2>/dev/null) && ld="$ld -d" || ld=ld
  $ld -r -o "$work" $inputs
  nmedit -s "$work.keep" "$work"
else
  ld -r -o "$work" $inputs
  objcopy --keep-global-symbols="$work.keep" "$work"
fi
rm -f "$out" && ar rcs "$out" "$work" && rm -f "$work" "$work.keep"
echo "$out: $(nm --defined-only -g "$out" | grep -c ' T ') exported functions, $(du -h "$out" | cut -f1)"
