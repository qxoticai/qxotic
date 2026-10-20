#!/bin/sh
# The C compiler native-image runs, by way of --native-compiler-path. GraalVM links the shared
# library with a version script naming only its own entry points; at the link step this adds the
# shim's public API, toknroll_*, to that list, drops the toknroll__* entry points from it (the shim calls
# them inside the library; nothing outside may), and otherwise gets out of the way.
[ -n "$TOKNROLL_CC_LOG" ] && printf '%s\n' "$*" >> "$TOKNROLL_CC_LOG"
for arg in "$@"; do
  case "$arg" in
    -Wl,-exported_symbols_list,*)
      printf '_toknroll_[!_]*\n' >> "${arg##*,}" ;;   # macOS ld64: one symbol per line, underscore-prefixed
    -Wl,--version-script,*exported_symbols.list)
      sed -i -e '/^"toknroll__/d' -e 's/^local: \*;/toknroll_[!_]*;\nlocal: *;/' "${arg##*,}" ;;   # [!_]: toknroll__* stays local
  esac
done
exec "${TOKNROLL_CC:-cc}" "$@"
