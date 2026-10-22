#!/bin/sh
# The C compiler native-image runs, by way of --native-compiler-path. GraalVM links the shared
# library with a version script naming only its own entry points; at the link step this adds
# the shims' public API, jinfer_* and toknroll_*, to that list, drops the jinfer__* and toknroll__*
# entry points from it (the shims call them inside the library; nothing outside may), and otherwise
# gets out of the way.
[ -n "$JINFER_CC_LOG" ] && printf '%s\n' "$*" >> "$JINFER_CC_LOG"   # the exact compiler and linker lines, when asked
for arg in "$@"; do
  case "$arg" in
    -Wl,-exported_symbols_list,*)
      # macOS: ld64 takes one symbol per line, underscore-prefixed, wildcards allowed. Untested until
      # the Mac that builds the jam natives has a GraalVM.
      printf '_jinfer_[!_]*\n_toknroll_[!_]*\n_jam_*\n_Java_com_qxotic_jam_libjam_*\n_JNI_OnLoad_jam\n' >> "${arg##*,}" ;;
    -Wl,--version-script,*exported_symbols.list)
      # jinfer_* and toknroll_*: the API. jam_* and Java_com_qxotic_jam_libjam_*: libjam, linked in whole, which jam's
      # Java side resolves with dlsym at run time, so it must stay in the dynamic symbol table.
      sed -i -e '/^"\(jinfer\|toknroll\)__/d' -e 's/^local: \*;/jinfer_[!_]*;\ntoknroll_[!_]*;\njam_*;\nJava_com_qxotic_jam_libjam_*;\nJNI_OnLoad_jam;\nlocal: *;/' "${arg##*,}" ;;   # [!_]: the __ entry points stay local
  esac
done
exec "${JINFER_CC:-cc}" "$@"
