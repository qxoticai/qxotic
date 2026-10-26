#!/bin/sh
# The C compiler native-image runs, by way of --native-compiler-path. GraalVM links the shared
# library with a version script naming only its own entry points; at the link step this adds
# the shims' public API, jinfer_* and toknroll_*, to that list, drops the jinfer__* and toknroll__*
# entry points from it (the shims call them inside the library; nothing outside may), and otherwise
# gets out of the way.
[ -n "$JINFER_CC_LOG" ] && printf '%s\n' "$*" >> "$JINFER_CC_LOG"   # the exact compiler and linker lines, when asked
for arg in "$@"; do
  shift
  case "$arg" in
    -Wl,--whole-archive,*,--no-whole-archive)   # libjam in whole: GNU ld's spelling in the pom, ld64's here
      [ "$(uname -s)" = Darwin ] && arg="-Wl,-force_load,${arg#-Wl,--whole-archive,}" && arg="${arg%,--no-whole-archive}" ;;
    -Wl,--version-script,*exported_symbols.list)   # GNU ld
      # jinfer_* and toknroll_*: the API. jam_* and Java_com_qxotic_jam_libjam_*: libjam, linked in whole, which jam's
      # Java side resolves with dlsym at run time, so it must stay in the dynamic symbol table.
      sed -i -e '/^"\(jinfer\|toknroll\)__/d' -e 's/^local: \*;/jinfer_[!_]*;\ntoknroll_[!_]*;\njam_*;\nJava_com_qxotic_jam_libjam_*;\nJNI_OnLoad_jam;\nlocal: *;/' "${arg##*,}" ;;   # [!_]: the __ entry points stay local
    -Wl,*exported_symbols.list)   # ld64, the list follows -Wl,-exported_symbols_list: one underscore-prefixed name per line
      list=${arg#-Wl,}
      sed -i '' -e '/^_jinfer__/d' -e '/^_toknroll__/d' "$list"
      for o in "$@"; do case "$o" in *_shim.o) nm -gU "$o" | awk '{print $3}' >> "$list" ;; esac; done   # the shims' globals are the API
      printf '_jam_*\n_Java_com_qxotic_jam_libjam_*\n_JNI_OnLoad_jam\n' >> "$list" ;;
  esac
  set -- "$@" "$arg"
done
exec "${JINFER_CC:-cc}" "$@"
