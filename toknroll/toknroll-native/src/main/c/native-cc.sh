#!/bin/sh
# The C compiler native-image runs for libtoknroll and libjinfer, by way of --native-compiler-path. GraalVM
# links a shared library with an export list naming only the image's own entry points; at the link step this
# makes the list the C API instead: the global symbols of every *_shim.o on the line (toknroll_*, jinfer_*),
# minus the private toknroll__*/jinfer__* entry points the shims call, plus libjam's names when its archive
# is linked in (jam's Java side resolves them with dlsym at run time). Otherwise it gets out of the way.
# NATIVE_CC_LOG=file appends every command line (static.sh replays the link from it); NATIVE_CC picks the compiler.
[ -n "$NATIVE_CC_LOG" ] && printf '%s\n' "$*" >> "$NATIVE_CC_LOG"
darwin=; [ "$(uname -s)" = Darwin ] && darwin=yes
shims() { for o in "$@"; do case "$o" in *_shim.o) nm -g ${darwin:+-U} ${darwin:---defined-only} "$o" | awk '{print $3}' ;; esac; done; }
api=$(shims "$@")   # a function: bash 3.2, the Mac's sh, misparses a case inside $(); Mach-O names carry their underscore already
case " $* " in *libjam.a*) api="$api
${darwin:+_}jam_*
${darwin:+_}Java_com_qxotic_jam_libjam_*
${darwin:+_}JNI_OnLoad_jam" ;; esac
for arg in "$@"; do
  shift
  case "$arg" in
    -Wl,--whole-archive,*,--no-whole-archive)   # libjam in whole: the pom spells it for GNU ld, ld64 says -force_load
      [ -n "$darwin" ] && arg="-Wl,-force_load,${arg#-Wl,--whole-archive,}" && arg="${arg%,--no-whole-archive}" ;;
    -Wl,--version-script,*exported_symbols.list)   # GNU ld: quoted names under global:, then local: *;
      list=${arg##*,}
      awk -v api="$(printf '%s;\n' $api)" '/^"[a-z][a-z]*__/ {next} /^local: \*;/ {print api} {print}' "$list" > "$list.new" && mv "$list.new" "$list" ;;
    -Wl,*exported_symbols.list)   # ld64, after -Wl,-exported_symbols_list: one name per line
      list=${arg#-Wl,}
      { grep -v '^_[a-z][a-z]*__' "$list"; printf '%s\n' $api; } > "$list.new" && mv "$list.new" "$list" ;;
  esac
  set -- "$@" "$arg"
done
exec "${NATIVE_CC:-cc}" "$@"
