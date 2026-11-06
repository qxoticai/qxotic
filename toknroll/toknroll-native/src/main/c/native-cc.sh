#!/bin/sh
# The C compiler native-image runs for libtoknroll and libjinfer, by way of --native-compiler-path. GraalVM
# links a shared library with an export list naming its own runtime API (graal_*), its entry points and
# their stubs; at the link step this makes the list the C API alone: the global symbols of every *_shim.o
# on the line (toknroll_*, jinfer_*), plus libjam's names when its archive is linked in (jam's Java side
# resolves them with dlsym at run time). GraalVM links with -Bsymbolic-functions on Linux, and Mach-O binds
# two-level, so the library's own calls never reach a same-named symbol of another library.
# NATIVE_SONAME=libx.so.0.4.0 (libx.0.4.0.dylib on a Mac) is the soname (install name, under @rpath) of the
# shared library linked as libx.so (libx.dylib), and of no other: the Makefiles derive it from the version.
# NATIVE_CC_LOG=file appends every command line (static.sh replays the link from it); NATIVE_CC picks the compiler.
[ -n "$NATIVE_CC_LOG" ] && printf '%s\n' "$*" >> "$NATIVE_CC_LOG"
set -f   # the jam_* patterns below are for the linker, not the shell
darwin=; [ "$(uname -s)" = Darwin ] && darwin=yes
shims() { for o in "$@"; do case "$o" in *_shim.o) nm -g ${darwin:+-U} ${darwin:---defined-only} "$o" | awk '{print $3}' ;; esac; done; }
api=$(shims "$@")   # a function: bash 3.2, the Mac's sh, misparses a case inside $(); Mach-O names carry their underscore already
case " $* " in *libjam.a*) api="$api
${darwin:+_}jam_*
${darwin:+_}Java_com_qxotic_jam_libjam_*
${darwin:+_}JNI_OnLoad_jam" ;; esac
own=   # this is the link of the library NATIVE_SONAME names
if [ -n "$NATIVE_SONAME" ]; then
  if [ -n "$darwin" ]; then stem=${NATIVE_SONAME%%.*}.dylib; else stem=${NATIVE_SONAME%%.so.*}.so; fi
  case " $* " in *" -shared "*) case " $* " in *"/$stem "*) own=yes ;; esac ;; esac
fi
for arg in "$@"; do
  shift
  case "$arg" in
    -Wl,--whole-archive,*,--no-whole-archive)   # libjam in whole: the pom spells it for GNU ld, ld64 says -force_load
      [ -n "$darwin" ] && arg="-Wl,-force_load,${arg#-Wl,--whole-archive,}" && arg="${arg%,--no-whole-archive}" ;;
    -Wl,--version-script,*exported_symbols.list)   # GNU ld: a version script, the API global and all else local
      list=${arg##*,}
      { printf '{\nglobal:\n'; printf '%s;\n' $api; printf 'local: *;\n};\n'; } > "$list" ;;
    -Wl,*exported_symbols.list)   # ld64, after -Wl,-exported_symbols_list: one name per line
      list=${arg#-Wl,}
      printf '%s\n' $api > "$list" ;;
    -Wl,-install_name,*)   # GraalVM's own, the file's name: NATIVE_SONAME's below takes its place
      [ -n "$own" ] && continue ;;
  esac
  set -- "$@" "$arg"
done
if [ -n "$own" ]; then
  if [ -n "$darwin" ]; then set -- "$@" "-Wl,-install_name,@rpath/$NATIVE_SONAME"; else set -- "$@" "-Wl,-soname,$NATIVE_SONAME"; fi
fi
exec "${NATIVE_CC:-cc}" "$@"
