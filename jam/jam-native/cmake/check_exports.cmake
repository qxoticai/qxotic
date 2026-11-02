# ctest -R exports: the shared library exports exactly the functions jam.h declares (plus the JNI entry
# points jam_jni.c defines, when the library carries it), nothing else. Run as
# cmake -DLIB=<lib> -DHEADER=<jam.h> [-DJNI_SOURCE=<jam_jni.c>] -P <this>.
file(STRINGS "${HEADER}" api REGEX "^JAM_API ")
set(expected "")
foreach(line IN LISTS api)
    string(REGEX MATCH "(jam_[a-z0-9_]+)\\(" _ "${line}")
    list(APPEND expected "${CMAKE_MATCH_1}")
endforeach()
if (JNI_SOURCE)
    file(STRINGS "${JNI_SOURCE}" jni REGEX "^(Java|JNI_OnLoad)_[A-Za-z0-9_]+\\(")
    foreach(line IN LISTS jni)
        string(REGEX MATCH "^[A-Za-z0-9_]+" name "${line}")
        list(APPEND expected "${name}")
    endforeach()
endif()

if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    execute_process(COMMAND nm -gUj "${LIB}" OUTPUT_VARIABLE out RESULT_VARIABLE rc)
    string(REGEX REPLACE "(^|\n)_" "\\1" out "${out}")   # Mach-O spells C names with a leading _
else()
    # POSIX format, "name type value size" per line: older binutils have no --format=just-symbols
    execute_process(COMMAND nm -D --defined-only -P "${LIB}" OUTPUT_VARIABLE out RESULT_VARIABLE rc)
    string(REGEX REPLACE " [^\n]*" "" out "${out}")
endif()
if (NOT rc EQUAL 0)
    message(FATAL_ERROR "nm failed on ${LIB}")
endif()
string(REPLACE "\n" ";" exported "${out}")
list(FILTER exported EXCLUDE REGEX "^$|^_init$|^_fini$|^_edata$|^_end$|^__bss_start$")   # the linker's own

list(SORT expected)
list(SORT exported)
if (NOT expected STREQUAL exported)
    set(extra ${exported})
    list(REMOVE_ITEM extra ${expected})
    set(missing ${expected})
    list(REMOVE_ITEM missing ${exported})
    message(FATAL_ERROR "exports differ from jam.h\n  not in the header: ${extra}\n  not exported: ${missing}")
endif()
list(LENGTH expected n)
message(STATUS "exports match jam.h: ${n} symbols")
