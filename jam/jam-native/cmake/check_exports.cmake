# ctest -R exports: the shared library exports exactly the functions jam.h declares (plus the JNI entry
# points when JNI=ON), nothing else. Run as cmake -DLIB=<lib> -DHEADER=<jam.h> -DJNI=<ON|OFF> -P <this>.
file(STRINGS "${HEADER}" api REGEX "^JAM_API ")
set(expected "")
foreach(line IN LISTS api)
    string(REGEX MATCH "(jam_[a-z_]+)\\(" _ "${line}")
    list(APPEND expected "${CMAKE_MATCH_1}")
endforeach()
if (JNI)
    list(APPEND expected JNI_OnLoad_jam Java_com_qxotic_jam_libjam_NativeJAM_newJni
                         Java_com_qxotic_jam_libjam_NativeJAM_freeJni Java_com_qxotic_jam_libjam_NativeJAM_mmJni)
endif()

if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    execute_process(COMMAND nm -gUj "${LIB}" OUTPUT_VARIABLE out RESULT_VARIABLE rc)
    string(REGEX REPLACE "(^|\n)_" "\\1" out "${out}")   # Mach-O spells C names with a leading _
else()
    execute_process(COMMAND nm -D --defined-only --format=just-symbols "${LIB}" OUTPUT_VARIABLE out RESULT_VARIABLE rc)
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
