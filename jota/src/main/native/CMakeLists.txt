cmake_minimum_required(VERSION 3.16)

project(jota_c_jni C)

find_package(JNI REQUIRED)

add_library(jota_c SHARED jota_c_jni.c)

target_include_directories(jota_c PRIVATE ${JNI_INCLUDE_DIRS})

if(UNIX AND NOT APPLE)
  target_link_libraries(jota_c PRIVATE dl)
endif()

set_target_properties(jota_c PROPERTIES OUTPUT_NAME "jota_c")
