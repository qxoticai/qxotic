cmake_minimum_required(VERSION 3.16)

project(jota_metal_jni LANGUAGES CXX OBJCXX)

find_package(JNI REQUIRED)

if(NOT APPLE)
  message(FATAL_ERROR "Metal backend requires macOS")
endif()

add_library(jota_metal SHARED jota_metal_jni.mm)

target_include_directories(jota_metal PRIVATE ${JNI_INCLUDE_DIRS})
target_link_libraries(jota_metal PRIVATE "-framework Foundation" "-framework Metal")

set_target_properties(jota_metal PROPERTIES OUTPUT_NAME "jota_metal")
