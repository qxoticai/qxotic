cmake_minimum_required(VERSION 3.16)

project(jota_hip_jni C)

find_package(JNI REQUIRED)

if(NOT HIP_INCLUDE_DIR)
  message(FATAL_ERROR "HIP_INCLUDE_DIR not set")
endif()
if(NOT AMD_HIP64_LIBRARY)
  message(FATAL_ERROR "AMD_HIP64_LIBRARY not set")
endif()

add_library(jota_hip SHARED jota_hip_jni.c)

target_compile_definitions(jota_hip PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(jota_hip PRIVATE ${JNI_INCLUDE_DIRS} ${HIP_INCLUDE_DIR})
target_link_libraries(jota_hip PRIVATE ${AMD_HIP64_LIBRARY})

if(UNIX AND NOT APPLE)
  get_filename_component(HIP_LIB_DIR ${AMD_HIP64_LIBRARY} DIRECTORY)
  set_target_properties(jota_hip PROPERTIES BUILD_RPATH "${HIP_LIB_DIR}")
endif()

set_target_properties(jota_hip PROPERTIES OUTPUT_NAME "jota_hip")
