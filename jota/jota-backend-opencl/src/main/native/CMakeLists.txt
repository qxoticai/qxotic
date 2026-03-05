cmake_minimum_required(VERSION 3.16)

project(jota_opencl_jni LANGUAGES C)

find_package(JNI REQUIRED)
find_package(OpenCL REQUIRED)

add_library(jota_opencl SHARED jota_opencl_jni.c)

target_include_directories(jota_opencl PRIVATE ${JNI_INCLUDE_DIRS})
target_compile_definitions(jota_opencl PRIVATE CL_TARGET_OPENCL_VERSION=120)
target_link_libraries(jota_opencl PRIVATE OpenCL::OpenCL)

set_target_properties(jota_opencl PROPERTIES OUTPUT_NAME "jota_opencl")
