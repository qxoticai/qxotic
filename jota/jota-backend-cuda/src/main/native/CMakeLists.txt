cmake_minimum_required(VERSION 3.21)

project(jota_cuda_jni LANGUAGES C)

if(APPLE)
  message(STATUS "CUDA backend is unsupported on macOS; skipping native target")
  return()
endif()

find_package(JNI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(jota_cuda SHARED jota_cuda_jni.c)
target_include_directories(jota_cuda PRIVATE ${JNI_INCLUDE_DIRS})
target_link_libraries(jota_cuda PRIVATE CUDA::cudart CUDA::cuda_driver)
set_target_properties(jota_cuda PROPERTIES OUTPUT_NAME "jota_cuda")
