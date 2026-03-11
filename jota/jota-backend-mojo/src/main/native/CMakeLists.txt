cmake_minimum_required(VERSION 3.16)

project(jota_mojo_jni C)

find_package(JNI REQUIRED)

set(_rocm_hint "")
if(ROCM_PATH)
  set(_rocm_hint "${ROCM_PATH}")
elseif(DEFINED ENV{ROCM_PATH})
  set(_rocm_hint "$ENV{ROCM_PATH}")
endif()

if(NOT HIP_INCLUDE_DIR)
  find_path(HIP_INCLUDE_DIR hip/hip_runtime.h
    HINTS
      ${_rocm_hint}
      /opt/rocm
    PATH_SUFFIXES include
  )
endif()
if(NOT AMD_HIP64_LIBRARY)
  find_library(AMD_HIP64_LIBRARY amdhip64
    HINTS
      ${_rocm_hint}
      /opt/rocm
    PATH_SUFFIXES lib lib64
  )
endif()

if(NOT HIP_INCLUDE_DIR)
  message(FATAL_ERROR "HIP_INCLUDE_DIR not found (set HIP_INCLUDE_DIR or ROCM_PATH)")
endif()
if(NOT AMD_HIP64_LIBRARY)
  message(FATAL_ERROR "AMD_HIP64_LIBRARY not found (set AMD_HIP64_LIBRARY or ROCM_PATH)")
endif()

add_library(jota_mojo SHARED jota_mojo_jni.c)

target_compile_definitions(jota_mojo PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(jota_mojo PRIVATE ${JNI_INCLUDE_DIRS} ${HIP_INCLUDE_DIR})
target_link_libraries(jota_mojo PRIVATE ${AMD_HIP64_LIBRARY})

if(UNIX AND NOT APPLE)
  get_filename_component(HIP_LIB_DIR ${AMD_HIP64_LIBRARY} DIRECTORY)
  set_target_properties(jota_mojo PROPERTIES BUILD_RPATH "${HIP_LIB_DIR}")
endif()

set_target_properties(jota_mojo PROPERTIES OUTPUT_NAME "jota_mojo")
