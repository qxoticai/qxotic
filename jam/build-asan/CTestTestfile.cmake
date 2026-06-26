# CMake generated Testfile for 
# Source directory: /home/mukel/Desktop/playground/qxotic/jam
# Build directory: /home/mukel/Desktop/playground/qxotic/jam/build-asan
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(correctness "/home/mukel/Desktop/playground/qxotic/jam/build-asan/jam_test")
set_tests_properties(correctness PROPERTIES  _BACKTRACE_TRIPLES "/home/mukel/Desktop/playground/qxotic/jam/CMakeLists.txt;166;add_test;/home/mukel/Desktop/playground/qxotic/jam/CMakeLists.txt;0;")
add_test(correctness_env_cap "/home/mukel/Desktop/playground/qxotic/jam/build-asan/jam_test")
set_tests_properties(correctness_env_cap PROPERTIES  ENVIRONMENT "JAM_ISA=avx2;JAM_NUM_THREADS=2" _BACKTRACE_TRIPLES "/home/mukel/Desktop/playground/qxotic/jam/CMakeLists.txt;167;add_test;/home/mukel/Desktop/playground/qxotic/jam/CMakeLists.txt;0;")
