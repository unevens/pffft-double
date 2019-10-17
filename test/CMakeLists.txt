cmake_minimum_required(VERSION 3.0.0)

project(pffft-double-test)

set(CMAKE_CXX_STANDARD 17)

add_executable(pffft-double-test ../pffft.c ../pffft-double.c pffft-double-test.cpp)

if (WIN32)
add_executable(pffft-double-test-no-avx ../pffft.c ../pffft-double.c pffft-double-test.cpp)
target_compile_options (pffft-double-test PRIVATE /arch:AVX)
target_compile_options (pffft-double-test-no-avx PRIVATE /arch:SSE2)
endif (WIN32)

if (UNIX)
target_compile_options (pffft-double-test PRIVATE -march:native)
endif (UNIX)
