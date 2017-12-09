cmake_minimum_required(VERSION 3.8)
project(libparenth VERSION 0.1.0 LANGUAGES CXX)

# OPTIONS
option(BUILD_TESTS "Build unit tests" ON)

# Set the building options.
set(CMAKE_CXX_STANDARD 17)

include_directories(
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/deps/fbitset/include"
)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()

