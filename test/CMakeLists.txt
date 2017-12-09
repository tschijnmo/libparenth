# Download the Catch2 single header.

file(DOWNLOAD
        "https://github.com/catchorg/Catch2/releases/download/v2.0.1/catch.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/catch.hpp"
        )

include_directories(testmain
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        )

# The main test driver.
add_executable(testmain
        testmain.cpp
        )
