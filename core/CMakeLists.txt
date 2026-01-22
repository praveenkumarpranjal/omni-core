cmake_minimum_required(VERSION 3.15)
project(omni VERSION 0.1.0 LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for Apple Silicon
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mcpu=apple-m1 -DNDEBUG -ffast-math")

# Source files
set(SOURCES
    src/model.cpp
    src/inference.cpp
    src/kernels.cpp
    src/quant.cpp
)

# Assembly files (if any)
file(GLOB ASM_SOURCES "asm/*.s")
if(ASM_SOURCES)
    enable_language(ASM)
    list(APPEND SOURCES ${ASM_SOURCES})
    message(STATUS "Found assembly files: ${ASM_SOURCES}")
endif()

# Build shared library
add_library(omni SHARED ${SOURCES})

# Include directories
target_include_directories(omni PUBLIC include)

# Link Apple Accelerate framework
find_library(ACCELERATE_FRAMEWORK Accelerate)
if(ACCELERATE_FRAMEWORK)
    target_link_libraries(omni ${ACCELERATE_FRAMEWORK})
    message(STATUS "Found Accelerate framework: ${ACCELERATE_FRAMEWORK}")
else()
    message(WARNING "Accelerate framework not found - performance will be degraded")
endif()

# Install targets
install(TARGETS omni DESTINATION lib)
install(FILES include/omni.h DESTINATION include)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS_RELEASE}")

