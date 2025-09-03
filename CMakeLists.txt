cmake_minimum_required(VERSION 3.10.0)
project(seismic-migration VERSION 0.1.0 LANGUAGES C CXX)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Download Matplotlib Cpp
include(FetchContent)
FetchContent_Declare(
    matplotlib
    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
    GIT_TAG        f23347fca25219d1c42cbb91608b5556814bf572
)
FetchContent_GetProperties(matplotlib)
if(NOT matplotlib_POPULATED)
    FetchContent_Populate(matplotlib)
endif()

file(GLOB SOURCES "src/*.cpp")

add_executable(seismic-migration ${SOURCES})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Include Directories
target_include_directories(seismic-migration 
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${matplotlib_SOURCE_DIR}
)

# Link Python and NumPy
target_link_libraries(seismic-migration
    PRIVATE
        Python3::Python
        Python3::NumPy
)

# Add Python Include Directories
target_include_directories(seismic-migration
    PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${NumPy_INCLUDE_DIRS}
)

# Add Compiler Definitions
target_compile_definitions(seismic-migration
    PRIVATE
        MATPLOTLIBCPP_PYTHON_HEADER=Python.h
)