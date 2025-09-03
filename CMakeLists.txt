cmake_minimum_required(VERSION 3.10.0)
project(seismic-migration VERSION 0.1.0 LANGUAGES C CXX)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Download Matplotlib Cpp
include(FetchContent)
FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master)

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
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
)

target_link_libraries(seismic-migration PUBLIC matplot)