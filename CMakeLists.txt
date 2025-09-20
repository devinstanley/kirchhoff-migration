cmake_minimum_required(VERSION 3.10.0)
project(seismic-migration VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Download Matplotlib Cpp
include(FetchContent)
FetchContent_Declare(matplotplusplus
    GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
    GIT_TAG origin/master
)
FetchContent_MakeAvailable(matplotplusplus)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(seismic-migration ${SOURCES})

# Include Directories
target_include_directories(seismic-migration 
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/optimization
        ${PROJECT_SOURCE_DIR}/include/linalg
)

target_link_libraries(seismic-migration PUBLIC matplot)

# Find OpenMP (Optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling support.")
    target_link_libraries(seismic-migration PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(seismic-migration PRIVATE HAVE_OPENMP)
    target_compile_options(seismic-migration PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
else()
    message(STATUS "OpenMP not found, building without OpenMP.")
endif()

## Find CUDA (Optional)
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found, enabling support.")

    # Add CUDA source files if they exist
    file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
    if(CUDA_SOURCES)
        target_sources(seismic-migration PRIVATE ${CUDA_SOURCES})
    endif()

    set_target_properties(seismic-migration PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_options(seismic-migration PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-arch=sm_80 -g -O0 --extended-lambda>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-arch=sm_80 -O3 --extended-lambda>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-arch=sm_80 -O3 -g --extended-lambda>)

    target_link_libraries(seismic-migration PRIVATE CUDA::cudart)
    target_compile_definitions(seismic-migration PRIVATE HAVE_CUDA)
else()
    message(STATUS "CUDA not found, building without CUDA.")
endif()

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)