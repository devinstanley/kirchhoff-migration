cmake_minimum_required(VERSION 3.10.0)
project(seismic-migration VERSION 0.1.0 LANGUAGES C CXX)

file(GLOB SOURCES "src/*.cpp")

add_executable(seismic-migration ${SOURCES})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_include_directories(seismic-migration PRIVATE ${PROJECT_SOURCE_DIR}/include)
