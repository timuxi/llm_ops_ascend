cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(llm_ops_ascend LANGUAGES CXX)

option(BUILD_PYTHON "build python SDK" ON)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

add_compile_options(-Wno-unused-parameter -Wno-unused-function -Wunused-value -Wcast-align)
add_compile_options(-Wcast-qual -Winvalid-pch -Wwrite-strings -Wsign-compare -Wextra)

if (${CMAKE_BUILD_TYPE} MATCHES "RELEASE")
    add_compile_options(-O3)
    add_compile_options(-fvisibility=hidden -fvisibility-inlines-hidden)
    add_compile_options(-fstack-protector-strong)
    add_compile_options(-fPIE -fPIC)
    message(STATUS "build type set to RELEASE")
else ()
    add_compile_options(-g -rdynamic)
endif ()

if(DEFINED ENV{DEBUG_MODE})
    if("$ENV{DEBUG_MODE}" STREQUAL "ON")
        add_compile_definitions(DEBUG_MODE)
        message(STATUS "Debug logging enabled from environment")
    endif()
endif()

set(PROJECT_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)

include(utils/cmake/config_envs.cmake)

if(DEFINED ASCEND_HOME_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
endif()
if(NOT ASCEND_HOME_PATH)
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set")
endif()
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
message(STATUS "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

add_subdirectory(ops)
