set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/utils/python/llm_ops/lib)

file(GLOB OP_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/register.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp
)

set(OP_PLUGIN_NAME llm_ops)

add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        torch_npu
        ascendcl
        nnopbase
        opapi
        dl
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(_ASCEND_ARCH_INCLUDE ${ASCEND_HOME_PATH}/aarch64-linux/include)
    set(_ASCEND_ARCH_LIB ${ASCEND_HOME_PATH}/aarch64-linux/lib64)
else()
    set(_ASCEND_ARCH_INCLUDE ${ASCEND_HOME_PATH}/x86_64-linux/include)
    set(_ASCEND_ARCH_LIB ${ASCEND_HOME_PATH}/x86_64-linux/lib64)
endif()

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
        ${_ASCEND_ARCH_LIB}
        ${ASCEND_HOME_PATH}/lib64
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/utils
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include/third_party/acl/inc
        ${TORCH_NPU_DIR}/include/third_party/hccl/inc
        ${TORCH_NPU_DIR}/include
        ${PYTHON_INCLUDE_DIR}
        ${_ASCEND_ARCH_INCLUDE}
)

install(TARGETS ${OP_PLUGIN_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
