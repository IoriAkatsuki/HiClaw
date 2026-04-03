cmake_minimum_required(VERSION 3.14)
project(venc_wrapper LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(ASCEND_ROOT "$ENV{ASCEND_TOOLKIT_HOME}")
if(NOT ASCEND_ROOT)
    set(ASCEND_ROOT /usr/local/Ascend/ascend-toolkit/latest)
endif()

if(EXISTS "${ASCEND_ROOT}/aarch64-linux")
    set(ASCEND_INC ${ASCEND_ROOT}/aarch64-linux/include)
    set(ASCEND_LIB ${ASCEND_ROOT}/aarch64-linux/lib64)
else()
    set(ASCEND_INC ${ASCEND_ROOT}/include)
    set(ASCEND_LIB ${ASCEND_ROOT}/lib64)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED
    PATHS /home/HwHiAiUser/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)

include_directories(${ASCEND_INC})
link_directories(${ASCEND_LIB})

add_library(venc_wrapper MODULE src/venc_wrap.cpp)

target_link_libraries(venc_wrapper
    PRIVATE
    pybind11::module
    acl_dvpp_mpi
    pthread
)

set_target_properties(venc_wrapper PROPERTIES PREFIX "" SUFFIX ".so")
