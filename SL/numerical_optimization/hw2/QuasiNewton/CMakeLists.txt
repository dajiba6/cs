cmake_minimum_required(VERSION 3.10)
project(tutorial_1 VERSION 1.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置构建类型为 Debug
set(CMAKE_BUILD_TYPE Debug)

add_executable(test1 src/test1.cpp)

target_include_directories(test1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)