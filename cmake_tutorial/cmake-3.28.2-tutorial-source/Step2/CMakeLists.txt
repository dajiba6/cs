cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(Tutorial VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file(TutorialConfig.h.in TutorialConfig.h)

# TODO 2: Use add_subdirectory() to add MathFunctions to this project
# 告诉cmake层级下面还有个cmake要编译
add_subdirectory(MathFunctions)
# add the executable
add_executable(Tutorial tutorial.cxx)

# TODO 3: Use target_link_libraries to link the library to our executable
# 将库链接到可执行文件，在链接阶段生效
target_link_libraries(Tutorial PUBLIC MathFunctions)
# TODO 4: Add MathFunctions to Tutorial's target_include_directories()
# Hint: ${PROJECT_SOURCE_DIR} is a path to the project source. AKA This folder!
# 确保tutorial在编译阶段能找到库的头文件
target_include_directories(Tutorial PUBLIC
# 加一个"${PROJECT_BINARY_DIR}"是因为里面会有由.h.in文件生成的.h文件，如step1中的版本号等信息
                           "${PROJECT_BINARY_DIR}"
                           "${PROJECT_SOURCE_DIR}/MathFunctions"
                           )
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h

