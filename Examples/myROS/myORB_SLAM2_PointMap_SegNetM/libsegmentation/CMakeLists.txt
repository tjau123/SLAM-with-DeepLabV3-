cmake_minimum_required (VERSION 2.8)           # cmake的最低版本号
project (mysegmentation)                                # 工程名


SET (SRC_LIST mylibsegmentation.cpp) 
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall")  # 设置编译器，这里设为C++编译器




find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

link_directories(/home/hjx/Programs/TensorFlow/bazel-bin/tensorflow)   # 链接库搜索目录
link_libraries("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
include_directories(  
        ${PROJECT_SOURCE_DIR}                    # 头文件的搜索目录
        /home/hjx/Programs/TensorFlow/
        /home/hjx/Programs/TensorFlow/bazel-genfiles
        /home/hjx/Programs/TensorFlow/bazel-bin/tensorflow
        /home/hjx/Programs/TensorFlow/tensorflow/contrib/makefile/downloads/nsync/public
        /home/hjx/Programs/TensorFlow/tensorflow/contrib/makefile/downloads/absl
        /home/hjx/Programs/TensorFlow/tensorflow/contrib/makefile/gen/protobuf/include
        /usr/local/include/eigen3
        /home/hjx/Programs/TensorFlow/tensorflow/contrib/makefile/downloa/absl
        /home/hjx/Programs/TensorFlow/tensorflow/contrib/makefile/downloa/protobuf/src
        )

#add_executable(${PROJECT_NAME}   mylibsegmentation.hpp mylibsegmentation.cpp)                # 将源码编译为目标文件


add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} /home/hjx/Programs/TensorFlow/bazel-bin/tensorflow/libtensorflow_cc.so
                    /home/hjx/Programs/TensorFlow/bazel-bin/tensorflow/libtensorflow_framework.so.1)




