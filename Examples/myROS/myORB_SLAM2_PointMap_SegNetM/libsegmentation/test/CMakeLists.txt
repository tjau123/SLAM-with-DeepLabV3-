cmake_minimum_required (VERSION 3.1.2)           # cmake的最低版本号
project (test)                                # 工程名

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g ")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
# Check C++11 or C++0x support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall")  # 设置编译器，这里设为C++编译器

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

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
        /media/hjx/Ubuntu 18.0/优盘/catkin_ws/src/DS-SLAM/Examples/myROS/ORB_SLAM2_PointMap_SegNetM/libsegmentation
)


#add_executable(${PROJECT_NAME}   main.cpp)                # 将源码编译为目标文件


#add_library(${PROJECT_NAME} SHARED main.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test
        main.cpp)
#target_link_libraries(test ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} /home/hjx/Programs/TensorFlow/bazel-bin/tensorflow/libtensorflow_cc.so
                    /home/hjx/Programs/TensorFlow/bazel-bin/tensorflow/libtensorflow_framework.so.1
        /media/hjx/Ubuntu 18.0/优盘/catkin_ws/src/DS-SLAM/Examples/myROS/ORB_SLAM2_PointMap_SegNetM/libsegmentation/build/libmysegmentation.so)




