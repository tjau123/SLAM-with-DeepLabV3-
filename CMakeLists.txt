cmake_minimum_required(VERSION 2.8)
project(myORB_SLAM2_PointMap_SegNetM)

#include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
#rosbuild_init()

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

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include_directories($ {EIGEN3_INCLUDE_DIRS})

find_package(Eigen3  REQUIRED)
find_package(Pangolin REQUIRED)

# adding for point cloud viewer and mapper
find_package( PCL 1.7 REQUIRED )
find_package(catkin REQUIRED COMPONENTS 
  roscpp
  sensor_msgs
  message_generation
  tf
  )

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}

${Pangolin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}

/usr/local/include  
/usr/local/cuda/include
/usr/include  
${PROJECT_SOURCE_DIR}/Examples/myROS/myORB_SLAM2_PointMap_SegNetM/mylibsegmentation
${catkin_INCLUDE_DIRS}
/home/hjx/cuda-10.2/targets/x86_64-linux/include

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


#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED
src/Camera.cc
src/mySystem.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/PangolinViewer.cc
src/FrameDrawer.cc
src/MapDrawer.cc
src/pointcloudmapping.cc
src/mySegment.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PCL_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
/usr/lib/x86_64-linux-gnu/libglog.so
-lboost_system
${PROJECT_SOURCE_DIR}/Examples/myROS/myORB_SLAM2_PointMap_SegNetM/libsegmentation/build/libmysegmentation.so
${catkin_LIBRARIES}
)

