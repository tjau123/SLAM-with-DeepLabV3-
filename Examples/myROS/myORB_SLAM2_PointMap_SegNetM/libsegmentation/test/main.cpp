#include "/media/hjx/Ubuntu 18.0/优盘/catkin_ws/src/DS-SLAM/Examples/myROS/ORB_SLAM2_PointMap_SegNetM/libsegmentation/mylibsegmentation.hpp"

#include <iostream>
#include <fstream>
#include<vector>
#include<ctime>
#include <opencv2/opencv.hpp>					// 使用opencv读取图片
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <boost/shared_ptr.hpp>

using namespace std;
using namespace tensorflow;
using namespace cv;

 int main(int argc,char **argv){
    string deeplab_model;
    deeplab_model=argv[1];
    cv::Mat img;
    img =cv::imread(argv[2]);
    Classifier* deeplab;
    cv::Mat LUT_img;
    cv::Mat preimg =deeplab->Predict(img,LUT_img);

    cv::imshow("predict",preimg);
    cv::waitKey(0);// 按任意键在0秒后退出窗口，不写这句话是不会显示出窗口的
     return 0;
}