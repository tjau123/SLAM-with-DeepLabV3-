/*
 hjx-2020-3-30
 -------------------------------------------------------------------------------------------------
 */

#ifndef MYSEGMENT_H
#define MYSEGMENT_H

#include "KeyFrame.h"
#include "Map.h"
#include "Tracking.h"
#include "/home/hjx/catkin_ws/src/MYSLAM/Examples/myROS/myORB_SLAM2_PointMap_SegNetM/libsegmentation/mylibsegmentation.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>

namespace ORB_SLAM2
{

class Tracking;

class mySegment
{

public:
    mySegment(const string &deeplab_model, const string &pascal_png);
    void SetTracker(Tracking* pTracker);
    void Run();
    int conbase = 64, jinzhi=4;
    int labeldata[20]={32,8,40,2,34,10,42,16,48,24,56,18,50,26,58,4,36,12,44,6};

    cv::Mat label_colours;
    Classifier* classifier;
    bool isNewImgArrived();
    bool CheckFinish();
    void RequestFinish();
    void Initialize(const cv::Mat& img);
    cv::Mat mImg;
    cv::Mat mImgTemp;
    cv::Mat mImgSegment_color;
    cv::Mat mImgSegment_color_final;
    cv::Mat mImgSegment;
    cv::Mat mImgSegmentLatest;
    Tracking* mpTracker;
    std::mutex mMutexGetNewImg;
    std::mutex mMutexFinish;
    bool mbFinishRequested;
    void ProduceImgSegment();
    std::mutex mMutexNewImgSegment;
    std::condition_variable mbcvNewImgSegment;
    bool mbNewImgFlag;
    int mSkipIndex;
    double mSegmentTime;
    int imgIndex;
    // Paremeters for caffe
    string model_file;
    string trained_file;
    string LUT_file;
};

}



#endif
