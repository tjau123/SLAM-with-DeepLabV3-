/*
hjx-2022-3-30-------------------------------------------------------------------
 */
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#include "/home/hjx/catkin_ws/src/MYSLAM/include/mySegment.h"
#include "Tracking.h"
#include "Camera.h"
#include <fstream>
#include <mutex>
#define SKIP_NUMBER 1
using namespace std;
using namespace cv;
void print_px_value(Mat& im)
{
    int rowNumber = im.rows;  //行数
    int colNumber = im.cols * im.channels();  //列数 x 通道数=每一行元素的个数

    //双重循环，遍历所有的像素值
    for (int i = 0; i < rowNumber; i++)  //行循环
    {
        uchar* data = im.ptr<uchar>(i);  //获取第i行的首地址
        for (int j = 0; j < colNumber; j++)   //列循环
        {
            //data[j] = data[j] / div * div + div / 2;
            cout << (int)data[j] <<" ";
        }  //行处理结束
    }
}
namespace ORB_SLAM2
{
mySegment::mySegment(const string &deeplab_model, const string &pascal_png) :mbFinishRequested(false),mSkipIndex(SKIP_NUMBER),
                 mSegmentTime(0),imgIndex(0)
{

    model_file = deeplab_model;
    LUT_file = pascal_png;
    label_colours = cv::imread(LUT_file,1);
    cv::cvtColor(label_colours, label_colours, CV_RGB2BGR);
    mImgSegmentLatest=cv::Mat(Camera::height,Camera::width,CV_8UC1);
    mbNewImgFlag=false;

}

void mySegment::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

bool mySegment::isNewImgArrived()
{
    unique_lock<std::mutex> lock(mMutexGetNewImg);
    if(mbNewImgFlag)
    {
        mbNewImgFlag=false;
        return true;
    }
    else
    return false;
}


void mySegment::Run()
{
    classifier=new Classifier(model_file, trained_file);
    cout << "Load deeplab model ..."<<endl;
    while(1)
    {

        usleep(1);
        if(!isNewImgArrived())
        continue;
//        cout<<"#############segment#################"<<endl;
        cout << "Wait for new RGB img time =" << endl;
        if(mSkipIndex==SKIP_NUMBER)
        {
            std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
            // Recognise by Semantin segmentation

            mImgSegment=classifier->Predict2(mImg, label_colours);
            cout<<"-------mImgSegment:"<<mImgSegment.size()<<endl ;
            cout<<"-------mImgSegment:"<<mImgSegment.channels()<<endl ;

            cout<<"#############segment-end#################"<<endl;

//            mImgSegment_color = classifier->Predict(mImg, label_colours);
            mImgSegment_color = mImgSegment.clone();
//            cv::cvtColor(mImgSegment,mImgSegment_color, CV_GRAY2BGR);
//            cout<<"-------mImgSegment_color:"<<mImgSegment_color.size()<<endl ;
//            cout<<"-------mImgSegment_color:"<<mImgSegment_color.channels()<<endl ;
            cout<<"************############color-end#################*****"<<endl;
//            LUT(mImgSegment_color, label_colours, mImgSegment_color_final);
            mImgSegment_color_final= classifier->Predict(mImg, label_colours);
            cout<<"-------mImgSegment_color_final:"<<mImgSegment_color_final.size()<<endl ;
            cout<<"-------mImgSegment_color_final:"<<mImgSegment_color_final.channels()<<endl ;


            cv::resize(mImgSegment, mImgSegment, cv::Size(Camera::width,Camera::height) );
            cv::resize(mImgSegment_color_final, mImgSegment_color_final, cv::Size(Camera::width,Camera::height) );
//            cv::imshow("mImgSegment_color_final",mImgSegment_color_final);
//            cv::waitKey(0);// 按任意键在0秒后退出窗口，不写这句话是不会显示出窗口的
//            cv::imshow("mImgSegment_color",mImgSegment_color);
//            cv::waitKey(0);// 按任意键在0秒后退出窗口，不写这句话是不会显示出窗口的
//            cv::imshow("mImgSegment",mImgSegment);
//            cv::waitKey(0);// 按任意键在0秒后退出窗口，不写这句话是不会显示出窗口的


            std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
            mSegmentTime+=std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
            mSkipIndex=0;
            imgIndex++;
        }
        mSkipIndex++;
        ProduceImgSegment();
        if(CheckFinish())
        {
            break;
        }

    }

}

bool mySegment::CheckFinish()
{
    unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}
  
void mySegment::RequestFinish()
{
    unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested=true;
}

void mySegment::ProduceImgSegment()
{
    std::unique_lock <std::mutex> lock(mMutexNewImgSegment);
    mImgTemp=mImgSegmentLatest;
    mImgSegmentLatest=mImgSegment;
    mImgSegment=mImgTemp;
    mpTracker->mbNewSegImgFlag=true;
}

}
