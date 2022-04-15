//
// Created by hjx on 2022/3/25.
//

#ifndef TF_TEST_MYLIBSEGMENTION_H
#define TF_TEST_MYLIBSEGMENTION_H
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

//using namespace caffe;

using namespace tensorflow;

//void tesor2Mat(&t,);

class Classifier
{
public:
    Classifier(const string& model_file,
               const string& trained_file);

    cv::Mat Predict(const cv::Mat& img,  cv::Mat LUT_image);
    cv::Mat Predict2(const cv::Mat& img,  cv::Mat LUT_image);

private:
    void SetMean(const string& mean_file);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);

private:
   // boost::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    cv::Size output_geometry_;
    int num_channels_;
    string model_path="/home/hjx/models-master/research/deeplab/exp/voc_train/train1025/export/frozen_inference_graph.pb";;
//    string model_path="/home/hjx/pb/frozen_inference_graph.pbpython";;
    string input_name="ImageTensor";
    string output_name="SemanticPredictions";
    const int Hei=224;
    const int Wid=224;

};

#endif //TF_TEST_MYLIBSEGMENTION_H

