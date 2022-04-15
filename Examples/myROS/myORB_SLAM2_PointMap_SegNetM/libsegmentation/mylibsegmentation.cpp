//
// Created by hjx on 2022/3/25.
//
#include"mylibsegmentation.hpp"
#include <chrono>
//#include <caffe/caffe.hpp>

using namespace tensorflow;
using namespace std;
using namespace cv;
const int hei=224;
const int wid=224;

void cvMat2tfTensor(cv::Mat input, tensorflow::Tensor& outputTensor)    //  mat 转 tensor  函数
{
    auto outputTensorMapped = outputTensor.tensor<unsigned char, 4>();

    input.convertTo(input, CV_32FC3);
    cv::resize(input, input, cv::Size(hei, wid));
    int height = input.size().height;
    int width = input.size().width;
    int depth = input.channels();
    const float* data = (float*)input.data;
    for (int y = 0; y < height; ++y)
    {
        const float* dataRow = data + (y * width * depth);
        for (int x = 0; x < width; ++x)
        {
            const float* dataPixel = dataRow + (x * depth);
            for (int c = 0; c < depth; ++c)
            {
                const float* dataValue = dataPixel + c;
                outputTensorMapped(0, y, x, c) = *dataValue;
            }
        }
    }
}

cv::Mat read_image(string image_path)    //  读取照片函数
{
    // 设置输入图像数据
    //利用opencv读取图片
    cv::Mat bgrImage, rgbImage, out_rgbImage;
    bgrImage = cv::imread(image_path);
    cv::cvtColor(bgrImage, rgbImage, cv::COLOR_BGR2RGB);    // 将图片由BGR 转成 RGB

    rgbImage.convertTo(rgbImage, CV_32F);     // 将图片转换成float格式的

    //		cv::INTER_LINEAR 双线性插值  进行图片大小等比例转换
    cv::resize(rgbImage, out_rgbImage, cv::Size(hei, wid), cv::INTER_LINEAR);  // 1600*789 是我的pb模型训练时候的inputsize
    return out_rgbImage;
}

class Classifier;
Classifier::Classifier(const string &model_file, const string &trained_file) {
model_path=model_file;

}
cv::Mat graymap(tensorflow::Tensor t) {

    static int colormap[wid][hei];

    auto out_shape = t.shape();
    auto out_val = t.tensor<int, 3>();
    for (int i = 0; i < out_shape.dim_size(1); i++) {
        for (int j = 0; j < out_shape.dim_size(2); j++) {
//            cout<<"out_val(0, i, j):"<<out_val(0, i, j)<<endl;
            colormap[i][j] = out_val(0, i, j);
        }
    }

//    cv::Mat outputrgbImg = cv::Mat(wid, hei, CV_8UC2, cv::Scalar(255, 255));
    cv::Mat outputrgbImg = cv::Mat(wid,hei,CV_8U,colormap);
    return outputrgbImg;

}



cv::Mat labelcolor_map(tensorflow::Tensor t)
{
    static int colormap[3][wid][hei];

    auto out_shape = t.shape();
    auto out_val = t.tensor<int, 3>();

    for (int i = 0; i < out_shape.dim_size(1); i++) {
        for (int j = 0; j < out_shape.dim_size(2); j++) {
            switch (out_val(0, i, j))
            {
                case 0:
                    colormap[0][i][j] = 0;
                    colormap[1][i][j] = 0;
                    colormap[2][i][j] = 0;			// 标签为0
                    break;
                case 1:
                    colormap[0][i][j] = 128;
                    colormap[1][i][j] = 0;
                    colormap[2][i][j] = 0;			// 标签为1
                    break;
                case 2:
                    colormap[0][i][j] = 0;			// 标签为2
                    colormap[1][i][j] = 128;
                    colormap[2][i][j] = 0;
                    break;
                case 3:
                    colormap[0][i][j] = 128;		//标签为3
                    colormap[1][i][j] = 128;
                    colormap[2][i][j] = 0;
                    break;
                case 4:
                    colormap[0][i][j] = 0;
                    colormap[1][i][j] = 0;
                    colormap[2][i][j] = 128;			// 标签为4
                    break;
                case 5:
                    colormap[0][i][j] = 128;
                    colormap[1][i][j] = 0;
                    colormap[2][i][j] = 128;			// 标签为5
                    break;
                case 6:
                    colormap[0][i][j] = 0;			// 标签为6
                    colormap[1][i][j] = 128;
                    colormap[2][i][j] = 128;
                    break;
                case 7:
                    colormap[0][i][j] = 128;		//标签为7
                    colormap[1][i][j] = 128;
                    colormap[2][i][j] = 128;
                    break;
                case 8:
                    colormap[0][i][j] = 64;
                    colormap[1][i][j] = 0;
                    colormap[2][i][j] = 0;			// 标签为8
                    break;
                case 9:
                    colormap[0][i][j] = 192;
                    colormap[1][i][j] = 0;
                    colormap[2][i][j] = 0;			// 标签为9
                    break;
                case 10:
                    colormap[0][i][j] = 64;			// 标签为10
                    colormap[1][i][j] = 128;
                    colormap[2][i][j] = 0;
                    break;
                case 11:
                    colormap[0][i][j] = 192;		//标签为11
                    colormap[1][i][j] = 128;
                    colormap[2][i][j] = 0;
                    break;
                case 12:
                    colormap[0][i][j] = 64;		//标签为12
                    colormap[1][i][j] = 128;
                    colormap[2][i][j] = 0;
                    break;
                case 13:
                    colormap[0][i][j] = 192;
                    colormap[1][i][j] = 0;
                    colormap[2][i][j] = 128;			// 标签为13
                    break;
                case 14:
                    colormap[0][i][j] = 64;
                    colormap[1][i][j] = 128;
                    colormap[2][i][j] = 128;			// 标签为14
                    break;
                case 15:
                    colormap[0][i][j] = 192;			// 标签为15
                    colormap[1][i][j] = 128;
                    colormap[2][i][j] = 128;
                    break;
                case 16:
                    colormap[0][i][j] = 0;		//标签为16
                    colormap[1][i][j] = 64;
                    colormap[2][i][j] = 0;
                    break;
                case 17:
                    colormap[0][i][j] = 128;
                    colormap[1][i][j] = 64;
                    colormap[2][i][j] = 0;			// 标签为17
                    break;
                case 18:
                    colormap[0][i][j] = 0;
                    colormap[1][i][j] = 192;
                    colormap[2][i][j] = 0;			// 标签为18
                    break;
                case 19:
                    colormap[0][i][j] = 128;			// 标签为19
                    colormap[1][i][j] = 192;
                    colormap[2][i][j] = 0;
                    break;
                case 20:
                    colormap[0][i][j] = 0;
                    colormap[1][i][j] = 64;
                    colormap[2][i][j] = 128;			// 标签为20
                    break;




            }
        }
    }



    cv::Mat outputrgbImg = cv::Mat(wid, hei, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 0; i < wid; i++)
    {
        for (int j = 0; j < hei; j++)
        {
            outputrgbImg.at<Vec3b>(i, j)[0] = colormap[2][i][j];   //	B
            outputrgbImg.at<Vec3b>(i, j)[1] = colormap[1][i][j];   //  G
            outputrgbImg.at<Vec3b>(i, j)[2] = colormap[0][i][j];   //  R
        }
    }
    return outputrgbImg;
}

void tensor2Mat(Tensor &t, cv::Mat &image) {
    auto output = t.tensor<int, 3>();  // (1,512,512)
    for (int i = 0; i < image.cols; ++i) {
        for (int j = 0; j < image.rows; ++j) {
            image.at<uchar>(i, j) = output(0, i, j);
        }
    }
}

//void tensor2Mat(Tensor &t, Mat &image) {
//    image.convertTo(image, CV_32FC1);
//    tensorflow::StringPiece tmp_data = t.tensor_data();
//    memcpy(image.data,const_cast<char*>(tmp_data.data()),hei * wid * sizeof(float));
//    image.convertTo(image, CV_8UC1);
//}

cv::Mat Classifier::Predict(const cv::Mat &img, cv::Mat LUT_image) {
    int channels = img.channels();
    tensorflow::Tensor input_tensor(DT_UINT8, TensorShape({1, Wid, Hei, channels}));
    cvMat2tfTensor(img, input_tensor);

    GraphDef graph_def;
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), model_path, &graph_def));
    std::unique_ptr<tensorflow::Session> session(
            tensorflow::NewSession(tensorflow::SessionOptions())
    );
    TF_CHECK_OK(session->Create(graph_def));
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs =
            {{input_name, input_tensor}};
    std::vector<tensorflow::Tensor> outputs;

    TF_CHECK_OK(session->Run(inputs, {output_name}, {}, &outputs));
    Tensor t = outputs[0];

//    cv::Mat prediction_map;
//    prediction_map.convertTo(prediction_map,CV_32FC1);
//    tensorflow::StringPiece tem_data=t.tensor_data();
//    memcpy(prediction_map.data,const_cast<char*>(tem_data.data()),640*480*sizeof (float));
//    prediction_map.convertTo(prediction_map,CV_8UC1);
//    return prediction_map;

//    cv::Mat tres;
//    const Tensor& encoded_locations = t;
//    auto input_map = encoded_locations.tensor<int, 3>();
//
//    int width =480, height = 360;
//
//    for (int y = 0; y < height; ++y) {
//        for (int x = 0; x < width; ++x) {
//            tres.at<int>(y, x) = input_map(y, x, 0);
//        }
//    }
//    return tres;


//
//    static int c[224][224];
//    auto out_shape = t.shape();
//    auto out_val = t.tensor<int, 3>();
//
//    for (int i = 0; i < out_shape.dim_size(1); i++) {
//        for (int j = 0; j < out_shape.dim_size(2); j++) {
//            c[i][j]=out_val(0, i, j);
//            //c[i][j]=c[i][j]*5;
//        }
//    }
//    cv::Mat color(224, 224, CV_8UC1, c);
//    return color;

    cv::Mat colored_img = labelcolor_map(t);
    return colored_img;

}

cv::Mat Classifier::Predict2(const cv::Mat &img, cv::Mat LUT_image) {
    int channels = img.channels();
    tensorflow::Tensor input_tensor(DT_UINT8, TensorShape({1, Wid, Hei, channels}));
    cvMat2tfTensor(img, input_tensor);

    GraphDef graph_def;
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), model_path, &graph_def));
    std::unique_ptr<tensorflow::Session> session(
            tensorflow::NewSession(tensorflow::SessionOptions())
    );
    TF_CHECK_OK(session->Create(graph_def));
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs =
            {{input_name, input_tensor}};
    std::vector<tensorflow::Tensor> outputs;

    TF_CHECK_OK(session->Run(inputs, {output_name}, {}, &outputs));
    Tensor t = outputs[0];

//    static int c[224][224];
//    auto out_shape = t.shape();
//    auto out_val = t.tensor<int, 3>();
//
//    for (int i = 0; i < out_shape.dim_size(1); i++) {
//        for (int j = 0; j < out_shape.dim_size(2); j++) {
//            c[i][j]=out_val(0, i, j);
//            //c[i][j]=c[i][j]*5;
//        }
//    }
//    cv::Mat color(224, 224, CV_8UC1, c);
//    return color;

    cv::Mat gray= graymap(t);
    return gray;

//    cv::Mat colored_img = labelcolor_map(t);
//    return colored_img;

}

//    Mat colored_map;
//
//    int *p = t.flat<int>().data();
//    colored_map = Mat(480, 360, CV_32SC1, p);
//    colored_map.convertTo(colored_map, CV_8UC1);
//    return colored_map;




void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{

}


void Classifier::Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels) {
// Convert the input image to the input image format of the network
    cv::Mat sample;

    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);
    cv::split(sample_float, *input_channels);
}


