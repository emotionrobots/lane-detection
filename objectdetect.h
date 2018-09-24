#ifndef _OBJECT_H
#define _OBJECT_H
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
//using namespace dnn;
using namespace cv;

class ObjectDetect{
public:
    
    Mat myobjectdetect(CommandLineParser parser,cv::dnn::experimental_dnn_v5::Net net,Mat& frame);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, vector<string>classes);
    void postprocess(Mat& frame, const std::vector<Mat>& out, cv::dnn::experimental_dnn_v5::Net& net, vector<string>classes,float confThreshold, float nmsThreshold);
    void callback(int pos, void* userdata,float confThreshold);
    vector<String> getOutputsNames(cv::dnn::experimental_dnn_v5::Net net);
private:
    float confThreshold;
    vector<string> classes;
    float nmsThreshold;
};



#endif