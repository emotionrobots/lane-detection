#ifndef _LANE_H
#define _LANE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <Eigen/Eigen/Dense>
#include "math.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;
using namespace Eigen;

class laneDetection
{
private:
    Mat perspectiveMatrix;
    Mat oriImage; //The original input image.
    Mat edgeImage; // The result of applying canny edge detection.
    Mat warpEdgeImage;
    Mat warpOriImage;
    vector<Mat> imageChannels;
    Mat RedBinary;
    Mat mergeImage;
    Mat mergeImageRGB;
    Mat histImage; //Histogram visualization.
    Mat maskImage; //The curve image used to blend to the original image.
    Mat maskImageWarp;
    Mat finalResult;
    vector<int> histogram; //Histogram of the detected features
    vector<Point2f> laneL;
    vector<Point2f> laneR;
    vector<Point2f> curvePointsL;
    vector<Point2f> curvePointsR;
    int laneLcount;
    int laneRcount;
    int midPoint; //The mid position of the view.
    int midHeight;
    int leftLanePos; //The detected left lane boundary position.
    int rightLanePos; //The detected right lane boundary position.
    short initRecordCount; // To record the number of times of executions in the first 5 frames.
    const int blockNum; //Number of windows per line.
    int stepY; //Window moving step.
    const int windowSize; //Window Size (Horizontal).
    Mat imageUndistorted,output,result,Lab,warped,fitted,hls,image,fitted_warped;
     //Threshold for Yellow Line
    int lthresh1=160;
    int lthresh2=255;
    //Threshold for White Line
    int thresh1=220;
    int thresh2=255;
    int z;
    double* xarray;
    double* yarray;
    int order;
    vector<Point> red;
    vector<Point>blue;

    Vector3d curveCoefL; //The coefficients of the curve (left).
    Vector3d curveCoefR; //The coefficients of the curve (left).
    Vector3d curveCoefRecordL[5]; //To keep the last five record to smooth the current coefficients (left).
    Vector3d curveCoefRecordR[5]; //To keep the last five record to smooth the current coefficients (right).
    int recordCounter;
    bool failDetectFlag; // To indicate whether the road marks is detected succesfully.
    void calHist();
    void boundaryDetection();
    void laneSearch(const int &lanePos, vector<Point2f> &_line, int &lanecount, vector<Point2f> &curvePoints, char dir);
    bool laneCoefEstimate();
    void laneFitting();
    Mat threshlaneSearch(Mat img);
    
public:
    laneDetection(Mat _oriImage, Mat _perspectiveMatrix);
    ~laneDetection();
    void laneDetctAlgo();
    Mat getEdgeDetectResult();
    Mat getWarpEdgeDetectResult();
    Mat getRedChannel();
    Mat getRedBinary();
    Mat getMergeImage();
    Mat getHistImage();
    Mat getMaskImage();
    Mat getWarpMask();
    Mat getFinalResult();
    Mat polyfit(int z,double* xarray,double* yarray,int order);
    float getLaneCenterDist();
    void setInputImage(Mat &image);
    
};


#endif