#ifndef _POLYFIT_H
#define _POLYFIT_H
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace cv;
using namespace std;

class Polyfit{
public:
    Mat mypolyfit(int z,double* xarray,double* yarray,int order);
private:
    int z;
    double* xarray;
    double* yarray;
    int order;
   
};

#endif