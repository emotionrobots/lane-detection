//#include <Eigen/Eigen/Dense>
#include "polyfit.h"
#include "objectdetect.h"
#include "calibration.h"
#include "lanedetect.h"

#include<armadillo>
using namespace cv;
using namespace std;
using namespace dnn;
//using namespace Eigen;

int main(int argc, char** argv){

// .........................Calibration..............................//
    FileStorage fsRead;
    Mat camMtx, distCoeffs;
    fsRead.open("Intrinsic.xml", FileStorage::READ);

    if (fsRead.isOpened() == false){
        CamCalibrate cam;
        cam.CamCal(); 
        cout<<"Calibration finish, Run again"<<endl;
        return 0;
    }
    else{
        fsRead["CameraMatrix"] >> camMtx;
        fsRead["Dist"] >> distCoeffs;
        fsRead.release();
    }  
//....................End Calibration...............................//

    //Path to video file or Webcam
    VideoCapture cap("/home/raj/CarND-Advanced-Lane-Lines/project_video.mp4");
    Mat imageUndistorted,image;


    while(cap.isOpened()){

    double timer = (double)getTickCount();
    
    //image=imread("./Data/IMG_4713.JPG");    
    cap >> image;
    if(image.empty()){
        cout<<"Finished Video Stream!"<<endl;
        return 0;
    }

    //Image Size
    resize(image,image,Point(1280,720));

    //Distortion Effect
    //undistort(image, imageUndistorted, camMtx, distCoeffs);
    imageUndistorted=image;
    
    //Calibration Points
    Point2f src_vertices[4];
    src_vertices[0] = Point(580,490);   //575,464
    src_vertices[1] = Point(680,490);   //707,464
    src_vertices[2] = Point(300,700);   //258,682
    src_vertices[3] = Point(1049,700);  //1049,682

    Point2f dst_vertices[4];
    dst_vertices[0] = Point(450, 0);
    dst_vertices[1] = Point(imageUndistorted.size().width-450, 0);
    dst_vertices[2] = Point(450, imageUndistorted.size().height);
    dst_vertices[3] = Point(imageUndistorted.size().width-450, imageUndistorted.size().height);

    //................Lane Detection........................//

    laneDetection LaneAlgoVideo(image, src_vertices,dst_vertices);
    LaneAlgoVideo.laneDetctAlgo();

    //Will display the Final Result
    Mat finalResult = LaneAlgoVideo.getFinalResult();

    //Will give Left Lane Points
    Mat LeftPoints = LaneAlgoVideo.getleftPoints();

    //Will give Right Lane Points
    Mat RighPoints = LaneAlgoVideo.getrightPoints();

    //Will Display Final Result
    imshow("Final",finalResult);

    //Hit ESC to Stop
    char c=(char)waitKey(25);
    if(c==27)
      break;

    }
    cap.release();
    destroyAllWindows();
    return 0;
}

    
