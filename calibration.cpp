#include "calibration.h"


using namespace std;
using namespace cv;

//......................Calibrate Camera...............//
void CamCalibrate::CamCal(){    
    //int numBoards = 0;
    int numCornersHor=6;
    int numCornersVer=9;
    

    //printf("Enter number of corners along width: ");
    //scanf("%d", &numCornersHor);

    //printf("Enter number of corners along height: ");
    //scanf("%d", &numCornersVer);

    //printf("Enter number of boards: ");
    //scanf("%d", &numBoards);
    String folder ="./camera_cal/*.jpg";
    vector<String> filename;
    glob(folder,filename);

    int numSquares = numCornersHor * numCornersVer;
    Size board_sz = Size(numCornersHor, numCornersVer);

    //VideoCapture capture = VideoCapture(0);

    vector<vector<Point3f>> object_points;
    vector<vector<Point2f>> image_points;

    vector<cv::Point2f> corners;
    //int successes=0;

    Mat image;
    Mat gray_image;

    //capture>>image;

    vector<cv::Point3f> obj;
    for(int j=0;j<numSquares;j++)
        obj.push_back(Point3f(j/numCornersHor, j%numCornersHor, 0.0f));


    for(size_t i=0;i<filename.size();++i)
    //while(successes<numBoards)
    {   

        image=imread(filename[i]);
        cvtColor(image, gray_image, CV_BGR2GRAY);

        bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if(found)
        {
            cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray_image, board_sz, corners, found);
            image_points.push_back(corners);
            object_points.push_back(obj);
            //cout<<"Snap sorted"<<endl;
        }

        //imshow("Original", image);
        //imshow("Gray", gray_image);
        
        //waitKey();
        //capture >> image;
    
    }

    Mat camMtx = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;

    camMtx.ptr<float>(0)[0] = 1;
    camMtx.ptr<float>(1)[1] = 1;

    calibrateCamera(object_points, image_points, image.size(), camMtx, distCoeffs, rvecs, tvecs);
    // ofstream myfile;
    // myfile.open("calibration.txt");
    
    // myfile<<camMtx<<endl;
    // myfile<<distCoeffs<<endl;
    //      //capture.release();
    // myfile.close();

    FileStorage fs;
    fs.open("Intrinsic.xml", FileStorage::WRITE);
    fs << "CameraMatrix" << camMtx;
    fs << "Dist" << distCoeffs;
    fs.release();
}

    //.................Calibration End..............................//
    
