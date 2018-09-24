//#include <Eigen/Eigen/Dense>
#include "polyfit.h"
#include "objectdetect.h"
#include "calibration.h"
#include "lanedetect.h"


using namespace cv;
using namespace std;
using namespace dnn;
//using namespace Eigen;

/*
const char* keys =
    "{ help  h     | | Print help message. }"
    "{ device      |  0 | camera device number. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
    "{ model m     | | Path to a binary file of model contains trained weights. "
                      "It could be a file with extensions .caffemodel (Caffe), "
                      ".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet).}"
    "{ config c    | | Path to a text file of model contains network configuration. "
                      "It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet).}"
    "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
    "{ mean        | | Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces. }"
    "{ scale       |  1 | Preprocess input image by multiplying on a scale factor. }"
    "{ width       | -1 | Preprocess input image by resizing to a specific width. }"
    "{ height      | -1 | Preprocess input image by resizing to a specific height. }"
    "{ rgb         |    | Indicate that model works with RGB input images instead BGR ones. }"
    "{ thr         | .5 | Confidence threshold. }"
    "{ nms         | .4 | Non-maximum suppression threshold. }"
    "{ backend     |  0 | Choose one of computation backends: "
                         "0: automatically (by default), "
                         "1: Halide language (http://halide-lang.org/), "
                         "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                         "3: OpenCV implementation }"
    "{ target      | 0 | Choose one of target computation devices: "
                         "0: CPU target (by default), "
                         "1: OpenCL, "
                         "2: OpenCL fp16 (half-float precision), "
                         "3: VPU }";





*/
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


//.........................parsing for  Yolo Object Detection model..................//

/*
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float confThreshold = parser.get<float>("thr");
    float nmsThreshold = parser.get<float>("nms");
    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    vector<string> classes;

    // Open file with classes names.
    if (parser.has("classes"))
    {
        std::string file = parser.get<String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    // Load a model.
    CV_Assert(parser.has("model"));
    Net net = readNet(parser.get<String>("model"), parser.get<String>("config"), parser.get<String>("framework"));
    net.setPreferableBackend(parser.get<int>("backend"));
    net.setPreferableTarget(parser.get<int>("target"));

    ObjectDetect detect;
    Mat frame;
    */

//..................................end parsing.............................................//

    //Path to video file or Webcam
    VideoCapture cap("/home/raj/CarND-Advanced-Lane-Lines/Data/IMG_4721.MOV");
    Mat imageUndistorted,image;
    int imgno;

    while(cap.isOpened()){

    double timer = (double)getTickCount();
    
    //image=imread("./Data/IMG_4713.JPG");    
    cap >> image;
    if(image.empty()){
        cout<<"Finished Video Stream!"<<endl;
        return 0;
    }

    resize(image,image,Point(1280,720));
    //image=detect.myobjectdetect(parser,net,image);
    //imageUndistorted=image;
    //###########  Uncomment Next Line to correct Distortion. #########//
    undistort(image, imageUndistorted, camMtx, distCoeffs);

    

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
    
    Mat M=getPerspectiveTransform(src_vertices,dst_vertices);
    Mat Minv=getPerspectiveTransform(dst_vertices,src_vertices);

    //warpPerspective(image,warped,M,Size(image.size().width,image.size().height),INTER_LINEAR);
    //imshow("warped",warped);

    
    laneDetection LaneAlgoVideo(image,src_vertices,dst_vertices);
    LaneAlgoVideo.laneDetctAlgo();
    Mat finalResult = LaneAlgoVideo.getFinalResult();

    string name = std::to_string(imgno) + ".png";
	int key = cv::waitKey(1);
	if ( key=='r' )
		cv::imwrite( "./output_images/"+name, finalResult );

    imgno++;    
    imshow("Final",finalResult);

    char c=(char)waitKey(25);
    if(c==27)
      break;

    }
    cap.release();
    waitKey();
    destroyAllWindows();
    return 0;
}

    
//     //Thresholding
//     //HLS For white lines
//     cvtColor(warped,hls,COLOR_BGR2HLS);

//     Mat hls_output=Mat(hls.rows,hls.cols,0.0);
//     vector<Mat>hls_channels;
//     split(hls,hls_channels);
//     //int thresh1=220;
//     //int thresh2=255;
//     Mat hls_l= hls_channels[1];
//     hls_l=hls_l*1.0;
//     //imshow("hls_l",hls_l);

//     for(int i=0;i<hls_l.rows;i++){
//         for (int j=0;j<hls_l.cols;j++){
//             Scalar pixel=hls_l.at<uchar>(i,j); 
//             if((pixel.val[0]>thresh1) & (pixel.val[0] <=thresh2)){
//                 hls_output.at<uchar>(i,j)=255;
//                     }
//             else{
//                 hls_output.at<uchar>(i,j)=0;
//             }
//        }
//     }
//     //imshow("hlsbin",hls_output);

    
//     //LAB for yellow lines
//     cvtColor(warped,Lab,COLOR_BGR2Lab);

//     Mat lab_output=Mat(Lab.rows,Lab.cols,0.0);
//     vector<Mat>lab_channels;
//     split(Lab,lab_channels);
//     //int lthresh1=160;
//     //int lthresh2=255;
//     Mat lab_b=lab_channels[2];

//     for(int i=0;i<lab_b.rows;i++){
//         for (int j=0;j<lab_b.cols;j++){
//             Scalar pixel2=lab_b.at<uchar>(i,j);
//             if((pixel2.val[0]>lthresh1) & (pixel2.val[0] <=lthresh2)){
//                 lab_output.at<uchar>(i,j)=255;
//             }
//             else{
//                 lab_output.at<uchar>(i,j)=0;
//             }
//         }
//     }
//     //imshow("labbin",lab_output);


//     //Combining the two
//     Mat combined=Mat(Lab.rows,Lab.cols,0.0);
//     for(int i=0;i<combined.rows;i++){
//         for (int j=0;j<combined.cols;j++){
//             if(lab_output.at<uchar>(i,j)==255 || hls_output.at<uchar>(i,j)==255){
//                 combined.at<uchar>(i,j)=255;
//             }
//             else{
//                 combined.at<uchar>(i,j)=0;
//             }
//         }
//     }
//     //imshow("Combined",combined);
     
//     Mat half_img=Mat(Lab.rows,Lab.cols/2,0.0);
//     for (int i=0;i<half_img.rows;i++){
//         for (int j=0;j<half_img.cols;j++){
//             Scalar pix = combined.at<uchar>(i,j);
//             half_img.at<uchar>(i,j)=pix.val[0];
//         }
//     }
//     //imshow("Half Image",half_img);

//     calcHist(&half_img,1,channels,Mat(),histogram,1,histSize,ranges,true,false);
    
//     histogram=histogram.t();
//     Point midpoint = histogram.size()/2;
//     int mid=midpoint.x;
//     int quarter=mid/2;
//     //cout<<mid<<" "<<quarter;

//     cvtColor(combined,fitted,CV_GRAY2BGR);
    
//     for(int i=0;i<combined.rows;i++){
//         for (int j=0;j<combined.cols;j++){
//             if((int)combined.at<uchar>(i,j) == 255 && j>quarter && j<=mid){
//                 fitted.at<Vec3b>(i,j)[0]=255;
//                 fitted.at<Vec3b>(i,j)[1]=0;
//                 fitted.at<Vec3b>(i,j)[2]=0;
//                 red.push_back(Point(j,i));
//                 //findNonZero(combined == 255,red);
//             }
//             if((int)combined.at<uchar>(i,j) == 255 && j>mid && j<=mid+quarter){
//                 fitted.at<Vec3b>(i,j)[0]=0;
//                 fitted.at<Vec3b>(i,j)[1]=0;
//                 fitted.at<Vec3b>(i,j)[2]=255;
//                 blue.push_back(Point(j,i));
//                 //findNonZero(combined==255,blue);
//             }

//         }
//     }
//     //cout<<red.size()<<" "<<blue.size()<<endl;
//     cout.precision(4);
//     cout.setf(ios::fixed);
//     int Z=red.size();
//     //random_shuffle(red.begin(),red.end());
//     double lrx[Z],lry[Z];
//     for(int i=0;i<Z;i++){
//         //cout<<red[i].x<<" "<<red[i].y<<" ";
//         lrx[i]=(double)red[i].x;
//         lry[i]=(double)red[i].y;
//     }


//     int Q=blue.size();
//     //random_shuffle(red.begin(),red.end());
//     double rrx[Q],rry[Q];
//     for(int i=0;i<Q;i++){
//         //cout<<red[i].x<<" "<<red[i].y<<" ";
//         rrx[i]=(double)blue[i].x;
//         rry[i]=(double)blue[i].y;
//     }

// // ################## Fitting a Polynomial ###################//

// Mat lcurve,rcurve,center;
// Polyfit rpoly;
// Polyfit lpoly;
// lcurve=lpoly.mypolyfit(Z,lry,lrx,2);
// rcurve=rpoly.mypolyfit(Q,rry,rrx,2);

// Mat fitted_warped=warped;
// polylines(fitted_warped,lcurve, false, Scalar(255,0,0), 20,8,0);
// polylines(fitted_warped,rcurve, false, Scalar(0,0,255), 10,8,0);

// // ......Viewing the Fitted Lines.......//
// polylines(fitted,lcurve, false, Scalar(0,255,0), 2,8,0);
// polylines(fitted,rcurve, false, Scalar(0,255,0), 2,8,0);
 
// imshow("combined_polyfit",fitted);

// //imshow("polyfit",fitted_warped);

// /*Mat fitted_warped=warped;
// for (int i = 0; i < curvePoints.size() - 1; i++){
//     line(fitted_warped, curvePoints[i], curvePoints[i + 1], Scalar(0,255,0), 2, 8,0);

// }*/

// //////////////////////////////////////// LINE FITTING ////////////////////////////////////////////////

//     /*
//     if(red.size() > 0 & red.size() < 10 ){
//         for (int i=0;i<red.size();i++){
//             tempred.push_back(red[i]);
//         }
//     }else{
//         for (int i=0;i<tempred.size();i++){
//             red.push_back(tempred[i]);
//         }
//     }

//     if(blue.size()>0 & blue.size() < 5){
//         for (int i=0;i<blue.size();i++){
//             tempblue.push_back(blue[i]);
//         }
//     }else{
//         for (int i=0;i<tempblue.size();i++){
//             blue.push_back(tempblue[i]);
//         }
//     }
//     */
    
//     //cout<<red.size()<<" "<<blue.size()<<" "<<tempred.size()<<" "<<tempblue.size()<<endl;
//     // Vec4f lines1,lines2;
//     // fitLine(red, lines1,CV_DIST_L2, 0, 0.01, 0.01);
//     // fitLine(blue,lines2,CV_DIST_L2, 0,0.01,0.01);
//     // line( fitted, Point(lines1[2],lines1[3]), Point(lines1[2]+lines1[0]*360,lines1[3]+lines1[1]*360),Scalar(0,255,255),1,8,0 );
//     // line( fitted, Point(lines1[2],lines1[3]), Point(lines1[2]-lines1[0]*500,lines1[3]-lines1[1]*500),Scalar(0,255,255),1,8,0 );

//     // line( fitted, Point(lines2[2],lines2[3]), Point(lines2[2]+lines2[0]*400,lines2[3]+lines2[1]*400),Scalar(0,255,255),1,8,0 );
//     // line( fitted, Point(lines2[2],lines2[3]), Point(lines2[2]-lines2[0]*400,lines2[3]-lines2[1]*400),Scalar(0,255,255),1,8,0 );
//     // cout<<lines1<<endl;
//     // cout<<lines2<<endl;

//     // Mat fitted_warped = warped;
//     // line( fitted_warped, Point(lines1[2],lines1[3]), Point(lines1[2]+lines1[0]*360,lines1[3]+lines1[1]*360),Scalar(0,0,255),20,8,0 );
//     // line( fitted_warped, Point(lines1[2],lines1[3]), Point(lines1[2]-lines1[0]*500,lines1[3]-lines1[1]*500),Scalar(0,0,255),20,8,0 );

//     // line( fitted_warped, Point(lines2[2],lines2[3]), Point(lines2[2]+lines2[0]*400,lines2[3]+lines2[1]*400),Scalar(255,0,0),10,8,0 );
//     // line( fitted_warped, Point(lines2[2],lines2[3]), Point(lines2[2]-lines2[0]*400,lines2[3]-lines2[1]*400),Scalar(255,0,0),10,8,0 );
   
//     //imshow("Fitted Warped",fitted_warped);

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
//     warpPerspective(fitted_warped,output,M,Size(fitted_warped.size().width,fitted_warped.size().height),WARP_INVERSE_MAP);
//     //imshow("final",output);

//     addWeighted(image,0.5,output,0.5,0.25,result);
//     float fps = getTickFrequency() / ((double)getTickCount() - timer);
//     string fps_label = format("Frame Per second: %.2f ", fps);
//     putText(result, fps_label, Point(900, 700), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0));

//     imshow("Result",result);

//     //cout<<red.size()<<" "<<blue.size()<<endl;

//     red.clear();
//     blue.clear();
//     //waitKey();
    
 




