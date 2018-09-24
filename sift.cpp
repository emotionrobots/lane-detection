#include <stdio.h>
#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


void drawMatches_self(Mat& img1, vector<KeyPoint> kp1, Mat& img2, vector<KeyPoint> kp2, vector<DMatch> matches,Mat des1,Mat des2){
    Mat out;
    vconcat(img1,img2,out);
    imshow("Temp",out);


}


int main(){
    std::vector<KeyPoint> kp_pf, kp;
    Mat des_pf, des;
    Mat img,prev_gray,prev_frame,gray;
    VideoCapture cap;
    
    
    //cap = VideoCapture("project_video.mp4");
    cap = VideoCapture(1);
    if (!cap.isOpened())
	{
		cout << "Could not open reference " << endl;
		return -1;
	}
    cap.read(img);
    cvtColor(img,prev_gray,COLOR_BGR2GRAY);

    prev_frame=img;

    
    Ptr<Feature2D>sift = SIFT::create(100);
    sift->detectAndCompute(prev_gray,Mat(),kp_pf,des_pf);

    while(cap.isOpened())
    {
        cap.read(img);
        cvtColor(img,gray,COLOR_BGR2GRAY);

        sift->detectAndCompute(gray,Mat(),kp,des);

        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher.match( des, des_pf, matches );

        // std::vector<DMatch> matches;
        // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        // matcher->match(des, des_pf, matches, Mat());
   
        // // Sort matches by score
        // std::sort(matches.begin(), matches.end());
   
        // // Remove not so good matches
        // const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
        // matches.erase(matches.begin()+numGoodMatches, matches.end());
   

        double max_dist = 0; double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < des.rows; i++ )
        { 
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        //printf("-- Max dist : %f \n", max_dist );
        //printf("-- Min dist : %f \n", min_dist );
        //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
        //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very small)
        //-- PS.- radiusMatch can also be used here.
        std::vector< DMatch > good_matches;
        for( int i = 0; i < des.rows; i++ )
        { 
            if( matches[i].distance <= max(2*min_dist, 0.02) )
            { good_matches.push_back( matches[i]); }
        }
        //-- Draw only "good" matches
        Mat img_matches;
        drawMatches( img, kp, prev_frame, kp_pf,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        drawMatches_self(img,kp,prev_frame,kp_pf,good_matches,des,des_pf);
        //-- Show detected matches
        imshow( "Good Matches", img_matches );
        kp=kp_pf;
        prev_frame=img;
        des=des_pf;
        
        for( int i = 0; i < (int)matches.size(); i++ )
        {
            //printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
            cout<<"Good Match ["<<i<<"] Keypoint img1: "<<good_matches[i].queryIdx<<" Keypoint img2: "<<good_matches[i].trainIdx<<endl;  
        }
        waitKey(0);
        int k = cv::waitKey(1);
		    if ( k==27 )
		        break;


    }


    return 0;
}