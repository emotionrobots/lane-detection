#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include "opencv2/video/tracking.hpp"
#include <armadillo>
#include <deque>
#include <math.h>
#include <iomanip>

using namespace std;
using namespace cv;
using namespace arma;

static void drawHsv(const cv::Mat& flow, cv::Mat& bgr) {
	//extract x and y channels
	cv::Mat xy[2]; //X,Y
	split(flow, xy);

	//calculate angle and magnitude
	cv::Mat magnitude, angle, hsv;
	cartToPolar(xy[0], xy[1], magnitude, angle, true);

	//translate magnitude to range [0;1]
	double mag_max;
	minMaxLoc(magnitude, 0, &mag_max);
	magnitude.convertTo(
		magnitude,    // output matrix
		-1,           // type of the ouput matrix, if negative same type as input matrix
		1.0 / mag_max // scaling factor
	);


	//build hsv image
	cv::Mat _hsv[3];
	_hsv[0] = angle;
	_hsv[1] = magnitude;
	_hsv[2] = cv::Mat::ones(angle.size(), CV_32F);

	merge(_hsv, 3, hsv);
	//convert to BGR and show
	cvtColor(hsv, bgr, COLOR_HSV2BGR);
}

class vanishingPtOF
{
	public:
	vector<int> temp;
	struct mypoint{
		double xi;
		double yi;
	};
	vector<mypoint> interpoints;
	vector<vector<int>>points;
	mat A,b, prevRes;
	mat Atemp, btemp, res, aug, error, soln;
	double temperr;
	struct datapoints{
	int x;
	int y;
	int rad;
	};

	
	bool playVideo = true;
	deque<datapoints>values;
	deque<datapoints>ndp;
	float epsilon;
	int imgno=0;
	cv::Mat flow, cflow, frame, gray, prevgray, img_bgr, img_hsv,tempframe;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy; 
	#define pdd pair<double, double>

	vanishingPtOF()
	{	
		VideoCapture cap("./Data/IMG_4721.MOV");
		if (!cap.isOpened())
		{
			cout << "Could not open reference " << endl;
			//return -1;
		}
		// cv::Mat flow, cflow, frame, gray, prevgray, img_bgr, img_hsv;
		// vector<vector<Point> > contours;
		// vector<Vec4i> hierarchy; 

		//namedWindow("flow", 1);

		for (;;)
		{	
			if(playVideo){
				cap >> frame;
			}
			if(frame.empty()){
				cout<<"video over"<<endl;
				break;
			}
			cv::resize(frame,frame,cv::Size(640,480));
			tempframe=frame;
			cvtColor(frame, gray, COLOR_BGR2GRAY);
		
			if (!prevgray.empty())
			{
				calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 5, 16, 3, 5, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
			// calculate dense optical flow
			/*calcOpticalFlowFarneback(
				prevgray,
				gray,
				flow, // computed flow image that has the same size as prev and type CV_32FC2
				0.5,  // image scale: < 1 to build pyramids for each image. 0.5 means a
					  // classical pyramid, where each next layer is twice smalller than the
					  // previous one
				5,    // number of pyramid layers
				15,   // averaging windows size. larger values increase the algorithm robustness
					  // to image noise and give more chances for fast motion detection, but
					  // yields more blurred motion field
				3,    // number of iterations for each pyramid level
				5,    // size of the pixel neighborhood used to find the polynomial expansion
					  // in each pixel
				1.1,  // standard deviation of the Gaussian used to smooth derivations
				OPTFLOW_FARNEBACK_GAUSSIAN     // flags
			);*/

				cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
				drawOptFlowMap(flow, cflow, 3, 64, (255, 0, 255));

				makeLines();
				eval();
				//displayintersection();
				imshow("flow", cflow);
			
			// 	//drawHsv(flow, img_bgr);
			// 	cv::Mat gray_bgr = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
			// 	cvtColor(img_bgr, gray_bgr, COLOR_BGR2GRAY);
			// 	normalize(gray_bgr, gray_bgr, 0, 255, NORM_MINMAX, CV_8UC1);
			// 	blur(gray_bgr, gray_bgr, Size(3, 3));
			// 	//imshow("gray", gray_bgr);

			// /// Detect edges using Threshold
			// 	cv::Mat img_thresh = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
			// 	threshold(gray_bgr, img_thresh, 155, 255, THRESH_BINARY_INV);
			// 	dilate(img_thresh, img_thresh, 0, Point(-1, -1), 2);
			// 	// imshow("tresh",img_thresh);
			// 	findContours(img_thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			// 	for (int i = 0; i< contours.size(); i++)
			// 	{
			// 		vector<vector<Point> > contours_poly(contours.size());
			// 		approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
			// 		Rect box = boundingRect(cv::Mat(contours_poly[i]));
			// 		if (box.width > 50 && box.height > 50 && box.width < 900 && box.height < 680) {
			// 			rectangle(frame,
			// 			box.tl(), box.br(),
			// 			Scalar(0, 255, 0), 4);
			// 		}
			// 	}	
	

			/// Show in a window
			//namedWindow("Contours", WINDOW_AUTOSIZE);
			//imshow("Contours", frame);
			}
		char key = waitKey(5);
        if(key == 'p')
            playVideo = !playVideo;


		char c = (char)waitKey(5);
			if (c == 27) break;
		std::swap(prevgray, gray);
		}
	}
	~vanishingPtOF(){}
	
	void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, double scale, int step, const Scalar& color)
	{	
	for (int y = cflowmap.rows/3; y < cflowmap.rows; y += step){
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x) * scale;
			//line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),color);
			circle(cflowmap, Point(x, y), 2, (255,255,0), -1);
			circle(cflowmap, Point(x+fxy.x, y+fxy.y), 2, CV_RGB(255,255,0), -1);

			// if ( abs(fxy.x) < 10 || abs(fxy.y) < 10) //check if almost vertical
			// 	continue;
			// //ignore shorter lines (x1-x2)^2 + (y2-y1)^2 < minlength
			// int minlength = cflowmap.cols * cflowmap.cols * 0.01 ;
			// if( ((fxy.x)*(fxy.x) +(fxy.y)*(fxy.y)) < minlength)
			// 	continue;
			
			//if(fxy.x >= 1 && fxy.y >= 1)
			if(sqrt( fxy.x * fxy.x + fxy.y*fxy.y) >= 20)
			{
				line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),color);
				temp.push_back(x);
				temp.push_back(y);
				temp.push_back(x+fxy.x);
				temp.push_back(y+fxy.y);
			}
		}
		if(temp.size()){
				points.push_back(temp);
				temp.clear();
			}
	}
	// for (int i=0;i<points.size();i++){
	// 	cout<<points[i][0]<<" "<<points[i][1]<<" "<<points[i][2]<<" "<<points[i][3]<<endl;
	// }
	// cout<<"-----------"<<endl;
	}

	void makeLines()
	{
		// to solve Ax = b for x
	    A = zeros<mat>(points.size(), 2);
	    b = zeros<mat>(points.size(), 1);
		
		
	    //convert given end-points of line segment into a*x + b*y = c format for calculations
	    //do for each line segment detected
	    for(int i=0; i<points.size(); i++)
	    {
            
			A(i,0)=-(points[i][3]-points[i][1]);			//-(y2-y1)
			A(i,1)=(points[i][2]-points[i][0]);				//x2-x1
			b(i,0)=A(i,0)*points[i][0]+A(i,1)*points[i][1];	//-(y2-y1)*x1 + (x2-x1)*y1
	    }
	}

	//estimate the vanishing point
	void eval()
	{
		//stores the estimated co-ordinates of the vanishing point with respect to the image
		soln= zeros<mat>(2,1);
			

		//initialize error
		double err = 9999999999;

		//calculate point of intersection of every pair of lines and
		//find the sum of distance from all other lines
		//select the point which has the minimum sum of distance
		for(int i=0; i<points.size(); i++)
		{
			for(int j=0; j<points.size(); j++)
			{
				if(i >= j)
				continue;

				//armadillo vector
				uvec indices;

				//store indices of lines to be used for calculation
				indices << i << j;

				
				//extract the rows with indices specified in uvec indices
				//stores the ith and jth row of matrices A and b into Atemp and btemp respectively
				//hence creating a 2x2 matrix for calculating point of intersection
				Atemp = A.rows(indices);
				btemp = b.rows(indices);
				
				//if lines are parallel then skip
				if(det(Atemp) == 0)
					continue;

				
				//solves for 'x' in A*x = b
				res = calc(Atemp, btemp);


				if(res.n_rows == 0 || res.n_cols == 0)
					continue;

				// calculate error assuming perfect intersection is 
				error = A*res - b;

				//reduce size of error
				error = error/1000;

				// to store intermediate error values
				temperr = 0;
				//summation of errors
				for(int i=0; i<error.n_rows ; i++)
                    temperr+=(error(i,0)*error(i,0))/1000;

                //scale errors to prevent any overflows
				temperr/=1000000;

				//if current error is smaller than previous min error then update the solution (point)
				if(err > temperr)
				{
					soln = res;
					err = temperr;
				}
			}
		}
		
		//cout<<"\n\nResult:\n"<<soln(0,0)<<","<<soln(1,0)<<"\nError:"<<err<<"\n\n";
		
		struct mysort {
        bool operator() (datapoints a, datapoints b){
            return a.rad<b.rad;
        	}
    	} sort_rad;
		
		
		Point pt(soln(0,0),soln(1,0));
		if(pt.x > 0 && pt.y > 0){
			values.push_front({pt.x,pt.y,pt.x*pt.x + pt.y*pt.y});
			if (values.size() >=6){
				ndp=values;
				sort(ndp.begin(),ndp.end(),sort_rad);
				values.pop_back();
				// for(int j=0 ;j<values.size();j++){
                // 	cout<<values[j].x<<" "<<values[j].y<<" "<<values[j].rad<<endl;
				// }
				// cout<<"Sorted NDP"<<endl;
				// for(int j=0 ;j<values.size();j++){
                // 	cout<<ndp[j].x<<" "<<ndp[j].y<<" "<<ndp[j].rad<<endl;
           		// }
				//values.clear();
			}
		}
		//cout<<"-------------"<<endl;
		

		// //draw a circle to visualize the approximate vanishing point
		if(soln(0,0) > 0 && soln(0,0) < frame.cols && soln(1,0) > 0 && soln(1,0) < frame.rows){
			cv::resize(frame,frame,cv::Size(640,480));
			cv::circle(frame, Point(soln(0,0), soln(1,0)), 3, cv::Scalar(255,0,0), 10);
		}

		// draw a circle to visualize the approximate vanishing point
		if(ndp[ndp.size()/2].x > 0 && ndp[ndp.size()/2].x < frame.cols && ndp[ndp.size()/2].y > 0 && ndp[ndp.size()/2].y< ((frame.rows/2)+100)  && ndp[ndp.size()/2].y> ((frame.rows/2)-100)){
			cv::resize(frame,frame,cv::Size(640,480));
			cv::circle(frame, Point(ndp[ndp.size()/2].x, ndp[ndp.size()/2].y), 3, cv::Scalar(0,0,255), 10);
		}

		// //Saving the images
		// string name = std::to_string(imgno) + ".png";
		// int key = cv::waitKey(1);
		// if ( key=='r' )
		// 	cv::imwrite( "./output_images/"+name, frame );

		//displaying the final output
		cv::resize(frame,frame,cv::Size(640,480));
		cv::imshow("win", frame);
		imgno++;
		
		pdd intersection=lineLineIntersection();
		// if (intersection.first == FLT_MAX && intersection.second==FLT_MAX)
    	// 	{
        // 	cout << "The given lines AB and CD are parallel.\n";
    	// 	}
		// else
    	// {
        // // NOTE: Further check can be applied in case
        // // of line segments. Here, we have considered AB
        // // and CD as lines
        // cout << "The intersection of the given lines AB "
        //         "and CD is: "<< intersection.first <<" "<<intersection.second<<endl;
   		// }

		displayintersection();   
		//flush the vector
		points.clear();
	//temp.clear();
	}
mat calc(mat A, mat b)
	{
	    mat x = zeros<mat>(2,1);
		solve(x,A,b);
	    return x;
	}

mypoint make_point(double x, double y) {
    mypoint inpt = {x, y};
    return inpt;
}

pdd lineLineIntersection()
{
   	for(int i=0; i<points.size(); i++)
	{
		for(int j=0; j<points.size(); j++)
		{
			if (i>=j)
				continue;
		pdd A = make_pair(points[i][0],points[i][1]);
		pdd B = make_pair(points[i][2],points[i][3]);
		pdd C = make_pair(points[j][0],points[j][1]);
		pdd D = make_pair(points[j][2],points[j][3]);

		// Line AB represented as a1x + b1y = c1
    	double a1 = B.second - A.second;
    	double b1 = A.first - B.first;
    	double c1 = a1*(A.first) + b1*(A.second);
 
    	// Line CD represented as a2x + b2y = c2
    	double a2 = D.second - C.second;
    	double b2 = C.first - D.first;
    	double c2 = a2*(C.first)+ b2*(C.second);
 
    	double determinant = a1*b2 - a2*b1;
	
		double xi,yi;
    	if (determinant)
    		{
			xi = (b2*c1 - b1*c2)/determinant;
        	yi = (a1*c2 - a2*c1)/determinant;
			if(xi > 0 && yi > 0)
				interpoints.push_back(make_point(xi,yi));
    		}
    	else
    		{
        	// The lines are parallel. This is simplified
        	// by returning a pair of FLT_MAX
			xi=FLT_MAX;
			yi=FLT_MAX;
        	return make_pair(xi,yi);
    		}
		}

	}
}

void displayintersection(){
	for (int i=0;i<interpoints.size();i++){
		if(interpoints[i].xi > 0 && interpoints[i].xi < tempframe.cols && interpoints[i].yi > 0  && interpoints[i].yi < ((tempframe.rows/2) +100) && interpoints[i].yi > ((tempframe.rows/2) -100) ){
			cout<<interpoints[i].xi<<" "<<interpoints[i].yi<<endl;
			cv::circle(tempframe, Point(interpoints[i].xi, interpoints[i].yi), 2, cv::Scalar(0,255,0), 4);
		}
	}
	imshow("Inteersecrion",tempframe);
	interpoints.clear();
}
};


int main(int argc, char** argv)
{   
    vanishingPtOF obj;
	cv::destroyAllWindows();
	return 0;
}