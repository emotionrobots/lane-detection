#include <iostream>
#include <armadillo>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <fstream>
#include <math.h>
#include <ctime>
#include <deque>

//add OpenCV and Armadillo namespaces
using namespace cv;
using namespace std;
using namespace arma;


class vanishingPt
{
	public:
	cv::Mat image, img, gray;
	cv::Mat frame;
	vector< vector<int> > points;
	mat A,b, prevRes;
	mat Atemp, btemp, res, aug, error, soln;
	//ofstream out1, out2;
	struct datapoints{
		int x;
		int y;
		int rad;
	};
	deque<datapoints>values;
	deque<datapoints>ndp;
	float epsilon;
	int imgno=0;


	//store slope (m) and y-intercept (c) of each lines
	float m,c;

	//store minimum length for lines to be considered while estimating vanishing point
	int minlength;

	//temporary vector for intermediate storage
	vector<int> temp;

	//store (x1, y1) and (x2, y2) endpoints for each line segment
	vector<cv::Vec4i> lines_std;

	//video capture object from OpenCV
	cv::VideoCapture cap;

	//to store intermediate errors
	double temperr;

	//constructor to set video/webcam and find vanishing point
	vanishingPt()
	{
		//cv::namedWindow("win", 2);
		//cv::namedWindow("Lines", 2);
		
		// to calculate fps
		clock_t begin, end;

		//read from video file on disk
		//to read from webcam initialize as: cap = VideoCapture(int device_id);
		//cap = VideoCapture(1);
		cap = VideoCapture("./Data/IMG_4721.MOV");

		if( cap.isOpened() )//check if camera/ video stream is available
		{
			//get first frame to intialize the values
			cap.read(frame);
        	image= cv::Mat(cv::Size(frame.rows,frame.cols), CV_8UC1, 0.0);
		}
        
    	// define minimum length requirement for any line
		minlength = image.cols * image.cols * 0.01 ;

	    int flag=0;
		while( cap.isOpened() )//check if camera/ video stream is available
		{   

		    if ( ! cap.grab() )
		        continue;

			if(!cap.retrieve(img))
				continue;

			//it's advised not to modify image stored in the buffer structure of the opencv.
			frame  = img.clone();
            if(img.empty()){
                cout<<"Video Over"<<endl;
                break;
                
            }    

			//to calculate fps
			begin = clock();
			
			cv::cvtColor(frame,image , cv::COLOR_BGR2GRAY);

			//resize frame to 480x320
			cv::resize(image, image, cv::Size(640,480));

			//equalize histogram
			cv::equalizeHist(image, image);

			//initialize the line segment matrix in format y = m*x + c	
			init(image, prevRes);

			//draw lines on image and display
			makeLines(flag);
			
			//approximate vanishing point
			eval();

			//to calculate fps
			end = clock();
			//cout<<"fps: "<<1/(double(end-begin)/CLOCKS_PER_SEC)<<endl;

			//hit 'esc' to exit program
		    int k = cv::waitKey(1);
		    if ( k==27 )
		        break;
		}

	}
    ~vanishingPt(){}

	void init(cv::Mat image, mat prevRes)
	{
		//create OpenCV object for line segment detection
		cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);

		//initialize
    	lines_std.clear();

    	//detect lines in image and store in linse_std
    	//store (x1, y1) and (x2, y2) endpoints for each line segment
    	ls->detect(image, lines_std);

    	// Show found lines
    	cv::Mat drawnLines (image);
		//cout<<lines_std.size()<<endl;

		for(int i=0; i<lines_std.size(); i++)
		{

			//ignore if almost vertical
			if ( abs(lines_std[i][0]-lines_std[i][2]) < 10 || abs(lines_std[i][1]-lines_std[i][3]) < 10) //check if almost vertical
				continue;
			//ignore shorter lines (x1-x2)^2 + (y2-y1)^2 < minlength
			if( ((lines_std[i][0]-lines_std[i][2])*(lines_std[i][0]-lines_std[i][2]) +(lines_std[i][1]-lines_std[i][3])*(lines_std[i][1]-lines_std[i][3])) < minlength)
				continue;

            //store valid lines' endpoints for calculations
			for(int j=0; j<4; j++)
			{
				temp.push_back(lines_std[i][j]);
			}

			
			points.push_back(temp);
			temp.clear();
		}
		ls->drawSegments(drawnLines, lines_std);
		//cv::imshow("Lines", drawnLines);
		//cout<<"Detected:"<<lines_std.size()<<endl;
		cout<<"Filtered:"<<points.size()<<endl;

		for (int i=0;i<points.size();i++){
		cout<<points[i][0]<<" "<<points[i][1]<<" "<<points[i][2]<<" "<<points[i][3]<<endl;
		}
	}
	void makeLines(int flag)
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
			if (values.size() >=16){
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
		if(soln(0,0) > 0 && soln(0,0) < image.cols && soln(1,0) > 0 && soln(1,0) < image.rows){
			cv::resize(frame,frame,cv::Size(640,480));
			cv::circle(frame, Point(soln(0,0), soln(1,0)), 3, cv::Scalar(255,0,0), 10);
		}

		// draw a circle to visualize the approximate vanishing point
		if(ndp[ndp.size()/2].x > 0 && ndp[ndp.size()/2].x < image.cols && ndp[ndp.size()/2].y > 0 && ndp[ndp.size()/2].y< image.rows){
			cv::resize(frame,frame,cv::Size(640,480));
			cv::circle(frame, Point(ndp[ndp.size()/2].x, ndp[ndp.size()/2].y), 3, cv::Scalar(0,0,255), 10);
		}

		//Saving the images
		string name = std::to_string(imgno) + ".png";
		int key = cv::waitKey(1);
		if ( key=='r' )
			cv::imwrite( "./output_images/"+name, frame );

		//displaying the final output
		cv::resize(frame,frame,cv::Size(640,480));
		cv::imshow("win", frame);
		imgno++;
		//flush the vector
		points.clear();
		

		//toDo: use previous frame's result to reduce calculations and stabilize the region of vanishing point
		prevRes = soln;
	}

	//function to calculate and return the intersection point
	mat calc(mat A, mat b)
	{
	    mat x = zeros<mat>(2,1);
		solve(x,A,b);
	    return x;
	}
};


int main()
{
	// make object
    vanishingPt obj;
    cv::destroyAllWindows();
    return 0;
}