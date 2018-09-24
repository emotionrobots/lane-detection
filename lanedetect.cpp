#include "lanedetect.h"
laneDetection::laneDetection(const Mat _oriImage,Point2f src[4],Point2f dst[4])
:oriImage(_oriImage), blockNum(9), windowSize(150), recordCounter(0), initRecordCount(0), failDetectFlag(true)
{
    histogram.resize(_oriImage.size().width);
    midPoint = _oriImage.size().width >> 1;
    midHeight = _oriImage.size().height * 0.55;
    stepY = oriImage.size().height / blockNum;
    perspectiveMatrix=getPerspectiveTransform(src,dst);
}

laneDetection::~laneDetection() {}
//The core of lane detection algorithm.
void laneDetection::laneDetctAlgo()
{
    //Conduct Canny edge detection.
    Mat oriImageGray;
    
    cvtColor(oriImage, oriImageGray, COLOR_RGB2GRAY);
    Canny(oriImageGray, edgeImage, 100, 150, 3);
    warpPerspective(edgeImage, warpEdgeImage, perspectiveMatrix, edgeImage.size());
    inRange(warpEdgeImage, Scalar(1),Scalar(255),warpEdgeImage);

    //Split the color image into different channels.
    warpPerspective(oriImage, warpOriImage, perspectiveMatrix, oriImage.size());
    split(warpOriImage, imageChannels);

    //Conduct binarization for R channel.
    inRange(imageChannels[2], Scalar(200), Scalar(255),RedBinary);

    //Merge the binarized R channel image with edge detected image.
    add(warpEdgeImage, RedBinary, mergeImage);
    cvtColor(mergeImage, mergeImageRGB, CV_GRAY2RGB);

    ///...........................
    fitted_warped=threshlaneSearch(warpOriImage);
    warpPerspective(fitted_warped,output,perspectiveMatrix,Size(fitted_warped.size().width,fitted_warped.size().height),WARP_INVERSE_MAP);


    //...............
    //Calculate the histogram.
    //calHist();

}


//Calculate the histogram of the lane features along x axis.
void laneDetection::calHist()
{
    histogram.clear();
    for(int i = 0; i < mergeImage.size().width; i++)
    {
        Mat ROI = mergeImage(Rect(i, oriImage.size().height-midHeight-1, 1, midHeight));
        Mat dst;
        divide(255, ROI, dst);
        histogram.push_back((int)(sum(dst)[0]));
    }
    int maxValue = 0;
    maxValue = (*max_element(histogram.begin(), histogram.end())); //the maximum value of the histogram.
    histImage.create(maxValue, histogram.size(), CV_8UC3);
    histImage = Scalar(255,255,255);

    //To create the histogram image
    for(int i=0; i<histogram.size(); i++)
    {
        line(histImage, Point2f(i,(maxValue-histogram.at(i))), Point2f(i,maxValue), Scalar(0,0,255), 1);
    }
}




Mat laneDetection::getEdgeDetectResult()
{
    return edgeImage;
}


Mat laneDetection::getWarpEdgeDetectResult()
{
    return warpEdgeImage;
}

Mat laneDetection::getRedChannel()
{
    return imageChannels[2];
}

Mat laneDetection::getRedBinary()
{
    return RedBinary;
}

Mat laneDetection::getMergeImage()
{
    return mergeImageRGB;
}

Mat laneDetection::getHistImage()
{
    return histImage;
}

Mat laneDetection::getMaskImage()
{
    return maskImage;
}

Mat laneDetection::getWarpMask()
{
    return maskImageWarp;
}

Mat laneDetection::getFinalResult()
{
    addWeighted(output, 0.5, oriImage, 0.5, 0, finalResult);
    //addWeighted(maskImageWarp, 0.5, oriImage, 1, 0, finalResult);
    return finalResult;
}

void laneDetection::setInputImage(Mat &image)
{
    oriImage = image.clone();
}

float laneDetection::getLaneCenterDist()
{
    float laneCenter = ((rightLanePos - leftLanePos) / 2) + leftLanePos;
    float imageCenter = mergeImageRGB.size().width / 2;
    float result;
    result = (laneCenter -imageCenter)* 3.5 / 600; //Assume the lane width is 3.5m and about 600 pixels in our image.
    return result;
}

Mat laneDetection::threshlaneSearch(Mat warped)
{
    cvtColor(warped,hls,COLOR_BGR2HLS);
    Mat fitted_warped=warped.clone();

    Mat hls_output=Mat(hls.rows,hls.cols,0.0);
    vector<Mat>hls_channels;
    split(hls,hls_channels);
    Mat hls_l= hls_channels[1];
    hls_l=hls_l*1.0;
    //imshow("hls_l",hls_l);

    for(int i=0;i<hls_l.rows;i++){
        for (int j=0;j<hls_l.cols;j++){
            Scalar pixel=hls_l.at<uchar>(i,j); 
            if((pixel.val[0]>thresh1) & (pixel.val[0] <=thresh2)){
                hls_output.at<uchar>(i,j)=255;
                    }
            else{
                hls_output.at<uchar>(i,j)=0;
            }
       }
    }
    //imshow("hlsbin",hls_output);

    
    //LAB for yellow lines
    cvtColor(warped,Lab,COLOR_BGR2Lab);

    Mat lab_output=Mat(Lab.rows,Lab.cols,0.0);
    vector<Mat>lab_channels;
    split(Lab,lab_channels);
    Mat lab_b=lab_channels[2];

    for(int i=0;i<lab_b.rows;i++){
        for (int j=0;j<lab_b.cols;j++){
            Scalar pixel2=lab_b.at<uchar>(i,j);
            if((pixel2.val[0]>lthresh1) & (pixel2.val[0] <=lthresh2)){
                lab_output.at<uchar>(i,j)=255;
            }
            else{
                lab_output.at<uchar>(i,j)=0;
            }
        }
    }
    //imshow("labbin",lab_output);


    //Combining the two
    Mat combined=Mat(Lab.rows,Lab.cols,0.0);
    for(int i=0;i<combined.rows;i++){
        for (int j=0;j<combined.cols;j++){
            if(lab_output.at<uchar>(i,j)==255 || hls_output.at<uchar>(i,j)==255){
                combined.at<uchar>(i,j)=255;
            }
            else{
                combined.at<uchar>(i,j)=0;
            }
        }
    }
    //imshow("Combined",combined);
     
    Mat half_img=Mat(Lab.rows,Lab.cols/2,0.0);
    for (int i=0;i<half_img.rows;i++){
        for (int j=0;j<half_img.cols;j++){
            Scalar pix = combined.at<uchar>(i,j);
            half_img.at<uchar>(i,j)=pix.val[0];
        }
    }
    //imshow("Half Image",half_img);

   
    int mid=640;
    int quarter=mid/2;
    //cout<<mid<<" "<<quarter;

    cvtColor(combined,fitted,CV_GRAY2BGR);
    
    for(int i=0;i<combined.rows;i++){
        for (int j=0;j<combined.cols;j++){
            if((int)combined.at<uchar>(i,j) == 255 && j>quarter && j<=mid){
                fitted.at<Vec3b>(i,j)[0]=255;
                fitted.at<Vec3b>(i,j)[1]=0;
                fitted.at<Vec3b>(i,j)[2]=0;
                red.push_back(Point(j,i));
                //findNonZero(combined == 255,red);
            }
            if((int)combined.at<uchar>(i,j) == 255 && j>mid && j<=mid+quarter){
                fitted.at<Vec3b>(i,j)[0]=0;
                fitted.at<Vec3b>(i,j)[1]=0;
                fitted.at<Vec3b>(i,j)[2]=255;
                blue.push_back(Point(j,i));
                //findNonZero(combined==255,blue);
            }

        }
    }
    //cout<<red.size()<<" "<<blue.size()<<endl;
    cout.precision(4);
    cout.setf(ios::fixed);
    int Z=red.size();
    //random_shuffle(red.begin(),red.end());
    double lrx[Z],lry[Z];
    for(int i=0;i<Z;i++){
        //cout<<red[i].x<<" "<<red[i].y<<" ";
        lrx[i]=(double)red[i].x;
        lry[i]=(double)red[i].y;
    }
    
    polylines(fitted_warped,polyfit(Z,lry,lrx,2), false, Scalar(255,0,0), 20,8,0);
    curvePointsL=polyfit(Z,lry,lrx,2);
    

    int Q=blue.size();
    //random_shuffle(red.begin(),red.end());
    double rrx[Q],rry[Q];
    for(int i=0;i<Q;i++){
        //cout<<red[i].x<<" "<<red[i].y<<" ";
        rrx[i]=(double)blue[i].x;
        rry[i]=(double)blue[i].y;
    }
 
    polylines(fitted_warped,polyfit(Q,rry,rrx,2), false, Scalar(0,0,255), 10,8,0);
    curvePointsR=polyfit(Q,rry,rrx,2);

    red.clear();
    blue.clear();

    return fitted_warped;
}


Mat laneDetection::polyfit(int Z,double* Xarray,double* Yarray,int Order)
{
    int i,j,k,n,N;
    
    cout.precision(4);                        //set precision
    cout.setf(ios::fixed);
    N=Z;                                                       
    double x[N],y[N];

    for (i=0;i<N;i++)
        x[i]=Xarray[i];
    
    for (i=0;i<N;i++)
        y[i]=Yarray[i];
    n=Order;                                // n is the degree of Polynomial 
    double X[2*n+1];                        //Array that will store the values of sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
    for (i=0;i<2*n+1;i++)
    {
        X[i]=0;
        for (j=0;j<N;j++)
            X[i]=X[i]+pow(x[j],i);        //consecutive positions of the array will store N,sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
    }
    double B[n+1][n+2],a[n+1];            //B is the Normal matrix(augmented) that will store the equations, 'a' is for value of the final coefficients
    for (i=0;i<=n;i++)
        for (j=0;j<=n;j++)
            B[i][j]=X[i+j];            //Build the Normal matrix by storing the corresponding coefficients at the right positions except the last column of the matrix
    double Y[n+1];                    //Array to store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
    for (i=0;i<n+1;i++)
    {    
        Y[i]=0;
        for (j=0;j<N;j++)
        Y[i]=Y[i]+pow(x[j],i)*y[j];        //consecutive positions will store sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
    }
    for (i=0;i<=n;i++)
        B[i][n+1]=Y[i];                //load the values of Y as the last column of B(Normal Matrix but augmented)
    n=n+1;                //n is made n+1 because the Gaussian Elimination part below was for n equations, but here n is the degree of polynomial and for n degree we get n+1 equations
    

    for (i=0;i<n;i++)                    //From now Gaussian Elimination starts(can be ignored) to solve the set of linear equations (Pivotisation)
        for (k=i+1;k<n;k++)
            if (B[i][i]<B[k][i])
                for (j=0;j<=n;j++)
                {
                    double temp=B[i][j];
                    B[i][j]=B[k][j];
                    B[k][j]=temp;
                }
    
    for (i=0;i<n-1;i++)            //loop to perform the gauss elimination
        for (k=i+1;k<n;k++)
            {
                double t=B[k][i]/B[i][i];
                for (j=0;j<=n;j++)
                    B[k][j]=B[k][j]-t*B[i][j];    //make the elements below the pivot elements equal to zero or elimnate the variables
            }
    for (i=n-1;i>=0;i--)                //back-substitution
    {                        //x is an array whose values correspond to the values of x,y,z..
        a[i]=B[i][n];                //make the variable to be calculated equal to the rhs of the last equation
        for (j=0;j<n;j++)
            if (j!=i)            //then subtract all the lhs values except the coefficient of the variable whose value                                   is being calculated
                a[i]=a[i]-B[i][j]*a[j];
        a[i]=a[i]/B[i][i];            //now finally divide the rhs by the coefficient of the variable to be calculated
        
    }

    int start_point_x = Xarray[0];
    int end_point_x = Xarray[N-1];
    vector<Point2f> curvePoints;

    if(start_point_x>end_point_x){
        end_point_x=start_point_x;
        start_point_x=Xarray[N-1];
    }

    float xfit;
    //Define the curve through equation. In this example, a simple parabola
    for (float yfit = start_point_x; yfit <= end_point_x; yfit++){
        xfit = a[2]*yfit*yfit + a[1]*yfit + a[0];
        Point2f new_point = Point2f(xfit, yfit);                  //resized to better visualize
        curvePoints.push_back(new_point);                       //add point to vector/list
    }

//Option 1: use polylines
    Mat curve(curvePoints, true);
    curve.convertTo(curve, CV_32S); //adapt type for polylines

   return curve;
}

Mat laneDetection::getleftPoints(){
    return curvePointsL;
}

Mat laneDetection::getrightPoints(){
    return curvePointsR;
}