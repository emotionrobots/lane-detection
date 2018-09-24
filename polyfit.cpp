#include "polyfit.h"

using namespace std;
using namespace cv;

Mat Polyfit::mypolyfit(int Z,double* Xarray,double* Yarray,int Order)
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
