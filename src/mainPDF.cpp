#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat BarDetection( const Mat& );
void SquareDetection( const Mat&, vector<vector<Point> >& );
static double angle( Point, Point, Point );

Mat src, aux;
String window_name = "gradient";
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
int valueHough = 100;
int thresCanny1 = 1;
int thresCanny2 = 1;
int thresGauss = 10;
int const max_value = 600;
int const max_value_Canny1 = 600;
int const max_value_Canny2 = 600;

String trackbar_type = "Cantidad puntos Hough";
String threshold1Canny = "Threshold 1 Canny";
String threshold2Canny = "Threshold 2 Canny";



int main(void)
{
    VideoCapture capture;

    capture.open( -1 );
    if ( ! capture.isOpened() ) {
        cout<<"--(!)Error opening video capture"<<endl;
        return -1;
    }

    while ( (char)waitKey(10) != 27 )
    {
        capture.read( src );
        vector<vector<Point> > squares;                    //!!!!!!!!!!????????????!!!!!!!!!!!???? No hay otra forma de vaciar el vector que volviendolo a hacer???

        //src.convertTo(src, CV_8U, 0.00390625);
        //vector<vector<Point> > squares;

        createTrackbar( trackbar_type, "detected lines", &valueHough, max_value );
        createTrackbar( threshold1Canny, "detected lines", &thresCanny1, max_value_Canny1 );
        createTrackbar( threshold2Canny, "detected lines", &thresCanny2, max_value_Canny2 );

        if( src.empty() )
        {
            cout<<" --(!) No captured frame -- Break!"<<endl;
            break;
        }

        flip( src, src, 1);
        aux = BarDetection( src );

        namedWindow( window_name, CV_WINDOW_AUTOSIZE );
        imshow( window_name, aux);

        SquareDetection( aux, squares );



    }

    return 0;
}

/* Funcion que retecta el codigo. Usa el gradiente de la imagen para determinar donde se encuentra el codigo de barras*/
Mat BarDetection ( const Mat& src ){

    Mat src_gray;
    Mat grad, grad_gauss_out, thres_out, rect, morph_out;
    Mat erode_out, dilate_out;


    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

    cvtColor( src, src_gray, CV_BGR2GRAY );

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    //convertScaleAbs( grad_x, abs_grad_x );
    /// Gradient Y
    Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    //convertScaleAbs( grad_y, abs_grad_y );

    subtract(grad_x,grad_y,grad);
    convertScaleAbs( grad, grad );

    //Gauss Blur
    GaussianBlur( grad, grad_gauss_out, Size( 9, 9 ), 0, 0 );

    //Any pixel in the gradient image that is not greater than 225 is set to 0 (black). Otherwise, the pixel is set to 255 (white).
    threshold( grad_gauss_out, thres_out, 230, 255, 0 );

    //construct a closing kernel and apply it to the thresholded image
    rect = getStructuringElement(MORPH_RECT, Size(30,10));
    morphologyEx(thres_out, morph_out, MORPH_CLOSE, rect );

    //perform a series of erosions and dilations
    Mat element = getStructuringElement( MORPH_RECT, Size(5,5));

    erode(morph_out, erode_out, element);
    dilate(erode_out, dilate_out, element);

    return dilate_out;
}

/* funcion que detecta rectangulos en la imagen y los remarca */
void SquareDetection ( const Mat& funcSrc, vector<vector<Point> >& squares ){

    vector<vector<Point> > contours;
    vector<Point> approx;
    Mat dst,dst2,cvt_out,equalize_out;
    vector<Vec4i> hierarchy;

    //------------FILTERS-------------
    //cvtColor(src, cvt_out, CV_BGR2GRAY);
    //equalizeHist( cvt_out, equalize_out );


    equalizeHist( funcSrc, equalize_out );

/*
    for ( int i = 1; i < 10; i = i + 2 )                                        // Gaussian filter
    {
        GaussianBlur( cvt_out, gauss_out, Size( i, i ), 0, 0 );
    }
*/
    Canny(equalize_out, dst, thresCanny1, thresCanny2, 3);
    //------------FILTERS-------------*/


    vector<Vec2f> lines;
    // detect lines
    HoughLines(dst, lines, 1, CV_PI/180, valueHough, 0, 0 );



    // draw lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( dst, pt1, pt2, Scalar(255,255,255), 2, CV_AA);
    }

    //cvtColor(dst, dst2, CV_BGR2GRAY);

    findContours( dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    //findContours(dst, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    // test each contour
    for( size_t i = 0; i < contours.size(); i++ )
    {
        // approximate contour with accuracy proportional
        // to the contour perimeter
        approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

        // square contours should have 4 vertices after approximation
        // relatively large area (to filter out noisy contours)
        // and be convex.
        // Note: absolute value of an area is used because
        // area may be positive or negative - in accordance with the
        // contour orientation
        if( approx.size() == 4 && fabs(contourArea(Mat(approx))) > 1000 && isContourConvex(Mat(approx)) )
        {
            double maxCosine = 0;

            for( int j = 2; j < 5; j++ )
            {
                // find the maximum cosine of the angle between joint edges
                double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                maxCosine = MAX(maxCosine, cosine);
            }

            // if cosines of all angles are small
            // (all angles are ~90 degree) then write quandrange
            // vertices to resultant sequence
            if( maxCosine < 0.3 )
                squares.push_back(approx);
        }
    }

    //Drawing contours
    Mat drawing = Mat::zeros( dst.size(), CV_8UC3 );

    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        Scalar color( 0, 255, 0 );
        drawContours( src, squares, idx, color, 9, 8);
    }

    namedWindow( "source", CV_WINDOW_NORMAL );
    namedWindow( "detected lines", CV_WINDOW_NORMAL );
    imshow("source", src);
    imshow("detected lines", dst);

}


static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
