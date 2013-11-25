//============================================================================
// Name        : aia2.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : use fourier descriptors to classify leafs in images
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void getContourLine(Mat& img, vector<Mat>& objList, int thresh, int k);
Mat makeFD(Mat& contour);
Mat normFD(Mat& fd, int n);
void showImage(Mat& img, string win, double dur=-1);
void plotFD(Mat& fd, string win, double dur=-1);

/* usage:
argv[1] path to query image
argv[2] example image for class 1
argv[3] example image for class 2
*/
// main function, loads and saves image
int main(int argc, char** argv) {

	// check if image paths were defined
	if (argc != 4){
	    cerr << "Usage: aia2 <input image>  <class 1 example>  <class 2 example>" << endl;
	    return -1;
	}
  
	// process image data base
	// load image as gray-scale, paths in argv[2] and argv[3]
	Mat exC1 = imread( argv[2], 0);
	Mat exC2  = imread( argv[3], 0);
	if ( (!exC1.data) || (!exC2.data) ){
	    cout << "ERROR: Cannot load class examples in\n" << argv[2] << "\n" << argv[3] << endl;
	    return -1;
	}
	
	// parameters
	int binThreshold;			// threshold for image binarization
	int numOfErosions;			// number of applications of the erosion operator
	int steps;				// number of dimensions of the FD
	double detThreshold;			// threshold for detection

	// get contour line from images
	vector<Mat> contourLines1;
	vector<Mat> contourLines2;
	// TO DO !!!
	binThreshold = 120;
	numOfErosions = 2;
	getContourLine(exC1, contourLines1, binThreshold, numOfErosions);
	getContourLine(exC2, contourLines2, binThreshold, numOfErosions);

	// you could also reuse img1 here
    Mat mask = Mat::zeros(exC2.rows, exC2.cols, CV_8UC1);

	contourLines2.pop_back();

    // CV_FILLED fills the connected components found
    drawContours(mask, contourLines2, -1, Scalar(255), CV_FILLED);

	// Mitja: just for showing preliminary results
	showImage(exC1, "Result", 0);
	showImage(exC2, "Result", 0);
	showImage(mask, "Result", 0);
	
	// calculate fourier descriptor
	Mat fd1 = makeFD(contourLines1.front());
	Mat fd2 = makeFD(contourLines2.front());



	Mat ch1, ch2;
	// "channels" is a vector of 3 Mat arrays:
	vector<Mat> channels(2);
	// split img:
	split(fd1, channels);
	// get the channels (dont forget they follow BGR order in OpenCV)
	ch1 = channels[0];
	ch2 = channels[1];

	//plotFD(fd1, "fd1");
	cout << ch2.size() << " ch2 länge " << "\n";

	cout << fd1.at<float>(0,1) << " 0 1 " << "\n";
	cout << ch1.at<float>(0,0) << " ch1 0 0 " << "\n";
	cout << ch2.at<float>(0,0) << " ch2 0 0 " << "\n";
	cout << contourLines1.size() << " länge " << "\n";
	cout << contourLines1.front().channels() << " channels " << "\n";
	cout << fd1.channels() << " channels " << "\n";
	cv::Size s = exC1.size();
	cout << s.height << " Breite " << s.width << "\n";
	s = fd1.size();
	cout << s.height << " Breite " << s.width << "\n";
	s = contourLines1.front().size();
	cout << s.height << " Breite " << s.width << "\n";
	s = contourLines1.back().size();
	cout << s.height << " Breite " << s.width << "\n";	
	//// normalize  fourier descriptor
	//// TO DO !!!
	//steps = 1;
	//Mat fd1_norm = normFD(fd1, steps);
	//Mat fd2_norm = normFD(fd2, steps);

	//// process query image
	//// load image as gray-scale, path in argv[1]
	//Mat query = imread( argv[1], 0);
	//if (!query.data){
	//    cout << "ERROR: Cannot load query image in\n" << argv[1] << endl;
	//    return -1;
	//}
	//
	//// get contour lines from image
	//vector<Mat> contourLines;
	//// TO DO !!!
	//binThreshold = 120;
	//numOfErosions = 2;
	//getContourLine(query, contourLines, binThreshold, numOfErosions);
	//
	//cout << "Found " << contourLines.size() << " object candidates" << endl;

	//// just to visualize classification result
	//Mat result(query.rows, query.cols, CV_8UC3);
	//vector<Mat> tmp;
	//tmp.push_back(query);
	//tmp.push_back(query);
	//tmp.push_back(query);
	//merge(tmp, result);

	//// loop through all contours found
	//int i=1;
	//// TO DO !!!
	//detThreshold = 120;
	//for(vector<Mat>::iterator c = contourLines.begin(); c != contourLines.end(); c++, i++){

	//    cout << "Checking object candidate no " << i << " :\t";
	//  
	//    for(int i=0; i < c->cols; i++){
	//	result.at<Vec3b>(c->at<Vec2i>(0,i)[1], c->at<Vec2i>(0,i)[0]) = Vec3b(255,0,0);
	//    }
	//    showImage(result, "Current Object", 0);
	//    
	//    // if fourier descriptor has too few components (too small contour), then skip it
	//    if (c->cols < steps*2){
	//	cout << "Too less boundary points" << endl;
	//	continue;
	//    }
	//    
	//    // calculate fourier descriptor
	//    Mat fd = makeFD(*c);
	//    
	//    // normalize fourier descriptor
	//    Mat fd_norm = normFD(fd, steps);
	//    
	//    // compare fourier descriptors
	//    double err1 = norm(fd_norm, fd1_norm)/steps;
	//    double err2 = norm(fd_norm, fd2_norm)/steps;

	//    // if similarity is too small, then reject
	//    if (min(err1, err2) > detThreshold){
	//	cout << "No class instance ( " << min(err1, err2) << " )" << endl;
	//	continue;
	//    }
	//    
	//    // otherwise: assign color according to class
	//    Vec3b col;
	//    if (err1 > err2){
	//	col = Vec3b(0,0,255);
	//	cout << "Class 2 ( " << err2 << " )" << endl;
	//    }else{
	//	col = Vec3b(0,255,0);
	//	cout << "Class 1 ( " << err1 << " )" << endl;
	//    }
	//    for(int i=0; i < c->cols; i++){
	//	result.at<Vec3b>(c->at<Vec2i>(0,i)[1], c->at<Vec2i>(0,i)[0]) = col;
	//    }
	//    
	//    // for intermediate results, use the following line
	//    showImage(result, "Current Object", 0);

	//}
	//// save result
	//imwrite("result.png", result);
	//// show final result
	//showImage(result, "Result", 0);
	
	return 0;
}

// normalize a given fourier descriptor
/*
fd		the given fourier descriptor
n		number of used frequencies (should be even)
out		the normalized fourier descriptor
*/
Mat normFD(Mat& fd, int n){
	//vector<Mat> channels;
	//split(fd, channels);
 //   // translation 
	//double t1 = -channels[0].at<float>(0,0)/fd.size().height;
	//double t2 = -channels[1].at<float>(0,0)/fd.size().height;
	//channels[0]+=fd.size().height * t1;
	//channels[1]+=fd.size().height * t1;
	//// scale 
	//double f1 = abs(channels[0].at<float>(1,0));
	//double f2 = abs(channels[1].at<float>(1,0));
	//channels[0]/=f1;
	//channels[1]/=f2;
	//// rotation
	//double f1 = abs(channels[0].at<float>(1,0));
	//double f2 = abs(channels[1].at<float>(1,0));
	//channels[0]/=f1;
	//channels[1]/=f2;
	//Mat tFD = Mat::zeros(fd.size(), CV_32F);
	//merge(channels,2,tFD);
	//fd.at<float>(0,0) = 0;
	//fd.at<float>(0,1) = 0;

	for(int i =0;i<2;++i) {
		for(int j=0;j<fd.size().height;++j){
			fd.at<float>(j,i) = abs(fd.at<float>(j,i)/abs(fd.at<float>(j,1)));
		}
	}

	return fd;
  
}

// calculates the (unnormalized) fourier descriptor from a list of points
/*
contour		1xN 2-channel matrix, containing N points (x in first, y in second channel)
out		fourier descriptor (not normalized)
*/
Mat makeFD(Mat& contour){

	Mat floatContour;
	contour.convertTo(floatContour,CV_32F);
	dft(floatContour,floatContour); 
	return floatContour;
}

// calculates the contour line of all objects in an image
/*
img		the input image
objList		vector of contours, each represented by a two-channel matrix
thresh		threshold used to binarize the image
k		number of applications of the erosion operator
*/
void getContourLine(Mat& img, vector<Mat>& objList, int thresh, int k){

	threshold(img, img, thresh, 255,THRESH_BINARY);
	dilate(img,img,Mat(), Point(-1, -1), k);
	cv::findContours(img,objList,CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
}

// plot fourier descriptor
/*
fd	the fourier descriptor to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/
void plotFD(Mat& fd, string win, double dur){
	// use copy for normalization
    Mat tempDisplay = fd.clone();
    if (fd.channels() == 1) normalize(fd, tempDisplay, 0, 255, CV_MINMAX);
    namedWindow( win.c_str(), CV_WINDOW_AUTOSIZE );
    imshow( win.c_str(), tempDisplay );
    // wait
    if (dur>=0) waitKey(dur);
}

// shows the image
/*
img	the image to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/
void showImage(Mat& img, string win, double dur){
  
    // use copy for normalization
    Mat tempDisplay = img.clone();
    if (img.channels() == 1) normalize(img, tempDisplay, 0, 255, CV_MINMAX);
    // create window and display omage
    namedWindow( win.c_str(), CV_WINDOW_AUTOSIZE );
    imshow( win.c_str(), tempDisplay );
    // wait
    if (dur>=0) waitKey(dur);
    
}