/*
 wrappers provide an easy-to-use interface to OpenCv functions when using data
 from openFrameworks.
 
 in ofxOpenCv, these were methods of ofxCvImage. for completeness, we need:
 ROI methods (set, get, reset)
 pixel manipulation (set, +, -, *, /)
 filtering (erode, dilate, blur, gaussian, invert, stretch contrast, range)
 transform (resize, mirror, translate, rotate, scale, abritrary combo)
 undistort, remap
 count nonzero pixels
 */

#pragma once

#include "ofMain.h"
#include "opencv2/opencv.hpp"
#include "ofxCv/Utilities.h"

namespace ofxCv {
	
	using namespace cv;
	
	// wrapThree are based on functions that operate on three Mat objects.
	// the first two are inputs, and the third is an output. for example,
	// the min() function: min(x, y, result) will calculate the per-element min
	// between x and y, and store that in result. both y and result need to
	// match x in dimensions and type. while wrapThree functions will use
	// imitate() to make sure your data is allocated correctly, you shouldn't
	// epect the function to behave properly if you haven't already allocated
	// your y argument. in general, OF images contain noise when newly allocated
	// so the result will also contain that noise.
#define wrapThree(name) \
template <class X, class Y, class Result>\
void name(X& x, Y& y, Result& result) {\
imitate(y, x);\
imitate(result, x);\
Mat xMat = toCv(x);\
Mat yMat = toCv(y);\
Mat resultMat = toCv(result);\
cv::name(xMat, yMat, resultMat);\
}
	wrapThree(max);
	wrapThree(min);
	wrapThree(multiply);
	wrapThree(divide);
	wrapThree(add);
	wrapThree(subtract);
	wrapThree(absdiff);
	wrapThree(bitwise_and);
	wrapThree(bitwise_or);
	wrapThree(bitwise_xor);
	
	// also useful for taking the average/mixing two images
	template <class X, class Y, class R>
	void lerp(X& x, Y& y, R& result, float amt = .5) {
		imitate(result, x);
		Mat xMat = toCv(x);
		Mat yMat = toCv(y);
		Mat resultMat = toCv(result);
		if(yMat.cols == 0) {
			copy(x, result);
		} else if(xMat.cols == 0) {
			copy(y, result);
		} else {
			cv::addWeighted(xMat, amt, yMat, 1. - amt, 0., resultMat);
		}
	}
	
	// normalize the min/max to [0, max for this type] out of place
	template <class S, class D>
	void normalize(S& src, D& dst) {
		imitate(dst, src);
		Mat srcMat = toCv(src);
		Mat dstMat = toCv(dst);
		cv::normalize(srcMat, dstMat, 0, getMaxVal(getDepth(dst)), NORM_MINMAX);
	}
	
	// normalize the min/max to [0, max for this type] in place
	template <class SD>
	void normalize(SD& srcDst) {
		normalize(srcDst, srcDst);
	}
	
	// threshold out of place
	template <class S, class D>
	void threshold(S& src, D& dst, float thresholdValue, bool invert = false) {
		imitate(dst, src);
		Mat srcMat = toCv(src);
		Mat dstMat = toCv(dst);
		int thresholdType = invert ? THRESH_BINARY_INV : THRESH_BINARY;
		float maxVal = getMaxVal(dstMat);
		cv::threshold(srcMat, dstMat, thresholdValue, maxVal, thresholdType);
	}
	
	// threshold in place
	template <class SD>
	void threshold(SD& srcDst, float thresholdValue, bool invert = false) {
		ofxCv::threshold(srcDst, srcDst, thresholdValue, invert);
	}
	
	// CV_RGB2GRAY, CV_HSV2RGB, etc. with [RGB, BGR, GRAY, HSV, HLS, XYZ, YCrCb, Lab, Luv]
	// you can convert whole images...
	template <class S, class D>
	void convertColor(S& src, D& dst, int code) {
		// cvtColor allocates Mat for you, but we need this to handle ofImage etc.
		int targetChannels = getTargetChannelsFromCode(code);
		imitate(dst, src, getCvImageType(targetChannels, getDepth(src)));
		Mat srcMat = toCv(src);
		Mat dstMat = toCv(dst);
		cvtColor(srcMat, dstMat, code);
	}
	// ...or single colors.
	Vec3b convertColor(Vec3b color, int code);
	ofColor convertColor(ofColor color, int code);
	
	int forceOdd(int x);
	
	// Gaussian blur
	template <class S, class D>
	void blur(S& src, D& dst, int size) {
		imitate(dst, src);
		size = forceOdd(size);
		Mat srcMat = toCv(src);
		Mat dstMat = toCv(dst);
		cv::GaussianBlur(srcMat, dstMat, cv::Size(size, size), 0, 0);
	}
	
	// in-place Gaussian blur
	template <class SD>
	void blur(SD& srcDst, int size) {
		ofxCv::blur(srcDst, srcDst, size);
	}
	
	// Median blur
	template <class S, class D>
	void medianBlur(S& src, D& dst, int size) {
		imitate(dst, src);
		size = forceOdd(size);
		Mat srcMat = toCv(src);
		Mat dstMat = toCv(dst);
		cv::medianBlur(srcMat, dstMat, size);
	}
	
	// in-place Median blur
	template <class SD>
	void medianBlur(SD& srcDst, int size) {
		ofxCv::medianBlur(srcDst, srcDst, size);
	}
	
	// Canny edge detection assumes your input and output are grayscale 8-bit
	template <class S, class D>
	void Canny(S& src, D& dst, double threshold1, double threshold2, int apertureSize=3, bool L2gradient=false) {
		imitate(dst, src, CV_8UC1);
		Mat srcMat = toCv(src);
		Mat dstMat = toCv(dst);
		cv::Canny(srcMat, dstMat, threshold1, threshold2, apertureSize, L2gradient);
	}	
	
	// dst does not imitate src
	template <class S, class D>
	void warpPerspective(S& src, D& dst, vector<Point2f>& dstPoints, int flags = INTER_LINEAR) {
		Mat srcMat = toCv(src);
		Mat dstMat = toCv(dst);
		int w = srcMat.cols;
		int h = srcMat.rows;
		vector<Point2f> srcPoints(4);
		srcPoints[0] = Point2f(0, 0);
		srcPoints[1] = Point2f(w, 0);
		srcPoints[2] = Point2f(w, h);
		srcPoints[3] = Point2f(0, h);
		Mat transform = getPerspectiveTransform(&srcPoints[0], &dstPoints[0]);
		warpPerspective(srcMat, dstMat, transform, dstMat.size(), flags);
	}
	
	// dst does not imitate src
	template <class S, class D>
	void unwarpPerspective(S& src, D& dst, vector<Point2f>& srcPoints, int flags = INTER_LINEAR) {
		Mat srcMat = toCv(src);
		Mat dstMat = toCv(dst);
		int w = dstMat.cols;
		int h = dstMat.rows;
		vector<Point2f> dstPoints(4);
		dstPoints[0] = Point2f(0, 0);
		dstPoints[1] = Point2f(w, 0);
		dstPoints[2] = Point2f(w, h);
		dstPoints[3] = Point2f(0, h);
		Mat transform = getPerspectiveTransform(&srcPoints[0], &dstPoints[0]);
		warpPerspective(srcMat, dstMat, transform, dstMat.size(), flags);
	}
    
    // ** added (the reason is that we may want to compute the trasform matriz ONCE and for all...)
    // dst does not imitate src
	template <class D>
    void computeUnwarpTransformation(vector<Point2f>& srcPoints, D& dst, Mat& unwarpTransform, int flags = INTER_LINEAR) {
		//Mat srcMat = toCv(src);
		Mat dstMat = toCv(dst);
		int w = dstMat.cols;
		int h = dstMat.rows;
		vector<Point2f> dstPoints(4);
		dstPoints[0] = Point2f(0, 0);
		dstPoints[1] = Point2f(w, 0);
		dstPoints[2] = Point2f(w, h);
		dstPoints[3] = Point2f(0, h);
		unwarpTransform = getPerspectiveTransform(&srcPoints[0], &dstPoints[0]);
		//warpPerspective(srcMat, dstMat, transform, dstMat.size(), flags);
    }
    
    // ** added (the reason is that we may want to compute the trasform matriz ONCE and for all...)
    // dst does not imitate src
	template <class S>
    void computeWarpTransformation(S& src, vector<Point2f>& dstPoints, Mat& warpTransform, int flags = INTER_LINEAR) {
		Mat srcMat = toCv(src);
		//Mat dstMat = toCv(dst);
		int w = srcMat.cols;
		int h = srcMat.rows;
		vector<Point2f> srcPoints(4);
		srcPoints[0] = Point2f(0, 0);
		srcPoints[1] = Point2f(w, 0);
		srcPoints[2] = Point2f(w, h);
		srcPoints[3] = Point2f(0, h);
		warpTransform = getPerspectiveTransform(&srcPoints[0], &dstPoints[0]);
		//warpPerspective(srcMat, dstMat, transform, dstMat.size(), flags);
    }

	// dst does not imitate src
	template <class S, class D>
	void warpPerspective(S& src, D& dst, Mat& transform, int flags = INTER_LINEAR) {
		Mat srcMat = toCv(src);
		Mat dstMat = toCv(dst);
		warpPerspective(srcMat, dstMat, transform, dstMat.size(), flags);
	}
    
    // added: apply homography transform to a single point (need to template it?):
    void transformMyPoint(Point2f& src, Point2f& dst, Mat& transform);
	
	template <class D>
	void fillPoly(vector<cv::Point>& points, D& dst) {
		cv::Mat dstMat = toCv(dst);
		const cv::Point* ppt[1] = { &(points[0]) };
		int npt[] = { points.size() };
		dstMat.setTo(Scalar(0));
		fillPoly(dstMat, ppt, npt, 1, Scalar(255));
	}
	
	// older wrappers, need to be templated..	
	// for contourArea()/arcLength(), see ofPolyline::getArea()/getPerimiter()
	// not sure if these three need to be templated. convexHull returning an
	// ofPolyline when given an ofPolyline is the key factor...
	ofPolyline convexHull(ofPolyline& polyline);
	cv::RotatedRect minAreaRect(ofPolyline& polyline);
	cv::RotatedRect fitEllipse(ofPolyline& polyline);
	
	void invert(ofImage& img);
	void rotate(ofImage& source, ofImage& destination, double angle, unsigned char fill = 0, int interpolation = INTER_LINEAR);
	void autorotate(ofImage& original, ofImage& thresh, ofImage& output, float* rotation = NULL);
	void autothreshold(ofImage& original, ofImage& thresh, bool invert = false);
	void autothreshold(ofImage& original, bool invert = false);
	//void threshold(FloatImage& img, float value, bool invert = false);
	//void threshold(FloatImage& original, FloatImage& thresh, float value, bool invert = false);
	//void matchRegion(ofImage& source, ofRectangle& region, ofImage& search, FloatImage& result);
	void matchRegion(Mat& source, ofRectangle& region, Mat& search, Mat& result);
	//void convolve(ofImage& source, FloatImage& kernel, ofImage& destination);
	//void convolve(ofImage& img, FloatImage& kernel);
	void resize(ofImage& source, ofImage& destination, int interpolation = INTER_LINEAR); // options: INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER LANCZOS4
	void resize(ofImage& source, ofImage& destination, float xScale, float yScale, int interpolation = INTER_LINEAR);
	
}
