/*
 utilities are used internally by ofxCv, and make it easier to write code that
 can work with OpenCv and openFrameworks data.
 */

#pragma once

#include "ofMain.h"
#include "opencv2/opencv.hpp"

namespace ofxCv {
	
	using namespace cv;
	
	// 1 utility functions
	// for toCv/toOf the function signature reveals the behavior:
	// 1       Type& argument // shallow copy of the data
	// 2 const Type& argument // deep copy of the data
	// 3       Type  argument // deep copy of the data
	// style 1 is used when possible (for Mat conversion). style 2 is used when
	// dealing with a lot of data that can't/shouldn't be shallow copied. style 3
	// is used for small objects where the compiler can optimize the copying if
	// necessary. the reference is avoided to make inline toCv/toOf use easier.
	
	// toCv functions
	Mat toCv(Mat& mat);
	template <class T> inline Mat toCv(ofPixels_<T>& pix) {
		return Mat(pix.getHeight(), pix.getWidth(), getCvImageType(pix), pix.getPixels(), 0);
	}
	template <class T> inline Mat toCv(ofImage_<T>& img) {
		return Mat(img.getHeight(), img.getWidth(), getCvImageType(img), img.getPixels(), 0);
	}
	Mat toCv(ofBaseHasPixels& img);
	Mat toCv(ofMesh& mesh);
	Point2f toCv(ofVec2f vec);
	Point3f toCv(ofVec3f vec);
     //Point2f toCv(ofVec3f vec); // this seems needed because ofPoint is ofVec3f, but I cannot overload a funciton by its return type...
	cv::Rect toCv(ofRectangle rect);
	vector<cv::Point2f> toCv(const ofPolyline& polyline);
	Scalar toCv(ofColor color); // might need more for other color types?
	
	// toOf functions
	ofVec2f toOf(Point2f point);
	ofVec3f toOf(Point3f point);
	ofRectangle toOf(cv::Rect rect);
	ofPolyline toOf(cv::RotatedRect rect);
	template <class T> inline ofPolyline toOf(const vector<cv::Point_<T> >& contour) {
		ofPolyline polyline;
		polyline.resize(contour.size());
		for(int i = 0; i < contour.size(); i++) {
			polyline[i].x = contour[i].x;
			polyline[i].y = contour[i].y;
		}
		polyline.close();
		return polyline;
	}
    // to add: toOF for meshes:
    /*
    template <class T> inline ofMesh toOf(const vector<cv::Point_<T> >& contour) {
		ofPolyline polyline;
		polyline.resize(contour.size());
		for(int i = 0; i < contour.size(); i++) {
			polyline[i].x = contour[i].x;
			polyline[i].y = contour[i].y;
		}
		polyline.close();
		return polyline;
	}
     */
	template <class T>
	void toOf(Mat mat, ofPixels_<T>& pixels) {
		pixels.setFromExternalPixels(mat.ptr<T>(), mat.cols, mat.rows, mat.channels());
	}
	
	// these functions are for accessing Mat, ofPixels and ofImage consistently.
	// they're very important for imitate().
	
	// width, height
	template <class T> inline int getWidth(T& src) {return src.getWidth();}
	template <class T> inline int getHeight(T& src) {return src.getHeight();}
	inline int getWidth(Mat& src) {return src.cols;}
	inline int getHeight(Mat& src) {return src.rows;}
	
	// depth
	inline int getDepth(Mat& mat) {
		return mat.depth();
	}
	template <class T> inline int getDepth(ofPixels_<T>& pixels) {
		switch(pixels.getBytesPerChannel()) {
			case 4: return CV_32F;
			case 2: return CV_16U;
			case 1: default: return CV_8U;
		}
	}
	template <class T> inline int getDepth(ofBaseHasPixels_<T>& img) {
		return getDepth(img.getPixelsRef());
	}
	
	// channels
	inline int getChannels(int cvImageType) {
		return CV_MAT_CN(cvImageType);
	}
	inline int getChannels(ofImageType imageType) {
		switch(imageType) {
			case OF_IMAGE_COLOR_ALPHA: return 4;
			case OF_IMAGE_COLOR: return 3;
			case OF_IMAGE_GRAYSCALE: default: return 1;
		}
	}
	inline int getChannels(Mat& mat) {
		return mat.channels();
	}
	template <class T> inline int getChannels(ofPixels_<T>& pixels) {
		return pixels.getNumChannels();
	}
	template <class T> inline int getChannels(ofBaseHasPixels_<T>& img) {
		return getChannels(img.getPixelsRef());
	}
	
	// image type
	inline int getCvImageType(int channels, int depth = CV_8U) {
		return CV_MAKETYPE(depth, channels);
	}
	inline int getCvImageType(ofImageType imageType, int depth = CV_8U) {
		return CV_MAKETYPE(depth, getChannels(imageType));
	}
	template <class T> inline int getCvImageType(T& img) {
		return CV_MAKETYPE(getDepth(img), getChannels(img));
	}
	inline ofImageType getOfImageType(int cvImageType) {
		switch(getChannels(cvImageType)) {
			case 4: return OF_IMAGE_COLOR_ALPHA;
			case 3: return OF_IMAGE_COLOR;
			case 1: default: return OF_IMAGE_GRAYSCALE;
		}
	}
	template <class T> inline ofImageType getOfImageType(T& img) {
		switch(getChannels(img)) {
			case 4: return OF_IMAGE_COLOR_ALPHA;
			case 3: return OF_IMAGE_COLOR;
			case 1: default: return OF_IMAGE_GRAYSCALE;
		}
	}
	
	// allocationg
	template <class T> inline void allocate(T& img, int width, int height, int cvType) {
		img.allocate(width, height, getOfImageType(cvType));
	}
	inline void allocate(Mat& img, int width, int height, int cvType) {
		img.create(height, width, cvType);
	}
	inline void allocate(ofVideoPlayer& img, int width, int height, int cvType) {}
	inline void allocate(ofVideoGrabber& img, int width, int height, int cvType) {}
	
	// imitate() is good for preparing buffers
	// it's like allocate(), but uses the size and type of the original as a reference
	// like allocate(), the image being allocated is the first argument	
	
	// this version copies size, but manually specifies image type
	template <class M, class O> void imitate(M& mirror, O& original, int originalCvImageType) {
		int mw = getWidth(mirror);
		int mh = getHeight(mirror);
		int ow = getWidth(original);
		int oh = getHeight(original);
		int mt = getCvImageType(mirror);
		if(mw != ow || mh != oh || mt != originalCvImageType) {
			allocate(mirror, ow, oh, originalCvImageType);
		}
	}
	
	// this version copies size and image type
	template <class M, class O> void imitate(M& mirror, O& original) {
		imitate(mirror, original, getCvImageType(original));
	}
	
	// maximum possible values for that depth or matrix
	float getMaxVal(int depth);
	float getMaxVal(const Mat& mat);
	int getTargetChannelsFromCode(int conversionCode);
	
	// cross-toolkit, cross-bitdepth copying
	// should this do conversion? or should be handle conversion in convertColor?
	// or convert that handles color or bitdepth?
	template <class S, class D>
	void copy(S& src, D& dst) {
		imitate(dst, src);
		Mat srcMat = toCv(src);
		Mat dstMat = toCv(dst);
		if(srcMat.type() == dstMat.type()) {
			srcMat.copyTo(dstMat);
		} else {
			double alpha = getMaxVal(dstMat) / getMaxVal(srcMat);
			srcMat.convertTo(dstMat, dstMat.depth(), alpha);
		}
	}
}
