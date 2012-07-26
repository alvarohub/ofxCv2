#include "ofxCv/Helpers.h"
#include "ofxCv/Utilities.h"

namespace ofxCv {

	using namespace cv;
	
	void loadImage(Mat& mat, string filename) {
		mat = imread(ofToDataPath(filename));
	}
	
	void saveImage(Mat mat, string filename) {
		imwrite(ofToDataPath(filename), mat);
	}
	
	void loadMat(Mat& mat, string filename) {
		FileStorage fs(ofToDataPath(filename), FileStorage::READ);
		fs["Mat"] >> mat;
	}
	
	void saveMat(Mat mat, string filename) {
		FileStorage fs(ofToDataPath(filename), FileStorage::WRITE);
		fs << "Mat" << mat;
	}
	
	ofMatrix4x4 makeMatrix(Mat rotation, Mat translation) {
		Mat rot3x3;
		if(rotation.rows == 3 && rotation.cols == 3) {
			rot3x3 = rotation;
		} else {
			Rodrigues(rotation, rot3x3);
		}
		double* rm = rot3x3.ptr<double>(0);
		double* tm = translation.ptr<double>(0);
		return ofMatrix4x4(rm[0], rm[3], rm[6], 0.0f,
                           rm[1], rm[4], rm[7], 0.0f,
                           rm[2], rm[5], rm[8], 0.0f,
                           tm[0], tm[1], tm[2], 1.0f);
	}
	
	void drawMat(Mat& mat, float x, float y) {
		drawMat(mat, x, y, mat.cols, mat.rows);
	}
	
	void drawMat(Mat& mat, float x, float y, float width, float height) {
		int glType;
		Mat buffer;
		if(mat.depth() != CV_8U) {
			mat.convertTo(buffer, CV_8U);
		} else {
			buffer = mat;
		}
		if(mat.channels() == 1) {
			glType = GL_LUMINANCE;
		} else {
			glType = GL_RGB;
		}
		ofTexture tex;
		int w = buffer.cols;
		int h = buffer.rows;
		tex.allocate(w, h, glType);
		tex.loadData(buffer.ptr(), w, h, glType);
		tex.draw(x, y, width, height);
	}
    
    // added:
    void showMatData(Mat& mat) {
        if ((mat.rows<15)&&(mat.cols<15)) { // otherwise too large...
            cout << endl;
            for (int i=0; i<mat.rows; i++) {
                for (int j=0; j<mat.cols; j++) {
                    cout<< mat.at<float>(i,j) << " ";
                }
                cout << endl;
            }
        }
        
    }
	
	void applyMatrix(const ofMatrix4x4& matrix) {
		glMultMatrixf((GLfloat*) matrix.getPtr());
	}
	
	ofVec2f findMaxLocation(Mat& mat) {
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		minMaxLoc(mat, &minVal, &maxVal, &minLoc, &maxLoc);
		return ofVec2f(maxLoc.x, maxLoc.y);
	}
	
	void getBoundingBox(ofImage& img, ofRectangle& box, int thresh, bool invert) {
		Mat mat = toCv(img);
		int flags = (invert ? THRESH_BINARY_INV : THRESH_BINARY);
		
		Mat rowMat = meanRows(mat);
		threshold(rowMat, rowMat, thresh, 255, flags);
		box.y = findFirst(rowMat, 255);
		box.height = findLast(rowMat, 255);
		box.height -= box.y;
		
		Mat colMat = meanCols(mat);
		threshold(colMat, colMat, thresh, 255, flags);
		box.x = findFirst(colMat, 255);
		box.width = findLast(colMat, 255);
		box.width -= box.x;
	}
	
	int forceOdd(int x) {
		return (x / 2) * 2 + 1;
	}
	
	int findFirst(const Mat& arr, unsigned char target) {
		for(int i = 0; i < arr.rows; i++) {
			if(arr.at<unsigned char>(i) == target) {
				return i;
			}
		}
		return 0;
	}
	
	int findLast(const Mat& arr, unsigned char target) {
		for(int i = arr.rows - 1; i >= 0; i--) {
			if(arr.at<unsigned char>(i) == target) {
				return i;
			}
		}
		return 0;
	}
	
	Mat meanCols(const Mat& mat) {
		Mat colMat(mat.cols, 1, mat.type());
		for(int i = 0; i < mat.cols; i++) {
			colMat.row(i) = mean(mat.col(i));
		}	
		return colMat;
	}
	
	Mat meanRows(const Mat& mat) {
		Mat rowMat(mat.rows, 1, mat.type());
		for(int i = 0; i < mat.rows; i++) {
			rowMat.row(i) = mean(mat.row(i));
		}
		return rowMat;
	}
	
	Mat sumCols(const Mat& mat) {
		Mat colMat(mat.cols, 1, CV_32FC1);
		for(int i = 0; i < mat.cols; i++) {
			colMat.row(i) = sum(mat.col(i));
		}	
		return colMat;
	}
	
	Mat sumRows(const Mat& mat) {
		Mat rowMat(mat.rows, 1, CV_32FC1);
		for(int i = 0; i < mat.rows; i++) {
			rowMat.row(i) = sum(mat.row(i));
		}
		return rowMat;
	}
	
	Mat minCols(const Mat& mat) {
		Mat colMat(mat.cols, 1, CV_32FC1);
		double minVal, maxVal;
		for(int i = 0; i < mat.cols; i++) {
			minMaxLoc(mat.col(i), &minVal, &maxVal); 
			colMat.row(i) = minVal;
		}	
		return colMat;
	}
	
	Mat minRows(const Mat& mat) {
		Mat rowMat(mat.rows, 1, CV_32FC1);
		double minVal, maxVal;
		for(int i = 0; i < mat.rows; i++) {
			minMaxLoc(mat.row(i), &minVal, &maxVal); 
			rowMat.row(i) = minVal;
		}
		return rowMat;
	}
	
	Mat maxCols(const Mat& mat) {
		Mat colMat(mat.cols, 1, CV_32FC1);
		double minVal, maxVal;
		for(int i = 0; i < mat.cols; i++) {
			minMaxLoc(mat.col(i), &minVal, &maxVal); 
			colMat.row(i) = maxVal;
		}	
		return colMat;
	}
	
	Mat maxRows(const Mat& mat) {
		Mat rowMat(mat.rows, 1, CV_32FC1);
		double minVal, maxVal;
		for(int i = 0; i < mat.rows; i++) {
			minMaxLoc(mat.row(i), &minVal, &maxVal); 
			rowMat.row(i) = maxVal;
		}
		return rowMat;
	}
	
	void drawHighlightString(string text, ofPoint position, ofColor background, ofColor foreground) {
		drawHighlightString(text, position.x, position.y, background, foreground);
	}
	
	void drawHighlightString(string text, int x, int y, ofColor background, ofColor foreground) {
		vector<string> lines = ofSplitString(text, "\n");
		int textLength = 0;
		for(int i = 0; i < lines.size(); i++) {
			// tabs are not rendered
			int tabs = count(lines[i].begin(), lines[i].end(), '\t');
			int curLength = lines[i].length() - tabs;
			// after the first line, everything is indented with one space
			if(i > 0) {
				curLength++;
			}
			if(curLength > textLength) {
				textLength = curLength;
			}
		}
		
		int padding = 4;
		int fontSize = 8;
		float leading = 1.7;
		int height = lines.size() * fontSize * leading - 1;
		int width = textLength * fontSize;
	
		glPushAttrib(GL_DEPTH_BUFFER_BIT);
		glDisable(GL_DEPTH_TEST);
		ofPushStyle();
		ofSetColor(background);
		ofFill();
		ofRect(x, y, width + 2 * padding, height + 2 * padding);
		ofSetColor(foreground);
		ofNoFill();
		ofPushMatrix();
		ofTranslate(padding, padding);
		ofDrawBitmapString(text, x + 1, y + fontSize + 2);
		ofPopMatrix();
		ofPopStyle();
		glPopAttrib();
	}

}