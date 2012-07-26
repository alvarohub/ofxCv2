/*
 this class handles per-camera intrinsic calibration and undistortion.
 given a series of chessboard images, it will calculate the intrinsics.
 to use it:
 
 0 either load() from a yml file (skip to 5),
 or do the calibration as follows
 1 set the board and physical square size of the chess board. whatever
 if your squares are in mm, your focal length will also be in mm.
 2 add() each image containing a chess board
 3 when all the images are added, call calibrate()
 4 now you can save() a yml calibration file 
 5 now you can undistort() incoming images. 
 
 to do inter-camera (extrinsics) calibration, you need to first calibrate
 each camera individually. then use getTransformation to determine the
 rotation and translation from camera to another.
 */
 
#pragma once

#include "ofxCv.h"

namespace ofxCv {

	using namespace cv;    
    
    // ========================== THE INTRINSICS of the CAMERA or PROJECTOR =========================================
    
	class Intrinsics {
	public:
		void setup(Mat cameraMatrix, cv::Size imageSize, cv::Size sensorSize = cv::Size(0, 0));
		Mat getCameraMatrix() const;
		cv::Size getImageSize() const;
		cv::Size getSensorSize() const;
		cv::Point2d getFov() const;
		double getFocalLength() const;
		double getAspectRatio() const;
		Point2d getPrincipalPoint() const;
        
         
	protected:
		Mat cameraMatrix;
		cv::Size imageSize; 
        cv::Size sensorSize;
		cv::Point2d fov;
		double focalLength, aspectRatio;
		Point2d principalPoint;
	};

    // ============= CAMERA MODEL CLASS ==========================================================================================================
    // Including EXTRINSICS with respect to some special reference frame (another camera, etc). In fact, this extrinsics parameters would be loaded 
    // by the stereo object (with respect to the camera set as "origin"). 
    class CameraModel : public ofNode {
    public:
        
        void setImagerResolution(cv::Size imgRes) {imagerRes=imgRes;}
        cv::Size getImagerResolution() const {return imagerRes;}
        
        Mat distCoeffs;
        Intrinsics distortedIntrinsics;
		Intrinsics undistortedIntrinsics;
        
		void setOpenGLProjectionMatrix(bool distorted= false, float nearDist = 10., float farDist = 10000.) const;
        
    private:
        cv::Size imagerRes; // size of the added image in case of camera, or size of the projector imager
        
        Mat Extrinsics; // unnecessary perhaps, because it is a subclass of ofNode?
    };
    
    // ============= CALIBRATION SHAPE CLASS=================================================================================================
    // This contains information about the particular shape to detect. 
    enum CalibrationPattern {CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID}; 
    class CalibrationShape {
    public:
        CalibrationShape();
        
        void setPatternSize(int xCount, int yCount); // number of corners/circles in x and y directoin
		void setSquareSize(float squareSize);        // distance between corner/circles
		void setOrigin(int x, int y) {posX=x; posY=y;};
        void setSubpixelSize(int subpixelSize);		/// set this to the pixel size of your smallest square. default is 11
        void setPatternType(CalibrationPattern typeboard) {typePattern=typeboard;};
        void loadCalibrationShape(string filename);
        
        // void setPatternColor() .... < --- this will be useful when preprocessing of the image will include color segmentation. 
        
        cv::Size getPatternSize() const;
		float getSquareSize() const;
        
        CalibrationPattern typePattern;
        cv::Size patternSize, subpixelSize;
        int posX, posY; // origin of the pattern
		float squareSize;
    };
    
   
    // ============= CALiBRATION CLASS ============================================================================================
	class Calibration { // << NOTE: the CAMERA MODEL should be a subclass of ofNode, not the Calibration object itself...
	public:
		Calibration();
		
        // load and save calibration parameters:
        // - suggestion: saving features for the boards should be optional, or done in a separate method. 
		void save(string filename, bool absolute = false) const;
		void load(string filename, bool absolute = false);
		
        // Wrappers to set the CalibrationShape to be recognized:
		//void setPatternSize(int xCount, int yCount) {myPatternShape.setPatternSize(xCount, yCount);}
		//void setPatternSquareSize(float squareSize) {myPatternShape.setSquareSize(squareSize);}
		//void setPatternSubpixelSize(int subpixelSize) {myPatternShape.setSubpixelSize(subpixelSize);}
        //void setPatternType(CalibrationPattern typeboard) {myPatternShape.typePattern=typeboard;};
        void loadCalibrationShape(string filename) {myPatternShape.loadCalibrationShape(filename);}
       
        void setImagerResolution(cv::Size _imgrsize) {myCameraModel.setImagerResolution(_imgrsize);} // note: in general for the camera, the size of the image acquired is the size of the imager, but this is not the case for the projector. In fact, it can (and is often the case) for the camera too. So, I suggest that one decide ONCE about the camera or projector imager resolution (will change the focal length!). 
        
        
        void addImageToProcess(Mat img); // this will preprocess the image (color detection, etc) and store it in procImg 
        
		//note: treatment is done on the preprocessed image, which is filled by calling addImageToProcess. 
        bool generateCandidateImageObjectPoints();
        bool generateCandidateObjectPoints(Calibration& bestCalibratedCamera);
        bool candidatePatternDetected(); // this indicates that we could properly set image/object points for the added image. 
        
        
		bool clean(float minReprojectionError = 1.f); // delete the boards with reprojection error > minReprojectionError
        bool simultaneousClean(Calibration& calibrationObject, float minReprojectionError = 2.f); // note: the reprojection error test is done only on *this* object
        void deleteAllBoards(); // delete all the boards
        void deleteLastBoard(); // delete last board
        
		bool calibrate();
		bool calibrateFromDirectory(string directory);
        
        bool computeBoardPose(int i); // from stored arrays of boards
        void computeCandidateBoardPose(); // from candidate image rot/trans arrays
        
		void undistort(Mat img);
		void undistort(Mat src, Mat dst);
		
		ofVec2f undistort(ofVec2f &src);
		void undistort(vector<ofVec2f> &src, vector<ofVec2f> &dst);
		
		// Stereo calibration (extrinsics): 
        // For two cameras:
        bool getTransformation(Calibration& dst, Mat& rotation, Mat& translation);
		// For a projector and a camera:
        bool stereoCalibrationCameraProjector(Calibration& bestCamera, Mat& rotation, Mat& translation);
        
		float getReprojectionError() const;
		float getReprojectionError(int i) const;
		
		const Intrinsics& getDistortedIntrinsics() const;
		const Intrinsics& getUndistortedIntrinsics() const;
		Mat getDistCoeffs() const;
		
		// if you want a wider fov, say setFillFrame(false) before load() or calibrate()
		void setFillFrame(bool fillFrame);
		
        // findBoard uses the current image and myPatternShape to find the corners/circles image points:
		bool findBoard(Mat img, vector<Point2f> &pointBuf, bool refine=true);
        
        int size() const; // <<-- number of calibration boards used so far (after cleaning too)
        
        static vector<Point2f> createImagePointsFromPattern(const CalibrationShape& patternShape); // usually from projector CalibrationShape pattern
        static vector<Point3f> createObjectPointsFromPattern(const CalibrationShape& patternShape);// usually from camera CalibrationShape pattern 
        static vector<Point3f> createObjectPointsDynamic(const Point3f& posOrigin, const Point3f& axisX, const Point3f& axisY, const CalibrationShape& patternShape);
        
        void setCandidateImagePoints(); // this just sets the image points using createImagePointsFromPattern (usually for the projector).
        void setCandidateImagePoints(vector<Point2f> newImageSet); // overloaded function to display an arbitary set of image points 
        
        void setCandidateObjectPoints(); // this just sets the object points from createObjectPointsFromPattern (usually for camera)
        void setCandidateObjectPoints(vector<Point3f> newObjectSet);
        
        // Add candidate board data to the vector arrays:
        void addCandidateImagePoints();
        void addCandidateObjectPoints(); 
        void addCandidateBoardPose();
        
        void updateReprojectedImagePoints();
        //void updateBackProjectedObjectPoints();
        
        //static vector<Point3f> createObjectPointsForProjector(Mat img, CalibrationShape patternShape, Calibration bestCalibrationCamera); // we will backproject the points using the camera with better reprojection error. 
        bool backProject( const vector<Point2f>& imgPt, vector<Point3f>& worldPt, const Mat& boardRot, const Mat& boardTrans );
        
        // In the future, use MATRICES instead of rot and trans? (Extrinsic matrix)
        vector<Point2f> createImagePointsFrom3dPoints(vector<Point3f>& objpts,  
                                                      const Mat& rot, const Mat& trans, 
                                                      const Mat& rotExt=Mat::zeros(3, 1, CV_32F), const Mat& transExt=Mat::zeros(3, 1, CV_32F));
      
        
        // Drawing functions:
        
        // (1) Images:
        void drawPreprocessedImage(int, int, int, int); // essentially, this is for checking and interactively changing color/threshold for board detection
               
        //(2) Points (including image and reprojected object points, NOT using openGL):
        bool drawCandidateReprojection(int px, int py, int sx, int sy, ofColor col); // using the candidateObjectPoints
        bool drawCandidateImagePoints(int px, int py, int sx, int sy, ofColor col);
        //bool drawLatestProjectorPattern() {return(drawLatestImagePoints(1440, 0, 1024, 768, ofColor(0,0,0)));}// << this is to put in the projector class 
        
        // this function drawis from the candidateImagePoints array (the auxiliary points), NOT the latest image points in the list. 
        // It is just equal to drawCandidateImagePoints, but using special parameters. 
        bool drawCandidateProjectorPattern(int px=1440, int py=0, int sx=800, int sy=600, ofColor col=ofColor(255,255,255), float sizecircle=1);
        
        // This will be used to display points with the projector, in particular after using the method createImagePointsFrom3dPoints:
        bool drawArbitraryImagePoints(int px, int py, int sx, int sy, vector<Point2f>& imPoints, ofColor col, float sizecircle);
        
        bool drawCandidateAxis(int px, int py, int sx, int sy);
        
        //(3) 3D (needs openGL perspective matrix setting)
        void setOpenGLProjectionMatrix() {myCameraModel.setOpenGLProjectionMatrix();} // using distorted or undistorted?
        
		void customDraw(); // because it was an ofNode child... but I don't see the point (yet)
		void draw(int i) const;
      
		void draw3d() const;
		void draw3d(int i) const;
		
		//const bool &isReady;
		bool isReady();

        
        // DATA ================================================================================
        
        // A "BOARD" is defined in THIS object coordinate frames as the set of these quantitites:
        // (We could have a structure or class "BoardData" and make a vector of boards, with functions to delete, add...)
        vector<vector<Point2f> > imagePoints;
		vector<vector<Point3f> > objectPoints;
        vector<Mat> boardRotations, boardTranslations; // <<-- store as a single matrix4x4 ?
		
        
        // Auxiliary arrays (before "confirming" that to the vectors):
        vector<Point2f> candidateImagePoints; // this are auxiliary image points that the projector project until they are detected; then these can be added to imagePoints. The idea is that these points can be modified in real time as we move the PRINTED board, in order to find a configuration in which BOTH patterns are simultaneously visible. 
		vector<Point3f> candidateObjectPoints;
        Mat candidateBoardRotation, candidateBoardTranslation; 
        
   
        // The patten shape (protected?)
		CalibrationShape myPatternShape;
        
         
	protected:
       
        bool patternDetected; // this will set when trying to detect the board AND successfully compute image/object points. 
        
        Mat procImg; // auxiliary image to find the corners/circles on the current added image using the myPatternShape characteristics (including color?)
		
		vector<float> perViewErrors;
        float reprojectionError;
        
		bool fillFrame;
		Mat undistortBuffer;
		Mat undistortMapX, undistortMapY;
        
		void updateReprojectionErrors();
		void updateUndistortion();
		
        CameraModel myCameraModel; // in fact this can represent a camera or a projector
		
		bool _isReady; // set to true when the object could be calibrated
	};
	
}