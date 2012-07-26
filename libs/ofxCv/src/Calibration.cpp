#include "ofxCv/Calibration.h"
#include "ofxCv/Helpers.h"
#include "ofFileUtils.h"

namespace ofxCv {
    
	using namespace cv;
	
    // ========================== THE INTRINSICS of the CAMERA or PROJECTOR =========================================
    
    void Intrinsics::setup(Mat cameraMatrix, cv::Size imageSize, cv::Size sensorSize) {
		this->cameraMatrix = cameraMatrix;
		this->imageSize = imageSize;
		this->sensorSize = sensorSize;
		
        //Computes useful camera characteristics (fov, focalLength, principalPoint and aspectRatio) from the 
        // camera matrix, the camera frame resolution and the physical sensor size:
		calibrationMatrixValues(cameraMatrix, imageSize, sensorSize.width, sensorSize.height,
                                fov.x, fov.y, focalLength, principalPoint, aspectRatio);
        
	}
	
	Mat Intrinsics::getCameraMatrix() const {
		return cameraMatrix;
	}
	
	cv::Size Intrinsics::getImageSize() const {
		return imageSize;
	}
	
	cv::Size Intrinsics::getSensorSize() const {
		return sensorSize;
	}
	
	cv::Point2d Intrinsics::getFov() const {
		return fov;
	}
	
	double Intrinsics::getFocalLength() const {
		return focalLength;
	}
	
	double Intrinsics::getAspectRatio() const {
		return aspectRatio;
	}
	
	Point2d Intrinsics::getPrincipalPoint() const {
		return principalPoint;
	}
	
    
    // ============= CAMERA MODEL CLASS ==========================================================================================================
	
    // Load openGL projection matrix:
	void CameraModel::setOpenGLProjectionMatrix(bool distorted, float nearDist, float farDist) const {
		//glViewport(0, 0, imageSize.width, imageSize.height);
        Intrinsics auxIntrinsics;
        if (distorted) // use distorted intrinsics:
            auxIntrinsics=distortedIntrinsics;
        else 
            auxIntrinsics=undistortedIntrinsics;
        
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		float w = auxIntrinsics.getImageSize().width;
		float h =  auxIntrinsics.getImageSize().height;
		float fx = auxIntrinsics.getCameraMatrix().at<double>(0, 0);
		float fy = auxIntrinsics.getCameraMatrix().at<double>(1, 1);
		float cx = auxIntrinsics.getPrincipalPoint().x;
		float cy = auxIntrinsics.getPrincipalPoint().y;
		glFrustum(
                  nearDist * (-cx) / fx, nearDist * (w - cx) / fx,
                  nearDist * (cy - h) / fy, nearDist * (cy) / fy,
                  nearDist, farDist);
		//glMatrixMode(GL_MODELVIEW);
		//glLoadIdentity();
		//gluLookAt(
        //          0, 0, 0,   // position of camera
        //          0, 0, 1,   // looking towards the point (0,0,1) 
        //          0, -1, 0); // orientation
	}
    
    // ============= CALIBRATIONSHAPE CLASS=================================================================================================
    CalibrationShape::CalibrationShape() :
    patternSize(cv::Size(10, 7)), squareSize(2.5), // based on Chessboard_A4.pdf, assuming world units are centimeters
    subpixelSize(cv::Size(11,11)), posX(0), posY(0)
    {
        typePattern=CHESSBOARD;
    }
    
    void CalibrationShape::loadCalibrationShape(string filename) {
        FileStorage settings(ofToDataPath(filename), FileStorage::READ);
        if(settings.isOpened()) {
            int xCount = settings["xCount"], yCount = settings["yCount"]; 
            int x= settings["posX"], y= settings["posY"]; // this will be useful for the projector pattern in particular
            float squareSize = settings["squareSize"];// in pixels in case of projector pattern size
            int pat_type=settings["patternType"];
            // int hexcolor=settings["patternColor"]; // <<-- to do in the future
            setPatternSize(xCount, yCount);
            setSquareSize(squareSize);
            setOrigin(x, y);
            switch(pat_type) {
                case 0:
                    setPatternType(CHESSBOARD);
                    break;
                case 1:
                    setPatternType(CIRCLES_GRID);
                    break;
                case 2:
                    setPatternType(ASYMMETRIC_CIRCLES_GRID);
                    break;
            }
        } 
    }
    
    void CalibrationShape::setPatternSize(int xCount, int yCount) {
		patternSize = cv::Size(xCount, yCount);
	}
	void CalibrationShape::setSquareSize(float squareSize) {
		this->squareSize = squareSize;
	}
    void CalibrationShape::setSubpixelSize(int subpixelSize) {
		subpixelSize = MAX(subpixelSize,2);
		this->subpixelSize = cv::Size(subpixelSize,subpixelSize);
	}
    
    cv::Size CalibrationShape::getPatternSize() const {
		return patternSize;
	}
	float CalibrationShape::getSquareSize() const {
		return squareSize;
	}
    
    // ============= CALiBRATION CLASS ============================================================================================
    
	Calibration::Calibration() :
    fillFrame(true), // default value for getOptimalNewCameraMatrix
    _isReady(false),
    reprojectionError(0.0), 
    patternDetected(false)
	{
		
	}
    
    // Save calibration parameters (along with the detected feature image points):
	void Calibration::save(string filename, bool absolute) const {
		if(!_isReady){
			ofLog(OF_LOG_ERROR, "Calibration::save() failed, because your calibration isn't ready yet!");
		}
		FileStorage fs(ofToDataPath(filename, absolute), FileStorage::WRITE);
		cv::Size imageSize = myCameraModel.distortedIntrinsics.getImageSize();
		cv::Size sensorSize = myCameraModel.distortedIntrinsics.getSensorSize();
		Mat cameraMatrix = myCameraModel.distortedIntrinsics.getCameraMatrix();
		fs << "cameraMatrix" << cameraMatrix;
		fs << "imageSize_width" << imageSize.width;
		fs << "imageSize_height" << imageSize.height;
		fs << "sensorSize_width" << sensorSize.width;
		fs << "sensorSize_height" << sensorSize.height;
		fs << "distCoeffs" << myCameraModel.distCoeffs;
		fs << "reprojectionError" << reprojectionError;
		fs << "features" << "[";
		for(int i = 0; i < imagePoints.size(); i++) { // iteration on the number of detected boards
			fs << "{:" << "points" << "[:"; 
			for( int j = 0; j < imagePoints[i].size(); j++ ){ // iteration of the points for each board
				fs << imagePoints[i][j].x << imagePoints[i][j].y;
			}
			fs << "]" << "}";
		}
		fs << "]";
	}
	
	void Calibration::load(string filename, bool absolute) {
		imagePoints.clear();
		FileStorage fs(ofToDataPath(filename, absolute), FileStorage::READ);
		cv::Size imageSize, sensorSize;
		Mat cameraMatrix;
		FileNode features;
		fs["cameraMatrix"] >> cameraMatrix;
		fs["imageSize_width"] >> imageSize.width;
		fs["imageSize_height"] >> imageSize.height;
		fs["sensorSize_width"] >> sensorSize.width;
		fs["sensorSize_height"] >> sensorSize.height;
		fs["distCoeffs"] >> myCameraModel.distCoeffs;
		fs["reprojectionError"] >> reprojectionError;
		vector<float> points;		
		features = fs["features"];
		int idx = 0;
        
		for(FileNodeIterator it = features.begin(); it != features.end(); it++) {
			idx++;
			(*it)["points"] >> points;
			vector<Point2f> featureset;
			for(int i = 0; i < points.size(); i+=2){
				featureset.push_back(Point2f(points[i], points[i+1]));
			}
			imagePoints.push_back(featureset); // technique 1
		}
        
        myCameraModel.distortedIntrinsics.setup(cameraMatrix, imageSize, sensorSize);
		updateUndistortion();
        
        _isReady=true;
	}
	
    
	void Calibration::setFillFrame(bool fillFrame) {
		this->fillFrame = fillFrame;
	}
    
    // This is ONLY used by a calibration object of type CAMERA:
	bool Calibration::generateCandidateImageObjectPoints() { 
		
       //NOTE: preProcessImage(img) is done BEFORE calling this method (the reason is that we want to check the processed images visually)
        //preProcessImage(img);// <<-- this is needed to select color, etc for *this* calibration object (to do!!)
        
		// find corners (using the current myPatternShape)
        candidateImagePoints.clear(); // ATTENTION!!! findBoard ADDS points to the vector (common source of errors...)
		bool found = findBoard(procImg, candidateImagePoints, true);
		
		if (found) {
            setCandidateObjectPoints(); // this just sets the object points from createObjectPointsFromPattern (usually for camera)
            ofLog(OF_LOG_ERROR, "Calibration::add() worked for Camera");
        }
		else
			ofLog(OF_LOG_ERROR, "Calibration::add() failed for Camera");
        
        patternDetected=found;
		return found;
	}

    // This is used by the PROJECTOR (we assume that the image represent the projected candidateImagePoints, being displayed by the projector) 
    bool Calibration::generateCandidateObjectPoints(Calibration& bestCalibratedCamera) {
    
        //NOTE: preProcessImage(img) is done BEFORE calling this method (the reason is that we want to check the processed images visually)
        //preProcessImage(img);// <<-- this is needed to select color, etc for *this* calibration object (to do!!)
        
		vector<Point2f> imagePointsSeenByCameraBuf;
        vector<Point2f> projectedImagePointsBuf;
        vector<Point3f> objectPointsBuf;
        
    	bool found = findBoard(procImg, imagePointsSeenByCameraBuf, true);
        //cout << "Number of projected points seen by camera: " << imagePointsSeenByCameraBuf.size() << endl;
        
		if (found) {
            candidateObjectPoints.clear(); 
            // update the object points by backprojection using the best available calibrated camera and the last object pose:
            // (this means that the bestCalibratedCamera need to have been calibrated on this particular image)
            bestCalibratedCamera.backProject(imagePointsSeenByCameraBuf, 
                                             candidateObjectPoints, // <<-- the output
                                             bestCalibratedCamera.candidateBoardRotation,     //boardRotations.back(), 
                                             bestCalibratedCamera.candidateBoardTranslation); //boardTranslations.back());
            
            ofLog(OF_LOG_ERROR, "Calibration::add() worked for projector");
        }
		else
			ofLog(OF_LOG_ERROR, "Calibration::add() failed for projector");
        
        patternDetected=found;
		return found;
	}
    
    bool Calibration::candidatePatternDetected() {
        return patternDetected;
    }
    
    void Calibration::setCandidateObjectPoints() {
        candidateObjectPoints=createObjectPointsFromPattern(myPatternShape);
    }
    
    void Calibration::setCandidateObjectPoints(vector<Point3f> newObjectSet) {
        candidateObjectPoints=newObjectSet;
    }
    
    void Calibration::setCandidateImagePoints() { 
        // NOTE: no argument means the fixed pattern shape. This function is overloaded however to accept arbitrary sets of image points (see below) 
        candidateImagePoints=createImagePointsFromPattern(myPatternShape);
    }
    
    void Calibration::setCandidateImagePoints(vector<Point2f> newImageSet) { 
        candidateImagePoints=newImageSet;
    }
    
    // Confirm the addition of the auxiliary candidateImagePoints image points:
    void Calibration::addCandidateImagePoints() { 
        imagePoints.push_back(candidateImagePoints);
    }
    void Calibration::addCandidateObjectPoints() { 
        objectPoints.push_back(candidateObjectPoints);
    }
    
    void Calibration::computeCandidateBoardPose() {
        // default is useExtrinsicGuess=false, flags=ITERATIVE 
        solvePnP(candidateObjectPoints, candidateImagePoints, myCameraModel.distortedIntrinsics.getCameraMatrix(), myCameraModel.distCoeffs, 
                 candidateBoardRotation, candidateBoardTranslation);// <<-- the output
        
        // For checking:
        /*
        cout << "BOARD POSE (camera frame) from PnP algorithm: " << endl;
        Mat rot3x3 = Mat::zeros(3, 3, CV_32F);
        Rodrigues(candidateBoardRotation, rot3x3);
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++)  cout << ofToString(rot3x3.at<double>(i,j),1) + "    "; 
            cout << ofToString(candidateBoardTranslation.at<double>(i),1) << endl;
        }
         */
        
    }
    
    void Calibration::addCandidateBoardPose() {
        boardRotations.push_back(candidateBoardRotation.clone());
        boardTranslations.push_back(candidateBoardTranslation.clone());
    }
    
    
    
    // Before anything else, do some image processing (in the future, this will mean to use the color of the pattern to do detection)
    void Calibration::addImageToProcess(Mat img) { // this will fill procImg
        
        // Process image as dictated by the pattern type (preprocess info on CalibrationShape, loaded fro yml file)
        // TO FINISH!!! (COLOR SEGMENTATION, color mosaicing, etc, indicate which processing in the yml file, etc).
        
        //imagerResolution = img.size(); // This would not be needed anymore. In fact, it was weird to do this everytime, because it somehow presupposes that we can be processing images of different sizes, which does not make sense. What can be done of course is to resize the images to FIT imagerResolution...
        
        
        switch(myPatternShape.typePattern) {
            case CHESSBOARD:
                // for the time being, nothing:
                procImg=img;
            break;
            case CIRCLES_GRID:
            case ASYMMETRIC_CIRCLES_GRID:
                // bitwise_not(img, img);
                if(img.type() != CV_8UC1) {
                    cvtColor(img, procImg, CV_RGB2GRAY);
                } 
                threshold(procImg, procImg, 210, 255, THRESH_BINARY_INV);
                erode(procImg, procImg, Mat());
                dilate(procImg, procImg, Mat());          
                dilate(procImg, procImg, Mat());
                erode(procImg, procImg, Mat());
                
                break;
        }
    }
    
    
	bool Calibration::findBoard(Mat img, vector<Point2f>& pointBuf, bool refine) {
		// no CV_CALIB_CB_FAST_CHECK, because it breaks on dark images (e.g., dark IR images from kinect)
		int chessFlags = CV_CALIB_CB_ADAPTIVE_THRESH;// | CV_CALIB_CB_NORMALIZE_IMAGE;
		bool found=false;
        
        switch(myPatternShape.typePattern) {
            case CHESSBOARD:
                
                cout << "detecting chessboard..." << endl;
                
                found = findChessboardCorners(img, myPatternShape.patternSize, pointBuf, chessFlags);
                // improve corner accuracy:
                if(found) {
                    
                    if(img.type() != CV_8UC1) {
                        cvtColor(img, procImg, CV_RGB2GRAY);
                    } else {
                        procImg = img;
                    }
                    
                    if(refine) {
                        // the subpixelSize dictates the smallest image space square size allowed
                        // in other words, if your smallest square is 11x11 pixels, then set this to 11x11
                        cornerSubPix(procImg, pointBuf, myPatternShape.subpixelSize,  cv::Size(-1,-1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1 ));
                    }
                } else
                    cout << "failed!" << endl;
                
                break;
            case CIRCLES_GRID:
                found=findCirclesGrid(img, myPatternShape.patternSize, pointBuf, CALIB_CB_SYMMETRIC_GRID);                
                break;
            case ASYMMETRIC_CIRCLES_GRID:
                found=findCirclesGrid(img, myPatternShape.patternSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);
                break;
                
        }
		return found;
	}
	
	bool Calibration::clean(float minReprojectionError) {
		int removed = 0;
		for(int i = size() - 1; i >= 0; i--) {
			if(getReprojectionError(i) > minReprojectionError) {
				objectPoints.erase(objectPoints.begin() + i);
				imagePoints.erase(imagePoints.begin() + i);
                boardRotations.erase(boardRotations.begin() + i);
                boardTranslations.erase(boardTranslations.begin() + i);
				removed++;
			}
		}
		if(size() > 0) {
			if(removed > 0) {
                // Note: I prefer not to call calibration again here. It should be up to the user to do that if necessary. 
				ofLog(OF_LOG_ERROR, "Calibration::clean() removed "+ofToString(removed)+ " boards.");
                _isReady=false; // meaning that we will need to recalibrate before moving on
                return true;
			} else {
                ofLog(OF_LOG_ERROR, "Calibration::clean() did not remove any board.");
				return true;
			}
		} else {
            _isReady=false;
			ofLog(OF_LOG_ERROR, "Calibration::clean() removed all the boards!");
			return false;
		}
	}
    
    // CLEAN method for projector/camera calibration should be a method of stereoCameraProjector class, because
    // the object and image points should be deleted simultaneously (objectPoints are NOT always the same). 
    bool Calibration::simultaneousClean(Calibration& calibrationObject, float minReprojectionError) { 
        // note: the reprojection error test is done only on *this* object
        int removed = 0;
		for(int i = size() - 1; i >= 0; i--) {
			if(getReprojectionError(i) > minReprojectionError) {
                // (a) erase things for *this* object:
				objectPoints.erase(objectPoints.begin() + i);
				imagePoints.erase(imagePoints.begin() + i);
                // ... inlcuding the board rotation and translation! (this is different form old code: we need these rotations and translations in 
                // order, to do the stereo calibration camera/projector: the actual image points for the camera are computed by reprojecting the object
                // points of the projector, using the proper rot/trans of each board).  
                boardRotations.erase(boardRotations.begin() + i);
                boardTranslations.erase(boardTranslations.begin() + i);
                // (b) erase things for the other calibration object:
                calibrationObject.objectPoints.erase(calibrationObject.objectPoints.begin() + i);
				calibrationObject.imagePoints.erase(calibrationObject.imagePoints.begin() + i);
                calibrationObject.boardRotations.erase(calibrationObject.boardRotations.begin() + i);
                calibrationObject.boardTranslations.erase(calibrationObject.boardTranslations.begin() + i);
				removed++;
			}
		}
		
        if(size() > 0) {
			if(removed > 0) {
                // Note: I prefer not to call calibration again here. It should be up to the user to do that if necessary. 
				ofLog(OF_LOG_ERROR, "Calibration::simultaneousClean() removed "+ofToString(removed)+ " boards.");
                _isReady=false;
                return true;
			} else {
                ofLog(OF_LOG_ERROR, "Calibration::simultaneousClean() did not remove any board.");
				return true;
			}
		} else {
			ofLog(OF_LOG_ERROR, "Calibration::simultaneousClean() removed all the boards!");
            _isReady=false;
			return false;
		}
    
    }
    
    void Calibration::deleteAllBoards() { // delete all the current acquired pair of image/object points (all the "boards")
        objectPoints.clear();
        imagePoints.clear();
        boardRotations.clear();
        boardTranslations.clear();
    }
    
    void Calibration::deleteLastBoard() { // delete all the current acquired pair of image/object points (all the "boards")
      //  cout << objectPoints.size() << " " << imagePoints.size() << " " << boardRotations.size() << " " << boardTranslations.size() << endl;
        
        objectPoints.pop_back();
        imagePoints.pop_back();
        boardRotations.pop_back();
        boardTranslations.pop_back();
    }
    
    
    // This computes the rotation/translation for array of corresponding image/object points (without calling the calibrationCamera, i.e., without recomputing the instrinsics) for the i-th board (objectPoints and imagePoints should be properly set).
    bool Calibration::computeBoardPose(int i) {
        Mat rot, trans;
        // default is useExtrinsicGuess=false, flags=ITERATIVE 
        solvePnP(objectPoints[i], imagePoints[i], myCameraModel.distortedIntrinsics.getCameraMatrix(), myCameraModel.distCoeffs, rot, trans);
        boardRotations.push_back(rot);
        boardTranslations.push_back(trans);
        
        // For checking:
        /*
        cout << "BOARD POSE (camera frame) from PnP algorithm: " << endl;
        Mat rot3x3 = Mat::zeros(3, 3, CV_32F);
        Rodrigues(boardRotations.back(), rot3x3);
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++)  cout << ofToString(rot3x3.at<double>(i,j),1) + "    "; 
            cout << ofToString(boardTranslations.back().at<double>(i),1) << endl;
        }
         */
        
        
        // NOTE: openCV2.4 has a solvePnP function that returns a bool... but not openCV 2.3
        // hack:
        return(boardRotations.size()==size());
    }
    

    
    // The calibration routine for "this" calibration object (not stereo calibration), using ALL the boards not "cleaned":
	bool Calibration::calibrate() {
		Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
		Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
        
        /*
        cout << "Number of boards used for calibration (object/image): " << objectPoints.size() << ", " << imagePoints.size() << endl;
        for (int i=0; i<objectPoints.size() ; i++) {
            cout << "Num of points in board "<< i << " (ob/im) = " << objectPoints[i].size() << ", " << imagePoints[i].size() << endl;
        }
         */
        
		int calibFlags = 0;
		float rms = calibrateCamera(objectPoints, imagePoints, myCameraModel.getImagerResolution(), 
                                    cameraMatrix, distCoeffs, // <<-- these are outputs
                                    boardRotations, boardTranslations, // <<-- these are outputs too
                                    calibFlags);
        
        // For displaying purposes, we may want to update the candidate rot/trans as the latest in the list:
        candidateBoardRotation=boardRotations.back();
        candidateBoardTranslation=boardTranslations.back();
        
		ofLog(OF_LOG_VERBOSE, "calibrate() reports RMS error of " + ofToString(rms));
        
		_isReady = checkRange(cameraMatrix) && checkRange(distCoeffs);
        
		if(!_isReady) {
			ofLog(OF_LOG_ERROR, "calibrate() failed");
		}
		
        // update intrinsics:
		myCameraModel.distortedIntrinsics.setup(cameraMatrix, myCameraModel.getImagerResolution());
        myCameraModel.distCoeffs=distCoeffs;
		updateReprojectionErrors();
		updateUndistortion();
		
		return _isReady;
	}
	
	bool Calibration::isReady(){
		return _isReady;
	}
	
	bool Calibration::calibrateFromDirectory(string directory) {
		ofDirectory dirList;
		ofImage cur;
		dirList.listDir(directory);
		for(int i = 0; i < dirList.size(); i++) {
			cur.loadImage(dirList.getPath(i));
            addImageToProcess(toCv(cur));
			if(!generateCandidateImageObjectPoints()) {
				ofLog(OF_LOG_ERROR, "Calibration::add() failed on " + dirList.getPath(i));
			}
		}
		return calibrate(); 
	}
	void Calibration::undistort(Mat img) {
		img.copyTo(undistortBuffer);
		undistort(undistortBuffer, img);
	}
	void Calibration::undistort(Mat src, Mat dst) {
		remap(src, dst, undistortMapX, undistortMapY, INTER_LINEAR);
	}
	
	ofVec2f Calibration::undistort(ofVec2f &src)
	{
		ofVec2f dst;
		
		Mat matSrc = Mat(1, 1, CV_32FC2, &src.x);
		Mat matDst = Mat(1, 1, CV_32FC2, &dst.x);
		
		undistortPoints(matSrc, matDst, myCameraModel.distortedIntrinsics.getCameraMatrix(), myCameraModel.distCoeffs);
		
		return dst;
		
	}
	
	void Calibration::undistort(vector<ofVec2f> &src, vector<ofVec2f> &dst)
	{
		int nPoints = src.size();
		
		if (dst.size() != nPoints)
			dst.resize(src.size());
		
		Mat matSrc = Mat(nPoints, 1, CV_32FC2, &src[0].x);
		Mat matDst = Mat(nPoints, 1, CV_32FC2, &dst[0].x);
		
		undistortPoints(matSrc, matDst, myCameraModel.distortedIntrinsics.getCameraMatrix(), myCameraModel.distCoeffs);
		
	}
	
    // Compute extrinsics of the stereo calibration (from both calibration objects of type "camera"):
	bool Calibration::getTransformation(Calibration& dst, Mat& rotation, Mat& translation) {
		//if(imagePoints.size() == 0 || dst.imagePoints.size() == 0) {
		if(!_isReady) {
			ofLog(OF_LOG_ERROR, "getTransformation() requires both Calibration objects to have just been calibrated");
			return false;
		}
		if(imagePoints.size() != dst.imagePoints.size() || myPatternShape.patternSize != dst.myPatternShape.patternSize) {
			ofLog(OF_LOG_ERROR, "getTransformation() requires both Calibration objects to be trained simultaneously on the same board");
			return false;
		}
		Mat fundamentalMatrix, essentialMatrix;
		Mat cameraMatrix = myCameraModel.distortedIntrinsics.getCameraMatrix();
		Mat dstCameraMatrix = dst.getDistortedIntrinsics().getCameraMatrix();
		// uses CALIB_FIX_INTRINSIC by default
		stereoCalibrate(objectPoints, // <<-- fixed, and COMMON to both calibration camera objects
                        imagePoints, dst.imagePoints,
                        cameraMatrix, myCameraModel.distCoeffs,
                        dstCameraMatrix, dst.myCameraModel.distCoeffs,
                        myCameraModel.distortedIntrinsics.getImageSize(), 
                        rotation, translation,  // << ----  This is the result of the getTransformation method
                        essentialMatrix, fundamentalMatrix); // << ---- this is also the result (but is not made avaiblable by the method)
		return true;
	}		
    
    // Stereo Calibration Camera/Projector: actually a method of the subclass CalibrationProjector,
    // i.e it must be called on the PROJECTOR calibration object. Note that this function is equivalent to updateReprojectedImagePoints(), but uses
    // the intrinsics of the CAMERA to reproject, not of "this" object (the projector).
    // OR BETTER: should belong to a class STEREO CALIBRATION
    // OUTPUT: rotation and translation such that Xp=R.Xc+t (cameraToProjectorExtrinsics)
	bool Calibration::stereoCalibrationCameraProjector(Calibration& bestCamera, Mat& rotation, Mat& translation) {
		
        /*
        cout << "Number of boards for stereo calibration:" << endl;
        cout << "Camera: " << endl;
        cout << "Image boards:  " << bestCamera.objectPoints.size() << ", Object boards: " << bestCamera.imagePoints.size() << endl;
        cout << "Rotation mats: " << bestCamera.boardRotations.size() << ", trans mats: " << bestCamera.boardTranslations.size() << endl;
        cout << "Projector: "<< endl;
        cout << "Image boards:  " << objectPoints.size() << ", Object boards: " << imagePoints.size() << endl;
        */
        
        if (!_isReady||!bestCamera.isReady()) {
		
            ofLog(OF_LOG_ERROR, "stereoCalibrationCameraProjector() requires both Calibration objects to have just been calibrated");
			return false;
		
        } else {
        
        // NOTE: the projector calibration object contains the image and object points, but the camera object does NOT contain the image points corresponding to the object points projected by the projector (of course, we could have saved them, but this is not necessary since we can reproject them here  - both strategies may give different results, and perhaps the latter is better since the back-projection has been done with a camera that had different instrinsics... but I suppose things will converge for good). 
        
        // Compute image points for the camera in an auxiliary array, from ALL the projector object points:
        vector<vector<Point2f> > auxImagePointsCamera;
        for (int i=0; i<objectPoints.size() ; i++ ) { 
            // reminder: size() is just equal to imagePoints.size(), that should also be equal to objectPoints.size()
            vector<Point2f> auxImagePoints;
            projectPoints(Mat(objectPoints[i]), 
                          bestCamera.boardRotations[i], bestCamera.boardTranslations[i], 
                          bestCamera.myCameraModel.distortedIntrinsics.getCameraMatrix(), 
                          bestCamera.myCameraModel.distCoeffs, 
                          auxImagePoints);
            
            auxImagePointsCamera.push_back(auxImagePoints);
        }
        
        // Now we can proceed with the stereo calibration:
        
    	Mat fundamentalMatrix, essentialMatrix;
		Mat projectorMatrix = myCameraModel.distortedIntrinsics.getCameraMatrix();
		Mat CameraMatrix = bestCamera.getDistortedIntrinsics().getCameraMatrix();
		// NOTE: uses CALIB_FIX_INTRINSIC by default (which is good because we assume independent calibration converged to something good)
		// given the order of the parameters (image points and intrinsic matrices, this function gives the pose of the CAMERA with 
        // respect to the PROJECTOR coordinate system. 
        Mat rotation3x3; // auxiliary: stereoCalibrate gives a rotation matrix, not a rotation vector...
        stereoCalibrate(objectPoints, // common "3d" points (on the board)
                        
                        auxImagePointsCamera, // image points for the camera
                        imagePoints, // image points for the projector
                      
                         CameraMatrix, bestCamera.myCameraModel.distCoeffs,
                        projectorMatrix, myCameraModel.distCoeffs, 
                        
                        myCameraModel.distortedIntrinsics.getImageSize(), // <--- only used for initialization
                        rotation3x3, translation,  // << ----  OUPUT: position of CAMERA in PROJECTOR coordinate frame
                        essentialMatrix, fundamentalMatrix); // << ---- this is also the result (but is not made avaiblable by the method)
		Rodrigues(rotation3x3, rotation);// we need a rotation vector (1x3)
        //cout << " **** SIZE ROTATION VECTOR (before): " << rotation.size().width << "x" << rotation.size().height << endl;
        //if (rotation.size().width!=1) rotation=rotation.t();
        //cout << " **** SIZE ROTATION VECTOR (after): " << rotation.size().width << "x" << rotation.size().height << endl;
        return true;
        }
	}		
    
    // The following function is typically called with object points in "board" coordinates, rotation and translation matrix of the board
    // with respect to the camera, and the extrinsics of the projector (camera-to-projector rotation/translation).
    // The procedure is as follows:
    // (1) Take the latest object points for the camera (in "world" or "board" coordinates)
    // (2) Use latest rotation/translation in calibrationCamera to get the points in camera coordinates. 
    // (3) Use "extrinsics" to get the 3d points in Projector coordinates.
    // (4) Use the instrinics of the projector to actually project the points, and get imagepoints to be projected
    vector<Point2f> Calibration::createImagePointsFrom3dPoints(vector<Point3f>& objpts,  const Mat& R1,const Mat& t1,const Mat& Re,const Mat& te) {
        vector<Point2f> pointsToProject; // <- the final points to be displayed by the projector
        Mat finalRotVec, finalTransVec;
        
        // Compose the transformations:
        // (attn: composeRT applies the transformation in the parameters from left to right!)
        composeRT(R1, t1, Re, te, finalRotVec, finalTransVec);
        
        // Then, project the points (in general the instrinsics would be those of the PROJECTOR - this function is called by the projector object):
        projectPoints(Mat(objpts), 
                      finalRotVec, finalTransVec, 
                      myCameraModel.distortedIntrinsics.getCameraMatrix(), myCameraModel.distCoeffs, 
                      pointsToProject); //<<-- the output
        
        return pointsToProject;
    }
    
	float Calibration::getReprojectionError() const {
		return reprojectionError;
	}
	float Calibration::getReprojectionError(int i) const {
		return perViewErrors[i];
	}
	const Intrinsics& Calibration::getDistortedIntrinsics() const {
		return myCameraModel.distortedIntrinsics;
	}
	const Intrinsics& Calibration::getUndistortedIntrinsics() const {
		return myCameraModel.undistortedIntrinsics;
	}
	Mat Calibration::getDistCoeffs() const {
		return myCameraModel.distCoeffs;
	}
	int Calibration::size() const { // actually, number of boards used for calibration (change name?)
        // IN THE FUTURE, this should be a method of the "board list", and we should always check for consistency (not only image points, but also 
        // object points for intstance, and perhaps also board trans and rot). 
		return imagePoints.size();
	}
	
    // Drawing functions:
    
    void Calibration::drawPreprocessedImage(int x, int y, int dx, int dy) { // essentially, this is for checking and interactively changing color/threshold for board detection
        drawMat(procImg,x,y, dx, dy);
    }
    
    
    // (a) Draw image points (on the camera image, or on a full screen to project)
    // NOTE: important: in case of projection, we need to set the screen resolution to that of the projector in question. 
    //       This information is in the Intrinsics class (as "imageSize"). 
	void Calibration::customDraw() {
		for(int i = 0; i < size(); i++) {
			draw(i);
		}
	}
	void Calibration::draw(int i) const {
		ofPushStyle();
		ofNoFill();
		ofSetColor(ofColor::red); //<< --- color and fill will be given by the pattern color in case of the projector. 
		for(int j = 0; j < imagePoints[i].size(); j++) {
			ofCircle(toOf(imagePoints[i][j]), 5);
		}
		ofPopStyle();
	}
   
    bool Calibration::drawCandidateAxis(int px, int py, int sx, int sy) {
        
        if (objectPoints.size()>0) {
            vector<Point3f> axis;
            axis.push_back(Point3f(0.0f,0.0f,0.0f));
            axis.push_back(Point3f(10.0f,0,0.0f));
            axis.push_back(Point3f(0.0f,10.0f,0.0f));
            vector<Point2f> axisim;
            projectPoints(Mat(axis), 
                          candidateBoardRotation, candidateBoardTranslation, 
                          myCameraModel.distortedIntrinsics.getCameraMatrix(), myCameraModel.distCoeffs, 
                          axisim); //<<-- the output
            
            
            ofPushStyle();
            ofSetLineWidth(3);
            ofPushMatrix(); 
            
            ofTranslate(px, py);
            float ratx=(float)sx/myCameraModel.getImagerResolution().width, raty=(float)sy/myCameraModel.getImagerResolution().height;
            ofScale(ratx, raty, 1);
            
            ofSetColor(255, 0, 255);
            ofLine(axisim[0].x, axisim[0].y, axisim[1].x, axisim[1].y);
            ofSetColor(0, 255, 255);
            ofLine(axisim[0].x, axisim[0].y, axisim[2].x, axisim[2].y);
            
            
            ofPopMatrix();
            ofPopStyle();
        }
        return true;
        
    }
    
   
    
    // This is useful to check the quality of the reprojection (using the current intrinsics)
    bool Calibration::drawCandidateReprojection(int px, int py, int sx, int sy, ofColor col) {
        if (candidateObjectPoints.size()>0) { // otherwise do nothing (indicate with boolean)?
            // Compute reprojection of latest object points of THIS object:
            vector<Point2f> latestReprojectedImagePoints;
            projectPoints(Mat(candidateObjectPoints), 
                          candidateBoardRotation, candidateBoardTranslation, 
                          myCameraModel.distortedIntrinsics.getCameraMatrix(), myCameraModel.distCoeffs, 
                          latestReprojectedImagePoints);
            
            ofPushStyle();
            ofNoFill();
            ofSetColor(col); 
            ofSetLineWidth(1);
            
            ofPushMatrix(); 
            ofTranslate(px, py);
            float ratx=1.0*sx/myCameraModel.getImagerResolution().width, raty=1.0*sy/myCameraModel.getImagerResolution().height;
            ofScale(ratx, raty, 1); 
            for(int j = 0; j < latestReprojectedImagePoints.size(); j++) {
                ofCircle(toOf(latestReprojectedImagePoints[j]), 5);
            }
            ofPopMatrix();
            
            ofPopStyle();
            return true;
        } else 
            return false;
    }
    
    // This is useful to check the detected board corners:
    bool Calibration::drawCandidateImagePoints(int px, int py, int sx, int sy, ofColor col) {  
        if (candidateImagePoints.size()>0) {
            // TO DO: switch case for the patter type (circles, chessboard...)
            ofPushStyle();
            ofNoFill();
            ofSetColor(col); //<< --- color and fill will be given by the pattern color in case of the projector. 
            ofSetLineWidth(1);
            
            ofPushMatrix(); 
            ofTranslate(px, py);
            float ratx=1.0*sx/myCameraModel.getImagerResolution().width, raty=1.0*sy/myCameraModel.getImagerResolution().height;
            ofScale(ratx, raty, 1);
            for(int j = 0; j < candidateImagePoints.size(); j++) {
                ofCircle(toOf(candidateImagePoints[j]), 6);
            }
            ofPopMatrix();
            
            ofPopStyle();
            return true;
        }else
            return false;
    }
    
    // this function draws the projector pattern from candidateImagePoints; IF this pattern is detected, then it will be added to the 
    // list of imagePoints for the projector. NOTE: this is just the same than drawCandidateImagePoints. RENAME ALL THIS MESS. 
    bool Calibration::drawCandidateProjectorPattern(int px, int py, int sx, int sy, ofColor col, float sizecircle) { 
       if (candidateImagePoints.size()>0) {
            // TO DO: switch case for the patter type (circles, chessboard...)
            ofPushStyle();
            
            ofSetLineWidth(3);
            
            ofPushMatrix(); 
            ofTranslate(px, py);
            float ratx=sx/myCameraModel.getImagerResolution().width, raty=sy/myCameraModel.getImagerResolution().height;
            ofScale(ratx, raty, 1);
            for(int j = 0; j < candidateImagePoints.size(); j++) {
                ofFill();
                ofSetColor(col); //<< --- color and fill will be given by the pattern color in case of the projector. 
                ofCircle(toOf(candidateImagePoints[j]), sizecircle); //myPatternShape.squareSize/6);
                ofNoFill();
                ofSetColor(0); 
                ofCircle(toOf(candidateImagePoints[j]), sizecircle+3); //myPatternShape.squareSize/6+2);
            }
            ofPopMatrix();
            
            ofPopStyle();
            return true;
        }else
            return false;
    }
    
    // This will be used to display points with the projector, in particular after using the method createImagePointsFrom3dPoints. 
    bool  Calibration::drawArbitraryImagePoints(int px, int py, int sx, int sy, vector<Point2f>& imPoints, ofColor col, float sizecircle) {
        if (imPoints.size()>0) {
            // TO DO: switch case for the patter type (circles, chessboard...)
            ofPushStyle();
            ofFill();
            ofSetColor(col); //<< --- color and fill will be given by the pattern color in case of the projector. 
            ofSetLineWidth(2);
            
            ofPushMatrix(); 
            ofTranslate(px, py);
            float ratx=sx/myCameraModel.getImagerResolution().width, raty=sy/myCameraModel.getImagerResolution().height;
            ofScale(ratx, raty, 1);
            for(int j = 0; j < imPoints.size(); j++) {
                ofCircle(toOf(imPoints[j]), sizecircle);
            }
            ofPopMatrix();
            
            ofPopStyle();
            return true;
        }else
            return false;
    }
    
    
    
    
    // draw 3d for camera (rem: need to set the proper openGL projection matrix)
	void Calibration::draw3d() const {
		for(int i = 0; i < size(); i++) {
			draw3d(i);
		}
	}
	void Calibration::draw3d(int i) const {
		ofPushStyle();
		ofPushMatrix();
		ofNoFill();
		
		applyMatrix(makeMatrix(boardRotations[i], boardTranslations[i]));
		
		ofSetColor(ofColor::fromHsb(255 * i / size(), 255, 255));
		
		ofDrawBitmapString(ofToString(i), 0, 0);
		
		for(int j = 0; j < objectPoints[i].size(); j++) {
			ofPushMatrix();
			ofTranslate(toOf(objectPoints[i][j]));
			ofCircle(0, 0, .5);
			ofPopMatrix();
		}
        
		ofMesh mesh;
		mesh.setMode(OF_PRIMITIVE_LINE_STRIP);
		for(int j = 0; j < objectPoints[i].size(); j++) {
			ofVec3f cur = toOf(objectPoints[i][j]);
			mesh.addVertex(cur);
		}
		mesh.draw();
		
		ofPopMatrix();
		ofPopStyle();
	}
    
    
	void Calibration::updateReprojectionErrors() {
		vector<Point2f> imagePoints2;
		int totalPoints = 0;
		double totalErr = 0;
		
		perViewErrors.clear();
		perViewErrors.resize(objectPoints.size());
		
		for(int i = 0; i < objectPoints.size(); i++) {
			projectPoints(Mat(objectPoints[i]), boardRotations[i], boardTranslations[i], myCameraModel.distortedIntrinsics.getCameraMatrix(), myCameraModel.distCoeffs, imagePoints2);
			double err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);
			int n = objectPoints[i].size();
			perViewErrors[i] = sqrt(err * err / n);
			totalErr += err * err;
			totalPoints += n;
			cout << "view " + ofToString(i) + " has error of " + ofToString(perViewErrors[i]) << endl;
		}
		
		reprojectionError = sqrt(totalErr / totalPoints);
		
		ofLog(OF_LOG_VERBOSE, "all views have error of " + ofToString(reprojectionError));
	}
    
	void Calibration::updateUndistortion() {
		Mat undistortedCameraMatrix = getOptimalNewCameraMatrix(myCameraModel.distortedIntrinsics.getCameraMatrix(), myCameraModel.distCoeffs, myCameraModel.distortedIntrinsics.getImageSize(), fillFrame ? 0 : 1);
		initUndistortRectifyMap(myCameraModel.distortedIntrinsics.getCameraMatrix(), myCameraModel.distCoeffs, Mat(), undistortedCameraMatrix, myCameraModel.distortedIntrinsics.getImageSize(), CV_16SC2, undistortMapX, undistortMapY);
		myCameraModel.undistortedIntrinsics.setup(undistortedCameraMatrix, myCameraModel.distortedIntrinsics.getImageSize());
	}
	
    
    vector<Point3f> Calibration::createObjectPointsFromPattern(const CalibrationShape& patternShape) { 
        vector<Point3f> objectCorners;
		switch(patternShape.typePattern) {
			case CHESSBOARD: // same than circle grid
			case CIRCLES_GRID:
				for(int i = 0; i < patternShape.patternSize.height; i++)
					for(int j = 0; j < patternShape.patternSize.width; j++)
						objectCorners.push_back(Point3f(float(j * patternShape.squareSize), float(i * patternShape.squareSize), 0));
				break;
			case ASYMMETRIC_CIRCLES_GRID:
				for(int i = 0; i < patternShape.patternSize.height; i++)
					for(int j = 0; j < patternShape.patternSize.width; j++)
						objectCorners.push_back(Point3f(float(((2 * j) + (i % 2)) * patternShape.squareSize), float(i * patternShape.squareSize), 0));
				break;
		}
		return objectCorners;
    }
    
    vector<Point3f> Calibration::createObjectPointsDynamic(const Point3f& pos, const Point3f& axisX, const Point3f& axisY, const CalibrationShape& patternShape) { 
        // The difference with createObjectPointsFromPattern is that the pattern may be slanted and resized, and the origin is also a parameter.  
        vector<Point3f> objectCorners;
		switch(patternShape.typePattern) {
			case CHESSBOARD: // same than circle grid
			case CIRCLES_GRID:
				for(int i = 0; i < patternShape.patternSize.height; i++)
					for(int j = 0; j < patternShape.patternSize.width; j++)
                        objectCorners.push_back(pos+ axisX*j + axisY*i);
				break;
			case ASYMMETRIC_CIRCLES_GRID:
				for(int i = 0; i < patternShape.patternSize.height; i++)
					for(int j = 0; j < patternShape.patternSize.width; j++)
						objectCorners.push_back(pos+axisX*float((2 * j) + (i % 2))+axisY*i);
				break;
		}
		return objectCorners;
    }
    
    vector<Point2f> Calibration::createImagePointsFromPattern(const CalibrationShape& patternShape) { 
        vector<Point2f> projectedPattern;
		switch(patternShape.typePattern) {
			case CHESSBOARD: // same than circle grid
			case CIRCLES_GRID:
				for(int i = 0; i < patternShape.patternSize.height; i++)
					for(int j = 0; j < patternShape.patternSize.width; j++)
						projectedPattern.push_back(Point2f(patternShape.posX+float(j * patternShape.squareSize), patternShape.posY+float(i * patternShape.squareSize)));
				break;
			case ASYMMETRIC_CIRCLES_GRID:
				for(int i = 0; i < patternShape.patternSize.height; i++)
					for(int j = 0; j < patternShape.patternSize.width; j++)
						projectedPattern.push_back(Point2f(patternShape.posX+float(((2 * j) + (i % 2)) * patternShape.squareSize), patternShape.posY+float(i * patternShape.squareSize)));
				break;
		}
		return projectedPattern;
    }

    
    
    /*
     vector<Point3f> Calibration::createObjectPointsForProjector(Mat img, CalibrationShape patternShape, Calibration bestCalibrationCamera) {
     // First, use the camera to find the image points to be backprojected:
     vector<Point2f> auxPointBuf;
     bool found=bestCalibrationCamera.findBoard(img, auxPointBuf, true);
     // Then, back project using the bestCamera:
     vector<Point3f> corners;
     if (found) {
     bestCalibrationCamera.backProject(auxPointBuf, corners);//const Mat& boardRot64, const Mat& boardTrans64 
     return corners;
     }
     }
     */
    
    
    bool Calibration::backProject(const vector<Point2f>& imgPt, 
                                  vector<Point3f>& worldPt,
                                  const Mat& boardRot64, const Mat& boardTrans64 ) {
        
        if( imgPt.size() == 0 ) {
            return false;
        } 
        else 
        {
            Mat imgPt_h = Mat::zeros(3, imgPt.size(), CV_32F);
            for( int h=0; h<imgPt.size(); ++h ) {
                imgPt_h.at<float>(0,h) = imgPt[h].x;
                imgPt_h.at<float>(1,h) = imgPt[h].y;
                imgPt_h.at<float>(2,h) = 1.0f;
            }
            Mat Kinv64 = myCameraModel.undistortedIntrinsics.getCameraMatrix().inv();
            //Mat Kinv64 = myCameraModel.distortedIntrinsics.getCameraMatrix().inv();
            Mat Kinv,boardRot,boardTrans;
            Kinv64.convertTo(Kinv, CV_32F);
            boardRot64.convertTo(boardRot, CV_32F);
            boardTrans64.convertTo(boardTrans, CV_32F);
            
            
            // Transform all image points to world points in camera reference frame
            // and then into the plane reference frame
            Mat worldImgPt = Mat::zeros( 3, imgPt.size(), CV_32F );
            Mat rot3x3;
            Rodrigues(boardRot, rot3x3);
            
            //Mat TRT = rot3x3*boardTrans;
            //TRT = boardTrans - TRT;
            
            Mat transPlaneToCam = rot3x3.inv()*boardTrans;
            
            for( int i=0; i<imgPt.size(); ++i ) {
                Mat col = imgPt_h.col(i);
                Mat worldPtcam = Kinv*col;
                Mat worldPtPlane = rot3x3.inv()*(worldPtcam);
                
                float scale = transPlaneToCam.at<float>(2)/worldPtPlane.at<float>(2);
                Mat worldPtPlaneReproject = scale*worldPtPlane-transPlaneToCam;
                
                //std::cout << "worldpt: " << worldPtPlaneReproject << std::endl;
                Point3f pt;
                pt.x = worldPtPlaneReproject.at<float>(0);
                pt.y = worldPtPlaneReproject.at<float>(1);
                pt.z = 0;
                worldPt.push_back(pt);
            }
            
            
            // vector<Point2f> reprojPt;
            // projectPoints(Mat(worldPt), boardRot, boardTrans,
            //               myCameraModel.distortedIntrinsics.getCameraMatrix(), myCameraModel.distCoeffs, reprojPt);
            
        }
        
        return true;
    }
}	

