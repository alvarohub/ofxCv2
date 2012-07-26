#include "ofxCv/ContourFinder.h"
#include "ofxCv/Wrappers.h"

namespace ofxCv {
	
	using namespace cv;
    
    // *** added for sorting by area:
    bool sort_contours_by_area(vector<cv::Point> contourA, vector<cv::Point> contourB) {
        // use opencv to calc size, then sort based on size
        // type of parameters: vector<cv::Point> contourA, countourB;
        double areaa = contourArea(Mat(contourA)); //fabs(cvContourArea(a, CV_WHOLE_SEQ));
        double areab = contourArea(Mat(contourB)); //fabs(cvContourArea(b, CV_WHOLE_SEQ));
        return (areaa > areab);
    }
    
	
	ContourFinder::ContourFinder()
	:autoThreshold(true)
	,thresholdValue(5)// differential color threshold in case of trackcolor or floodfill
	,invert(false)
	,simplify(true)
	,useTargetColor(false)
    ,useSeedFloodFill(false) // for some reason, it DOES NOT GET THIS DEFAULT VALUE!!!
    ,autoUpdateSeed(false) {
		resetMinArea();
		resetMaxArea();
        setSeedPosition(ofVec2f(0,0)); // takes values as ofVec2f (or ofPoint)
	}
	
	void ContourFinder::findContours(Mat img) {
        
  
        // *** FLOOD FILL BASED on seed point (note: this may be redundant with the color threshold... but well)
        // This will MODIFY the "thresh" image (I could not fill on a separate mask, there seems to be a problem with the floodFill parameters...)
        if (useSeedFloodFill) { // note: this mode overrides useTargetColor mode (bad, should find a better way to charaterize the contour detection modes).
            Scalar offset(thresholdValue, thresholdValue, thresholdValue);
            
            // Define the fill color in the original image (not the mask):
            // Scalar colorfill = toCv(ofColor(255, 255, 255));
            Scalar colorfill =CV_RGB(255,255,255); //note: it would be complicated to use ofColor, it is not the same type. 
            
            // (the function cvFloodFill has not been wrapped in OF)
            
            // Now, do a flood fill with seed point, but filling the MASK and/or the color source image:
            // int newcolormask=255;
            int connectivity=8; //4 or 8
            int flags= connectivity;// + CV_FLOODFILL_FIXED_RANGE;//+ CV_FLOODFILL_MASK_ONLY ;
            cv::Rect rect;
            
            //cv::Mat label_image;
            //colorImage.convertTo(colorImage, CV_32FC1); // weird it doesn't support CV_32S!
            
            Mat colorImage=img.clone(); // because attention: floodFill will modify the image...
            //cvtColor(colorImage, colorImage, CV_RGB2HSV);
            int blurQuantity=3; // note: if the number is EVEN, it crashes??
            GaussianBlur(img, colorImage, cv::Size( blurQuantity, blurQuantity ), 0, 0);
            
            // cv::floodFill(colorImage, grayMask , seedPoint, colorfill, &rect, cv::Scalar(intensityDiff), cv::Scalar(intensityDiff),flags);
            cv::floodFill(colorImage , seedFloodPoint, colorfill, &rect, offset, offset,flags);
            
            //cv::imshow("flood color", colorImage);
            
            // NOTE: the resulting image "thresh" has to be a "binary image"
            if(colorImage.channels() == 1) {
				thresh = colorImage.clone();
			} else if(colorImage.channels() == 3) {
				cvtColor(colorImage, thresh, CV_RGB2GRAY);
			} else if(colorImage.channels() == 4) {
				cvtColor(colorImage, thresh, CV_RGBA2GRAY);
			}
            cv::threshold(thresh, thresh, 254, 255, THRESH_BINARY);
            cv::dilate(thresh,thresh,Mat(), cv::Point(-1,-1), 3);
            cv::erode(thresh,thresh,Mat(), cv::Point(-1,-1), 1);
            
            // for tests:
             cv::imshow("flood", thresh);
        }        
		else {
            // threshold the image using a tracked color or just binary grayscale
            if(useTargetColor) {
			Scalar offset(thresholdValue, thresholdValue, thresholdValue);
			Scalar base = toCv(targetColor);
			if(trackingColorMode == TRACK_COLOR_RGB) {
                
				inRange(img, base - offset, base + offset, thresh); 
                // this gives an CV_8U type image, with values between 0 or 255.
			} else {
				if(TRACK_COLOR_H) { //????
					offset[1] = 255;
					offset[2] = 255;
				}
				if(TRACK_COLOR_HS) {
					offset[2] = 255;//????
				}
				cvtColor(img, hsvBuffer, CV_RGB2HSV);
				base = toCv(convertColor(targetColor, CV_RGB2HSV));
				Scalar lowerb = base - offset;
				Scalar upperb = base + offset;
				inRange(hsvBuffer, lowerb, upperb, thresh);
			}
            }
        else {
			if(img.channels() == 1) {
				thresh = img.clone();
			} else if(img.channels() == 3) {
				cvtColor(img, thresh, CV_RGB2GRAY);
			} else if(img.channels() == 4) {
				cvtColor(img, thresh, CV_RGBA2GRAY);
			}
        }
       }
            
       // cv::imshow("after thresh", thresh);
        
        //We need a BINARY image to do findCountours or floofill?
        if(autoThreshold) {
            threshold(thresh, thresholdValue, invert);
        }
        
               
		// run the contour finder
		vector<vector<cv::Point> > allContours;
		int simplifyMode = simplify ? CV_CHAIN_APPROX_SIMPLE : CV_CHAIN_APPROX_NONE;
		cv::findContours(thresh, allContours, CV_RETR_EXTERNAL, simplifyMode);
        
		// filter the contours (if needed)
		bool needMinFilter = (minArea > 0);
		bool needMaxFilter = maxAreaNorm ? (maxArea < 1) : (maxArea < numeric_limits<float>::infinity());
		if(needMinFilter || needMaxFilter) {
			contours.clear();
			double imgArea = img.rows * img.cols;
			double imgMinArea = minAreaNorm ? (minArea * imgArea) : minArea;
			double imgMaxArea = maxAreaNorm ? (maxArea * imgArea) : maxArea;
			for(int i = 0; i < allContours.size(); i++) {
				double curArea = contourArea(Mat(allContours[i]));
				if((!needMinFilter || curArea >= imgMinArea) &&
					 (!needMaxFilter || curArea <= imgMaxArea)) {
					contours.push_back(allContours[i]);
				}
			}
		} else {
			contours = allContours;
		}
        
        // ** ARRANGE the remaining contours by SIZE (or other property!)
        if( contours.size() > 1 ) {
            std::sort( contours.begin(), contours.end(), sort_contours_by_area );
        }

        // Update SEED POINT for floodFill (only on the largest contour) - idea: in the future, have this for all contours, 
        // and apply a TRACKER to the centroids - even a kalman filter. 
        if ((autoUpdateSeed)&&(contours.size() > 0)) {
            seedFloodPoint=getCentroid(0); 
        }
        
		// generate polylines from the contours
		polylines.clear();
		for(int i = 0; i < size(); i++) {
			polylines.push_back(toOf(contours[i]));
		}
		
		// generate bounding boxes from the contours
		boundingBoxes.clear();
		for(int i = 0; i < size(); i++) {
			boundingBoxes.push_back(boundingRect(Mat(contours[i])));
		}
		
		// track bounding boxes
		tracker.track(boundingBoxes);
	}
	
    //--------------------------------------------------------------------------------
    
	vector<vector<cv::Point> >& ContourFinder::getContours() {
		return contours;
	}
	
	vector<ofPolyline>& ContourFinder::getPolylines() {
		return polylines;
	}
	
	unsigned int ContourFinder::size() const {
		return contours.size();
	}
	
	vector<cv::Point>& ContourFinder::getContour(unsigned int i) {
		return contours[i];
	}
	
	ofPolyline& ContourFinder::getPolyline(unsigned int i) {
		return polylines[i];
	}
	
	cv::Rect ContourFinder::getBoundingRect(unsigned int i) const {
		return boundingBoxes[i];
	}
	
	cv::Point2f ContourFinder::getCenter(unsigned int i) const {
		cv::Rect box = getBoundingRect(i);
		return cv::Point2f(box.x + box.width / 2, box.y + box.height / 2);
	}
	
	cv::Point2f ContourFinder::getCentroid(unsigned int i) const {
		Moments m = moments(Mat(contours[i]));
		return cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
	}
	
	cv::Point2f ContourFinder::getAverage(unsigned int i) const {
		Scalar average = mean(Mat(contours[i]));
		return cv::Point2f(average[0], average[1]);
	}
	
	cv::Vec2f ContourFinder::getBalance(unsigned int i) const {
		return cv::Vec2f(getCentroid(i) - getCenter(i));
	}
	
	double ContourFinder::getContourArea(unsigned int i) const {
		return contourArea(Mat(contours[i]));
	}
	
	double ContourFinder::getArcLength(unsigned int i) const {
		return arcLength(Mat(contours[i]), true);
	}
	
	vector<cv::Point> ContourFinder::getConvexHull(unsigned int i) const {
		vector<cv::Point> hull;
		convexHull(Mat(contours[i]), hull);
		return hull;
	}
	
	cv::RotatedRect ContourFinder::getMinAreaRect(unsigned int i) const {
		return minAreaRect(Mat(contours[i]));
	}
	
	cv::Point2f ContourFinder::getMinEnclosingCircle(unsigned int i, float& radius) const {
		cv::Point2f center;
		minEnclosingCircle(Mat(contours[i]), center, radius);
		return center;
	}
	
	cv::RotatedRect ContourFinder::getFitEllipse(unsigned int i) const {
		return fitEllipse(Mat(contours[i]));
	}
	
	vector<cv::Point> ContourFinder::getFitQuad(unsigned int i) const {
		vector<cv::Point> convexHull = getConvexHull(i);		
		vector<cv::Point> quad = convexHull;
		
		static const unsigned int targetPoints = 4; // this can be a parameter for a generalized polygon search...
		static const unsigned int maxIterations = 16;
		static const double infinity = numeric_limits<double>::infinity();
		double minEpsilon = 0;
		double maxEpsilon = infinity;
		double curEpsilon = 16; // good initial guess
		
		// unbounded binary search to simplify the convex hull until it's exactly targetPoints points
		if(convexHull.size() > targetPoints) { // otherwise don't do anything... ?
			for(int i = 0; i < maxIterations; i++) {
				approxPolyDP(Mat(convexHull), quad, curEpsilon, true);
				if(quad.size() == targetPoints) {
					break;
				}
				if(quad.size() > targetPoints) {
					minEpsilon = curEpsilon;
					if(maxEpsilon == infinity) {
						curEpsilon = curEpsilon *  2;
					} else {
						curEpsilon = (maxEpsilon + minEpsilon) / 2;
					}
				}
				if(quad.size() < targetPoints) {
					maxEpsilon = curEpsilon;
					curEpsilon = (maxEpsilon + minEpsilon) / 2;
				}
			}
		}
		
		return quad;
	}
	
	cv::Vec2f ContourFinder::getVelocity(unsigned int i) const {
		unsigned int label = tracker.getLabelFromIndex(i);
		if(tracker.existsPrevious(label)) {
			cv::Rect& previous = tracker.getPrevious(label);
			cv::Rect& current = tracker.getCurrent(label);
			cv::Vec2f previousPosition(previous.x + previous.width / 2, previous.y + previous.height / 2);
			cv::Vec2f currentPosition(current.x + current.width / 2, current.y + current.height / 2);
			return currentPosition - previousPosition;
		} else {
			return cv::Vec2f(0, 0);
		}
	}
	
	unsigned int ContourFinder::getLabel(unsigned int i) {
		return tracker.getCurrentLabels()[i];
	}
	
	RectTracker& ContourFinder::getTracker() {
		return tracker;
	}
	
	void ContourFinder::setAutoThreshold(bool autoThreshold) {
		this->autoThreshold = autoThreshold;
	}
	
	void ContourFinder::setThreshold(float thresholdValue) {
		this->thresholdValue = thresholdValue;
	}
	
	void ContourFinder::setInvert(bool invert) {
		this->invert = invert;
	}
	
    // added: NOTE: problem with default initialization? also, needed beause of new floodFill mode:
    void ContourFinder::setUseTargetColor(bool useTargetColor) {
        this->useTargetColor = useTargetColor;
    }
    
	void ContourFinder::setTargetColor(ofColor targetColor, TrackingColorMode trackingColorMode) {
		//useTargetColor = true;
		this->targetColor = targetColor;
		this->trackingColorMode = trackingColorMode;
	}
	
    void ContourFinder::setFloodFillMode(bool useSeedFloodFill) {
        this->useSeedFloodFill=useSeedFloodFill;
    }
    void ContourFinder::setAutoUpdateSeed(bool autoUpdateSeed){
        this->autoUpdateSeed=autoUpdateSeed;
    }
    
    void ContourFinder::setSeedPosition(ofVec2f seedFloodPoint) {// use ofVec2f so we can pass ofPoint, as well as CENTROID (float values).  
        //useSeedFloodFill=true;
        this->seedFloodPoint=toCv(seedFloodPoint); // toCv means here conversion to Point (alias for Point2i)
    }
    
	void ContourFinder::setSimplify(bool simplify) {
		this->simplify = simplify;
	}
	
	void ContourFinder::draw() {
		ofPushStyle();
		ofNoFill();
		for(int i = 0; i < polylines.size(); i++) {
			polylines[i].draw();
			ofRect(toOf(getBoundingRect(i))); // this should be optional
		}
		ofPopStyle();
	}
	
    // ** added
    void ContourFinder::draw(int contournumber) {
        if (contournumber<polylines.size()) {
		ofPushStyle();
		ofNoFill();
        polylines[contournumber].draw();
        //ofRect(toOf(getBoundingRect(contournumber)));
		ofPopStyle();
        }
	}
    
	void ContourFinder::resetMinArea() {
		setMinArea(0);
	}
	
	void ContourFinder::resetMaxArea() {
		setMaxArea(numeric_limits<float>::infinity());
	}
	
	void ContourFinder::setMinArea(float minArea) {
		this->minArea = minArea;
		minAreaNorm = false;
	}
	
	void ContourFinder::setMaxArea(float maxArea) {
		this->maxArea = maxArea;
		maxAreaNorm = false;
	}
	
	void ContourFinder::setMinAreaRadius(float minAreaRadius) {
		minArea = PI * minAreaRadius * minAreaRadius;
		minAreaNorm = false;
	}
	
	void ContourFinder::setMaxAreaRadius(float maxAreaRadius) {
		maxArea = PI * maxAreaRadius * maxAreaRadius;
		maxAreaNorm = false;
	}
	
	void ContourFinder::setMinAreaNorm(float minAreaNorm) {
		minArea = minAreaNorm;
		this->minAreaNorm = true;
	}
	
	void ContourFinder::setMaxAreaNorm(float maxAreaNorm) {
		maxArea = maxAreaNorm;
		this->maxAreaNorm = true;
	}
	
}
