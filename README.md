camera-projector-calib-of
=========================

Camera Projector Calibration tool addon for OF (in progress).

This addon is an extension of OfxCV by KyleMc Donald, adding several functionalities. In particular, the Calibration class has been heavily modified so that both cameras AND projectors can be modelled and calibrated simulatneously. (Other minor modifications include a useSeedFloodFill and a autoUpdateSeed boolean to track regions of uniform color and detecting quadrangles).  

An complete example showing camera/projector calibration can be found here:
 https://github.com/alvarohub/Example_CameraProjectorCalibration 

Check this video to see how to proceed:
  http://www.youtube.com/watch?v=pCq7u2TvlxU&list=UUtYM3-7ldtX7kf_sSoHt1Pw&index=1&feature=plcp

Things it would be nice to implement soon:
 - extend the possible patterns to detect (chessboard, circle grid, but also non coplanar points, etc). In fact, perhaps this could be implemented in the class "Tracking" as specific methods used by the calibration class. In general, it would be good to extend the tracking class, with advanced features, for instance, starting with a seed (position and color), then doing a flood-fill, and then use a self-windowing algorithm to "follow" the object to track, perhaps updating the color target as the object moves (using the average color of the object). This could be implememted as a sort of interactive ROI for the contour findinder class. 
 (then of course, we can also constrain tracking using 3d geometry of the object if known - this is the more serious method used in most SLAM methods). 
 - important too: these patterns should be segmented by color (this would simplify camera/projector calibration by projecting, say, a red circle grid of points on a blue chessboard, and be capable of detecting both patterns even when these superimpose each other)


