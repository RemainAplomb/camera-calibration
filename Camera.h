#pragma once


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;

class Camera
{
public:
	Mat cameraMatrix, distCoeffs;
	Mat rvec, tvec;
	Matx33f invrot;
	Matx31f invt;
	Vec4d intrinsic;
	void init(Mat& r, Mat&t);
	void update(Matx31f & pos , Rect & , Mat & ,int );

};

