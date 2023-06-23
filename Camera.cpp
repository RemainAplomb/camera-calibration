


#include "Camera.h"
#include <stdio.h>
#include <iostream>
using namespace std;

extern bool bshow;

void Camera::init(Mat& r, Mat&t) {

	intrinsic.val[0] = cameraMatrix.at<double>(0, 0);
	intrinsic.val[1] = cameraMatrix.at<double>(1, 1);
	intrinsic.val[2] = cameraMatrix.at<double>(0, 2);
	intrinsic.val[3] = cameraMatrix.at<double>(1, 2);

	rvec = r;
	tvec = t;

	Mat rot;
	cv::Rodrigues(r, rot);

	Mat inv = rot.inv();
	invrot = inv;


	invt.val[0] = t.at<double >(0);
	invt.val[1] = t.at<double>(1);
	invt.val[2] = t.at<double>(2);

	Matx31f rotated_point = invrot*invt;
	invt = -rotated_point;



}



void Camera::update(Matx31f & pos ,Rect & r ,Mat & src ,int box)
{

	vector<Point3f>  newpoints;
	
	
	//int box = 56;

	newpoints.push_back(Point3f(pos.val[0], pos.val[1] , pos.val[2] ));

	newpoints.push_back(Point3f(pos.val[0] + box, pos.val[1] + box, 0.0f));// pos.val[2] + box / 2));
	newpoints.push_back(Point3f(pos.val[0]-box, pos.val[1]+box,  0.0f));// pos.val[2] + box/2));
	newpoints.push_back(Point3f(pos.val[0]+box, pos.val[1]-box ,  0.0f));// pos.val[2] + box/2));
	newpoints.push_back(Point3f(pos.val[0]-box, pos.val[1]-box,  0.0f));// pos.val[2] + box/2));

	newpoints.push_back(Point3f(pos.val[0] + box, pos.val[1] + box, pos.val[2] - box/2));
	newpoints.push_back(Point3f(pos.val[0] - box, pos.val[1] + box, pos.val[2] - box/2));
	newpoints.push_back(Point3f(pos.val[0] + box, pos.val[1] - box, pos.val[2] - box/2));
	newpoints.push_back(Point3f(pos.val[0] - box, pos.val[1] - box, pos.val[2] - box/2));
	newpoints.push_back(Point3f(0,0,0));


	/*for each (Point3f var in newpoints)
	{
		std::cout << var << endl;
	}*/


	Mat distorted_points2d;

	projectPoints(Mat(newpoints), rvec, tvec, cameraMatrix,
		distCoeffs, distorted_points2d);

	float xmin = 10000;
	float xmax = -10000;
	float ymin = 10000;
	float ymax = -10000;

	Point2f rotated_point = distorted_points2d.at<Point2f>(0);
	rectangle(src, Rect(rotated_point.x - 2, rotated_point.y - 2, 5, 5), Scalar(255, 100, 0));

	Mat qview = imread("C:\\james\\images\\b0.bmp");
	rotated_point = distorted_points2d.at<Point2f>(9);
	rectangle(src, Rect(rotated_point.x - 2, rotated_point.y - 2, 5, 5), Scalar(0, 0, 255));

	for (int i = 1; i < 9; i++)
	{
		 rotated_point = distorted_points2d.at<Point2f>(i);

		if ( i < 5)
		rectangle(src, Rect(rotated_point.x - 2, rotated_point.y - 2, 5, 5), Scalar(0, 255, 0));
		else
			rectangle(src, Rect(rotated_point.x - 2, rotated_point.y - 2, 5, 5), Scalar(0, 0, 255));


		if (i < 5)
			rectangle(src, Rect(rotated_point.x - 2, rotated_point.y - 2, 5, 5), Scalar(0, 255, 0));
		else
			rectangle(src, Rect(rotated_point.x - 2, rotated_point.y - 2, 5, 5), Scalar(0, 0, 255));


					if (rotated_point.x < xmin)
						xmin = rotated_point.x;
					if (rotated_point.x > xmax)
						xmax = rotated_point.x;


					if (rotated_point.y < ymin)
						ymin = rotated_point.y;
					if (rotated_point.y > ymax)
						ymax = rotated_point.y;



	}

	if (xmin < 0)
		xmin = 0;
	if (ymin < 0)
		ymin = 0;


	r.x = xmin;
	r.y = ymin;
	r.width = xmax - xmin;
	r.height = ymax - ymin;

	if ((xmin + r.width) >= 640)
		r.width = 640 - xmin - 1;
	if ((r.height + ymin) >= 480)
	r.height = 480 - ymin;



	if (bshow)
	{

		imshow("q", src);
		cvWaitKey(1);
	}
}
