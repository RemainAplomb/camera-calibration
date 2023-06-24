#define _USE_MATH_DEFINES

#include <windows.h>
#include "calibrate.h"
#include <iostream>
#include <stdlib.h>  
#include <string>  
#include <math.h>

//#include "atlbase.h"  
//#include "atlstr.h"  
//#include "comutil.h"  


extern bool bshow;
using namespace std;

#ifndef _CRT_SECURE_NO_WARNINGS
# define _CRT_SECURE_NO_WARNINGS
#endif

//#include <boost/timer/timer.hpp>
//#include <boost/format.hpp>


/**
 * Represents a 4-dimensional axis-angle rotation.
 */
struct AxisAngle4d
{
	double angle; /**< The angle of rotation. */
	double x;     /**< The x-component of the rotation axis. */
	double y;     /**< The y-component of the rotation axis. */
	double z;     /**< The z-component of the rotation axis. */
};



//http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
/**
 * Constructs a rotation matrix from an axis-angle representation.
 *
 * @param a1 The axis-angle representation to convert.
 * @param r21 The resulting rotation matrix.
 */
void matrixFromAxisAngle(AxisAngle4d a1, Mat& r21)
{
	double c = cos(a1.angle);
	double s = sin(a1.angle);
	double t = 1.0 - c;

	// if axis is not already normalized, uncomment the following code to normalize it
	// double magnitude = sqrt(a1.x * a1.x + a1.y * a1.y + a1.z * a1.z);
	// if (magnitude == 0)
	//     throw error;
	// a1.x /= magnitude;
	// a1.y /= magnitude;
	// a1.z /= magnitude;

	double m00 = c + a1.x * a1.x * t;
	double m11 = c + a1.y * a1.y * t;
	double m22 = c + a1.z * a1.z * t;

	double tmp1 = a1.x * a1.y * t;
	double tmp2 = a1.z * s;
	double m10 = tmp1 + tmp2;
	double m01 = tmp1 - tmp2;

	tmp1 = a1.x * a1.z * t;
	tmp2 = a1.y * s;
	double m20 = tmp1 - tmp2;
	double m02 = tmp1 + tmp2;

	tmp1 = a1.y * a1.z * t;
	tmp2 = a1.x * s;
	double m21 = tmp1 + tmp2;
	double m12 = tmp1 - tmp2;

	// Assign the computed values to the rotation matrix
	r21.at<double>(0, 0) = m00;
	r21.at<double>(1, 0) = m01;
	r21.at<double>(2, 0) = m02;

	r21.at<double>(0, 1) = m10;
	r21.at<double>(1, 1) = m11;
	r21.at<double>(2, 1) = m12;

	r21.at<double>(0, 2) = m20;
	r21.at<double>(1, 2) = m21;
	r21.at<double>(2, 2) = m22;
}



void readPara(Camera & cam)
{
	// Retrieve the path of the current executable
	char  buffer[1024];
	GetModuleFileName(NULL,(LPWSTR) buffer, 1024);

	// Get the current working directory
	TCHAR NPath[MAX_PATH];
	GetCurrentDirectory(MAX_PATH, NPath);

	// Extract the directory portion of the executable path
	string::size_type pos = string(buffer).find_last_of("\\/");
	string tmp= string(buffer).substr(0, pos);
	
	// char* pf= NPath;
	// Convert the path string to a regular character string
	char nstring [1024];
	size_t convertedChars;

	size_t origsize = wcslen(NPath) + 1;
	size_t newsize = origsize*2;

	wcstombs_s(&convertedChars, nstring, newsize, NPath, _TRUNCATE);

	// TODO: Add comment explaining the purpose of the following code
	// MessageBox(0,(LPWSTR) tmp.c_str(), L"MessageBox caption", MB_OK);
	// FILE * outf = fopen("out.txt", "w+");
	// fprintf(outf, "%s\n", buffer );
	// fclose(outf);
	// MessageBox(0, L"OK", L"MessageBox caption", MB_OK);

	// TODO: Add comment explaining the purpose of the following code
	strstr(buffer, "/");

	// Specify the camera parameter file path
	const cv::String outputFileName = cv::String("C:\\james\\images_circle\\out_camera_data.xml");

	//Camera cam;
	// Initialize variables for camera parameters and image size
	vector<Mat> rvecs, tvecs;
	Size imageSize;

	Mat qview;
	Mat rot;
	
	// Load an image if the bshow flag is true
	if ( bshow)
	qview = imread("C:\\james\\images_circle\\test2.jpg");
	//MessageBox(0,(LPCWSTR)(outputFileName.c_str()), L"MessageBox caption", MB_OK);

	// TODO: Add comment explaining the purpose of the following code
	// MessageBox(0, (LPCWSTR)(outputFileName.c_str()), L"MessageBox caption", MB_OK);
	if (!readCameraParams(outputFileName, imageSize, cam.cameraMatrix, cam.distCoeffs, rvecs, tvecs))
	{
		return;
	}
	// MessageBox(0, L"read", L"MessageBox caption", MB_OK);

	// Print the camera matrix
	cout << cam.cameraMatrix << endl;


	// TODO: Add comment explaining the purpose of the following code
	// MessageBox(0, L"out", L"MessageBox caption", MB_OK);

	// TODO: Add comment explaining the purpose of the following code
	// cam.init(rvecs[1], tvecs[1]);

	// TODO: Add comment explaining the purpose of the following code
	// MessageBox(0, L"init", L"MessageBox caption", MB_OK);


	// rot = rvecs[0];
	//cv::Mat RM(3, 3, cv::DataType<double>::type);

	// Compute rotation matrices from rotation vectors
	Mat rot0;
	cv::Rodrigues(rvecs[0], rot0);

	Mat rot1;
	cv::Rodrigues(rvecs[1], rot1);

	// Print translation vectors and rotation matrices
	cout << tvecs[0] << endl;
	cout << rot0 << endl;

	cout << tvecs[1] << endl;
	cout << rot1 << endl;

	// Compute relative rotation matrix between the two rotations
	Mat r21 = rot1*rot0.inv() ;

	
	// Create an axis-angle representation
	AxisAngle4d a1;
	a1.angle = 22.5f*M_PI/180;
	a1.x = 0;
	a1.y = 0;
	a1.z = 1;

	// Convert axis-angle representation to a rotation matrix
	matrixFromAxisAngle(a1, r21);

	// Compute translation vector between the two translations
	//Mat t21 = -r21*tvecs[0].t() + tvecs[1].t(); �z�פW����.���p�G���I�b���,�i���]t ���@��,�G�i��t[0]
	Mat t21 = -r21*tvecs[0].t() + tvecs[0].t();

	// Create a point in the world coordinate system
	Mat pw=Mat::zeros(3,1,CV_64F);
	pw.at<double>(0) = 25.0f;
	
	// Transform the point to camera 1 and camera 2 coordinate systems
	Mat p1 = rot0*pw+ tvecs[0].t();
	Mat p2 = rot1*pw + tvecs[1].t();
	Mat p21 = r21 * p1 + t21;


	// Print the transformed points
	cout << p1 << endl;
	cout << p2 << endl;
	cout << p21 << endl;


	// Extract individual elements from the relative rotation matrix
	double R00 = r21.at<double>(0, 0);
	double R01 = r21.at<double>(1, 0);
	double R02 = r21.at<double>(2, 0);

	double R10 = r21.at<double>(0, 1);
	double R11 = r21.at<double>(1, 1);
	double R12 = r21.at<double>(2, 1);


	double R20 = r21.at<double>(0,2);
	double R21 = r21.at<double>(1, 2);
	double R22 = r21.at<double>(2, 2);

	// Compute the angle of rotation
	double angle = acos((R00 + R11 + R22 - 1) / 2)*180/M_PI;

	// Compute the axis of rotation
	double  ax, ay, az;
	ax = (R21 - R12) / sqrt((R21 - R12)*(R21 - R12) + (R02 - R20)*(R02 - R20) + (R10 - R01)*(R10 - R01));
	ay = (R02 - R20) / sqrt((R21 - R12)*(R21 - R12) + (R02 - R20)*(R02 - R20) + (R10 - R01)*(R10 - R01));
	az = (R10 - R01) / sqrt((R21 - R12)*(R21 - R12) + (R02 - R20)*(R02 - R20) + (R10 - R01)*(R10 - R01));

	// Check the rotation angle and axis
	cout << "angle: " << angle << endl;
	cout << "axis: (" << ax << ", " << ay << ", " << az << ")" << endl;


	// Get the number of translation vectors
	int nt =tvecs.size();

	// Compute the distance between each pair of translation vectors
	for (int k = 0; k < nt; k++)
	{
		for (int i = 0; i < nt; i++)
		{
			if (i == k)
				continue;

			Mat diff = tvecs[i] - tvecs[k];
			double err = sqrt(diff.dot(diff));
			cout << err << endl;
			//cout << diff << endl;
		}
		cout << "____________________" << endl;
	}

	
	// Create a vector of 3D points
	vector<Point3f>  newpoints;
	newpoints.push_back(Point3f(0, 0, 0));
	newpoints.push_back(Point3f(25, 0, 0));
	newpoints.push_back(Point3f(0, 100, 0));
	newpoints.push_back(Point3f(50, 0, 0));


//	newpoints.push_back(Point3f(-44, 42, 0));
//	newpoints.push_back(Point3f(-44, 42, -112));




	/*Mat newMatrix;
	newMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 0.5, imageSize, 0);
	
	cout << cameraMatrix << endl;
	cout << newMatrix << endl;
*/

	// Project the 3D points onto the image plane
	Mat distorted_points2d;
	//projectPoints(Mat(newpoints), cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), cameraMatrix, distCoeffs, distorted_points2d);
	projectPoints(Mat(newpoints), rvecs[1], tvecs[1], cam.cameraMatrix,
		cam.distCoeffs, distorted_points2d);

	Scalar cc3 = cv::Scalar(0, 0, 255, 255); //BGRA
	int cBlockSize = 5;

	// Draw rectangles around the projected points on the image
	for (int i = 0; i < distorted_points2d.rows; i++)
	{
		Point2f pt = distorted_points2d.at<Point2f>(i);

		rectangle(qview, cv::Rect(pt.x - cBlockSize, pt.y - cBlockSize, cBlockSize * 2 + 1, cBlockSize * 2 + 1), cc3);
	}


//	MessageBox(0, L"ok", L"MessageBox caption", MB_OK);

	if (bshow)
	{
		// Show the image and save
		imshow("", qview);
		cvWaitKey(0);
		imwrite("debug2.png", qview);
	}
}



static void help()
{
    cout <<  "This is a camera calibration sample." << endl
         <<  "Usage: calibration configurationFile"  << endl
         <<  "Near the sample file you'll find the configuration file, which has detailed help of "
             "how to edit it.  It may be any OpenCV supported file format XML/YAML." << endl;
}


static void read(const FileNode& node, Settings& x, const Settings& default_value = Settings())
{
    if(node.empty())
		// Assign the default value to x if the node is empty
        x = default_value;
    else
		// Call the read() function of the Settings class to read the node
        x.read(node);
}

// Enumeration to define three different states: DETECTION, CAPTURING, CALIBRATED
enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };



int runcalib()
{

	 Settings s;
	  const string inputSettingsFile =  "C:\\james\\images\\default.xml";


	  
    FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
    if (!fs.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
        return -1;
    }
    fs["Settings"] >> s;  // Read the "Settings" node from the file into the Settings object
    fs.release();  // Close the settings file

	// Check whether the inputs are good
    if (!s.goodInput)
    {
        cout << "Invalid input detected. Application stopping. " << endl;
        return -1;
    }

    vector<vector<Point2f>> imagePoints;  // Vector to store the detected 2D image points for calibration
	Mat cameraMatrix, distCoeffs;  // Matrices to store the camera matrix and distortion coefficients
	vector<Mat> rvecs, tvecs;  // Vectors to store the rotation and translation vectors
	Size imageSize;  // Size of the input images
	int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;  // Mode of operation: CAPTURING or DETECTION
	clock_t prevTimestamp = 0;  // Previous timestamp for capturing images
	const Scalar RED(0, 0, 255), GREEN(0, 255, 0);  // Scalar values for the colors red and green
	const char ESC_KEY = 27;  // Key code for the escape key

    for(int i = 0;;++i)
    {
      Mat view;
      bool blinkOutput = false;

      view = s.nextImage();

      //-----  If no more image, or got enough, then stop calibration and show result -------------
      if( mode == CAPTURING && imagePoints.size() >= (unsigned)s.nrFrames )
      {
          if( runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints,rvecs, tvecs))
              mode = CALIBRATED;
          else
              mode = DETECTION;
      }
      if(view.empty())          // If no more images then run calibration, save and stop loop.
      {
            if( imagePoints.size() > 0 && mode != CALIBRATED )
                runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints,rvecs, tvecs);
            break;
      }


        imageSize = view.size();  // Format input image.
        if( s.flipVertical )    flip( view, view, 0 );

        vector<Point2f> pointBuf;

        bool found;
        switch( s.calibrationPattern ) // Find feature points on the input format
        {
        case Settings::CHESSBOARD:
		{
			found = findChessboardCorners(view, s.boardSize, pointBuf,
				CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

			if (!found)
			{

				char fn[1024];
				int find=0;
				strcpy(fn, s.imageList[s.atImageList].c_str());
				sscanf(fn, "%*[^f]f%d.bmp", &find);

				int indx = 1;
				while (!found && indx < 30)
				{
					sprintf(fn, "images//f%d.bmp", find + indx);
					indx++;
					view = imread(fn);
					if (view.dims == 0)
						break;
					found = findChessboardCorners(view, s.boardSize, pointBuf,
						CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);


				}
			}
		}
            break;
        case Settings::CIRCLES_GRID:
            found = findCirclesGrid( view, s.boardSize, pointBuf );
            break;
        case Settings::ASYMMETRIC_CIRCLES_GRID:
            found = findCirclesGrid( view, s.boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID );
            break;
        default:
            found = false;
            break;
        }

        if ( found)                // If done with success,
        {
              // improve the found corners' coordinate accuracy for chessboard
                if( s.calibrationPattern == Settings::CHESSBOARD)
                {
                    Mat viewGray;
                    cvtColor(view, viewGray, COLOR_BGR2GRAY);
                    cornerSubPix( viewGray, pointBuf, Size(11,11),
                        Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
                }

                if( mode == CAPTURING &&  // For camera only take new samples after delay time
                    (!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC) )
                {
                    imagePoints.push_back(pointBuf);	// Store the detected points
      				prevTimestamp = clock();
                    blinkOutput = s.inputCapture.isOpened();
                }

                // Draw the corners.
                drawChessboardCorners( view, s.boardSize, Mat(pointBuf), found );
        }

        //----------------------------- Output Text ------------------------------------------------
        string msg = (mode == CAPTURING) ? "100/100" :
                      mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

        if( mode == CAPTURING )
        {
            if(s.showUndistorsed)
                msg = format( "%d/%d Undist", (int)imagePoints.size(), s.nrFrames );
            else
                msg = format( "%d/%d", (int)imagePoints.size(), s.nrFrames );
        }

        putText( view, msg, textOrigin, 1, 1, mode == CALIBRATED ?  GREEN : RED);

        if( blinkOutput )
            bitwise_not(view, view);

        //------------------------- Video capture  output  undistorted ------------------------------
        if( mode == CALIBRATED && s.showUndistorsed )
        {
            Mat temp = view.clone();
            undistort(temp, view, cameraMatrix, distCoeffs);
        }

        //------------------------------ Show image and check for input commands -------------------
        imshow("Image View", view);
//		imwrite( "save.bmp",view);

		char key;
		if ( i ==0 ) 
			key =waitKey();
		else 
			key = (char)waitKey(s.inputCapture.isOpened() ? 50 : s.delay);

        if( key  == ESC_KEY )
            break;

        if( key == 'u' && mode == CALIBRATED )
           s.showUndistorsed = !s.showUndistorsed;

        if( s.inputCapture.isOpened() && key == 'g' )
        {
            mode = CAPTURING;
            imagePoints.clear();
        }
    }

	 return 1;
}


int mymain(int argc, char* argv[])
{
	// Display help message
    help();

	// Create an instance of the Settings structure
    Settings s;
	
	// Define the input settings file path based on command-line arguments or use a default path
    const string inputSettingsFile = argc > 1 ? argv[1] : "C:\\james\\images\\default.xml";


	// Read the settings
    FileStorage fs(inputSettingsFile, FileStorage::READ);
	// Check if the settings file can be opened
    if (!fs.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
        return -1; // Return error code (-1)
    }

    fs["Settings"] >> s; // Read the settings from the file into the 's' object
    fs.release(); // Close the settings file

	// Check if the input settings are valid
    if (!s.goodInput)
    {
        cout << "Invalid input detected. Application stopping. " << endl;
        return -1; // Return error code (-1)
    }

    vector<vector<Point2f>> imagePoints; // Create a 2D vector to store image points
	Mat cameraMatrix, distCoeffs; // Create matrices to store camera matrix and distortion coefficients
	vector<Mat> rvecs, tvecs; // Create vectors to store rotation and translation vectors
	Size imageSize; // Define a variable to store image size
	int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION; // Determine the mode based on the input type in the settings
	clock_t prevTimestamp = 0; // Store the previous timestamp
	const Scalar RED(0, 0, 255), GREEN(0, 255, 0); // Define color constants
	const char ESC_KEY = 27; // Define the ASCII code for the ESC key

	// Call the function readCameraParams() to read camera parameters from the output file specified in the settings
	if (!readCameraParams(s.outputFileName, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs))
	{
		// Handle the case when camera parameters cannot be read
		// This part is currently empty in the code
	}

	// Uncommented lines related to boost timer
	// Currently, I don't know what this is for
	// boost::timer::cpu_timer timer;
	// timer.start();



    // -----------------------Show the undistorted image for the image list ------------------------
    if( s.inputType == Settings::IMAGE_LIST && s.showUndistorsed )
    {
        Mat view, rview, map1, map2;
		Mat  nview, nmap1, nmap2;
		Mat newMatrix;
		newMatrix= getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
           newMatrix, imageSize, CV_16SC2, map1, map2);

	   float ss = s.squareSize ;

		vector<Point3f>  newpoints;		
		vector<Point2f> imagePoints2;

		float bw=169;//195;//169;//ss*5;	// Set the value for bw
		float bh=230;//265;//228;//ss*7;	// Set the value for bh
		float bt=31.5;//19;//32;//ss;		// Set the value for bt

		float newx = ss*7; 
		float newy =ss*5;

		newpoints.push_back(Point3f(0,newy -bw,0));   //oldbase (0,0,0)  viewed from the newbase (0,newy,0) 
		newpoints.push_back(Point3f(bh,newy-bw,0));   // x 
		newpoints.push_back(Point3f(0,newy,0)); //y
		newpoints.push_back(Point3f(bh,newy ,0)); //x,y
		newpoints.push_back(Point3f(bh,newy,-bt));  //x,y
		newpoints.push_back(Point3f(0,newy-bw,-bt));  //oldbase
		newpoints.push_back(Point3f(bh,newy-bw,-bt));  //x
		newpoints.push_back(Point3f(0,newy,-bt));
		newpoints.push_back(Point3f(0,0,0));

        //for(int i = 0; i < (int)s.imageList.size(); i++ )
		int i=0;
        {
            view = imread("C:\\james\\images\\captured.png", 1);
			//view = imread(s.imageList[i], 1);
			// if(view.empty())
			//    continue;
			remap(view, rview, map1, map2, INTER_LINEAR);

		

			/*distCoeffs.at<double>(0) = 0;
			distCoeffs.at<double>(1) = 0;
			distCoeffs.at<double>(2) = 0;
			distCoeffs.at<double>(3) = 0;
			distCoeffs.at<double>(4) = 0;*/

			// Projecting 3D points onto the 2D image plane
			projectPoints(Mat(newpoints), rvecs[i], tvecs[i], newMatrix, distCoeffs, imagePoints2);
			
			// Creating matrices for mapping operations
			nmap1.create(view.size(), CV_32FC1);
			nmap2.create(view.size(), CV_32FC1);

			// Scaling factor for mapping
			float scale = 4.0f;

			// Vector for storing 3D points
			vector<Point3f> map3d;
			// Vector for storing 2D points
			vector<Point2f> mapp;


			// Calculating elapsed time and frame rate using timer (commented out)
			//boost::timer::cpu_times elapsed = timer.elapsed(); // nano seconds...
			//float t = (elapsed.wall / 1000.0f / 1000.0f / 1000.0f) / (float)100;
			//float fps = 1.0f / t ;
			//if (t > 0.001) {
			//    std::cout << "t=" << t << ", fps=" << fps << std::endl;
			//}

			// Restarting the timer (commented out)
			//timer.start();


			// Iterating over rows and columns of nmap1 to populate map3d
			for (int j = 0; j < nmap1.rows; j++) {
				for (int i = 0; i < nmap1.cols; i++) {
					if ((i < scale * bw + 50) && (j < scale * bh + 50)) {
						map3d.push_back(Point3d(bh - j / scale, newy - bw + i / scale, -bt));
					}
				}
			}
			
			//elapsed = timer.elapsed(); // nano seconds...
			// t = (elapsed.wall / 1000.0f / 1000.0f / 1000.0f) / (float)100;
			// fps = 1.0f / t ;
			//if (t > 0.001) {
			//	std::cout << "t=" << t << ", fps=" << fps << std::endl;
			//}
			//	timer.start();

			// Projecting 3D map points onto the 2D image plane using rvecs[0] and tvecs[0]
			projectPoints(Mat(map3d), rvecs[0], tvecs[0], newMatrix, distCoeffs, mapp);

			// Calculating elapsed time and frame rate using timer (commented out)
			//elapsed = timer.elapsed(); // nano seconds...
			//t = (elapsed.wall / 1000.0f / 1000.0f / 1000.0f) / (float)100;
			//fps = 1.0f / t ;
			//if (t > 0.001) {
			//    std::cout << "t=" << t << ", fps=" << fps << std::endl;
			//}

			// Restarting the timer (commented out)
			//timer.start();

			int k = 0;
			for (int j = 0; j < nmap1.rows; j++) {
				for (int i = 0; i < nmap1.cols; i++) {
					if ((i < scale * bw + 50) && (j < scale * bh + 50)) {
						nmap1.at<float>(j, i) = mapp[k].x;
						nmap2.at<float>(j, i) = mapp[k].y;
						k++;
					} else {
						nmap1.at<float>(j, i) = 0;
						nmap2.at<float>(j, i) = 0;
					}
				}
			}

			// Calculating elapsed time and frame rate using timer (commented out)
			//elapsed = timer.elapsed(); // nano seconds...
			//t = (elapsed.wall / 1000.0f / 1000.0f / 1000.0f) / (float)100;
			//fps = 1.0f / t ;
			//if (t > 0.001) {
			//    std::cout << "t=" << t << ", fps=" << fps << std::endl;
			//}
			// Restarting the timer (commented out)
			//timer.start();

			// updatmap( nmap1,nmap2, imagePoints2 ,bw,bh,scale);

			// Performing remapping of rview using nmap1 and nmap2
			remap(rview, nview, nmap1, nmap2, INTER_LINEAR);

			// Drawing lines and ellipse on nview and rview
			line(nview, Point2f(0, 0 * scale), Point2f(bw * scale, 0 * scale), Scalar(0, 0, 255), 2, 8);
			line(nview, Point2f(0, 60 * scale), Point2f(bw * scale, 60 * scale), Scalar(0, 255, 0), 2, 8);
			line(nview, Point2f(0, 77 * scale), Point2f(bw * scale, 77 * scale), Scalar(0, 0, 255), 2, 8);
			line(nview, Point2f(0, bh * scale), Point2f(bw * scale, bh * scale), Scalar(0, 0, 255), 2, 8);
			ellipse(rview, imagePoints2[8], Size(5, 5), 0, 0, 360, Scalar(255, 0, 0), 2, 8);
			line(rview, imagePoints2[0], imagePoints2[1], Scalar(0, 255, 0), 2, 8);
			line(rview, imagePoints2[0], imagePoints2[2], Scalar(0, 0, 255), 2, 8);
			line(rview, imagePoints2[1], imagePoints2[3], Scalar(0, 0, 255), 2, 8);
			line(rview, imagePoints2[2], imagePoints2[3], Scalar(0, 0, 255), 2, 8);
			line(rview, imagePoints2[3], imagePoints2[4], Scalar(0, 255, 255), 2, 8);
			line(rview, imagePoints2[0], imagePoints2[5], Scalar(0, 255, 255), 2, 8);
			line(rview, imagePoints2[5], imagePoints2[4], Scalar(0, 255, 255), 2, 8);
			line(rview, imagePoints2[1], imagePoints2[6], Scalar(0, 255, 255), 2, 8);
			line(rview, imagePoints2[2], imagePoints2[7], Scalar(0, 255, 255), 2, 8);

			// Showing nview and rview images
			imshow("map View", nview);
			imwrite("map.png", nview);
			imshow("Image View", rview);
			imwrite("result.png", rview);

			// Calculating elapsed time and frame rate using timer (commented out)
			//elapsed = timer.elapsed(); // nano seconds...
			//t = (elapsed.wall / 1000.0f / 1000.0f / 1000.0f) / (float)100;
			//fps = 1.0f / t ;
			//if (t > 0.001) {
			//    std::cout << "t=" << t << ", fps=" << fps << std::endl;
			//}
			// Restarting the timer (commented out)
			//timer.start();

			// Waiting for a key press
			char c = (char)waitKey();
			/* if( c  == ESC_KEY || c == 'q' || c == 'Q' )
				break;*/
        }
    }
    return 0;
}


// calculates the barycentric coordinates u, v, 
// and w of a point p with respect to a triangle formed 
// by vertices a, b, and c.
void Barycentric(Point2f p, Point2f a, Point2f b, Point2f c, float &u, float &v, float &w)
{
	// Calculate the vectors v0, v1, and v2
    Point2f v0 = b - a;
	Point2f v1 = c - a;
	Point2f v2 = p - a;

	// Calculate dot products
    float d00 = v0.dot( v0);
    float d01 = v0.dot(v1);
    float d11 = v1.dot( v1);
    float d20 = v2.dot( v0);
    float d21 = v2.dot( v1);

	// Calculate the denominator
    float denom = d00 * d11 - d01 * d01;

	// Calculate the barycentric coordinates
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
}

// The function cartesian takes five Point2f parameters: p0, p1, p2, 
// and the barycentric coordinates u, v, and w. It calculates the 
// Cartesian coordinate of a point within a triangle defined by vertices 
// p0, p1, and p2, using the given barycentric coordinates.
Point2f  cartesian( Point2f p0, Point2f p1, Point2f p2, float u, float v, float w) 
{
    // Calculate the Cartesian coordinate using the barycentric coordinates
    return u * p0 + v * p1 + w * p2;
}

//7,4,5,6
// The updatmap function updates the values of the nmap1 and nmap2 
// matrices based on the imagePoints2 and the provided dimensions bw and 
// bh. Here's the code with added comments:
void updatmap(Mat& nmap1, Mat& nmap2, vector<Point2f>& imagePoints2, int bw, int bh, float scale)
{
    float u, v, w;
    Point2f ret;

    for (int j = 0; j < nmap1.rows; j++)
    {
        for (int i = 0; i < nmap1.cols; i++)
        {
            ret = Point2f(0, 0);
            if ((j < scale * bw) || (i < scale * bh))
            {
                // Calculate the barycentric coordinates for the current pixel
                Barycentric(Point2f(i, j), Point2f(0, 0), Point2f(scale * bw, 0), Point2f(0, scale * bh), u, v, w);

                if ((u + v + w) <= 1.001 && (u >= 0 && v >= 0 && w >= 0))
                {
                    // Calculate the interpolated Cartesian coordinate within the triangle using the barycentric coordinates
                    ret = cartesian(imagePoints2[6], imagePoints2[4], imagePoints2[5], u, v, w);
                }
                else
                {
                    Barycentric(Point2f(i, j), Point2f(scale * bw, scale * bh), Point2f(scale * bw, 0), Point2f(0, scale * bh), u, v, w);
                    if ((u + v + w) <= 1.001 && (u >= 0 && v >= 0 && w >= 0))
                    {
                        ret = cartesian(imagePoints2[7], imagePoints2[4], imagePoints2[5], u, v, w);
                    }
                }
            }

            nmap1.at<float>(j, i) = ret.x;
            nmap2.at<float>(j, i) = ret.y;
        }
    }
}


// The computeReprojectionErrors function calculates the reprojection errors for a 
// calibrated camera.

// The function takes as input the 3D object points (objectPoints), 
// corresponding 2D image points (imagePoints), rotation vectors (rvecs), 
// translation vectors (tvecs), camera matrix (cameraMatrix), and distortion 
// coefficients (distCoeffs). It initializes variables to store the 
// reprojected image points, total error, and per-view errors.

// The function then iterates over each set of object points and image points. 
// It uses the projectPoints function to project the 3D object points onto the 
// 2D image plane using the given camera parameters. It calculates the Euclidean 
// distance between the observed image points and the reprojected image points.

// The per-view error is calculated as the root mean square error (RMSE) divided 
// by the number of points in each set. The total error accumulates the squared 
// errors from all sets, and the total number of points is also accumulated.

// Finally, the overall reprojection error is calculated as the root mean 
// square error (RMSE) of the total error divided by the total number of points. 
// The per-view errors are stored in the perViewErrors vector, and the overall 
// reprojection error is returned.
static double computeReprojectionErrors(const vector<vector<Point3f>>& objectPoints,
                                        const vector<vector<Point2f>>& imagePoints,
                                        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                        const Mat& cameraMatrix, const Mat& distCoeffs,
                                        vector<float>& perViewErrors)
{
    vector<Point2f> imagePoints2; // Stores the reprojected 2D image points
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size()); // Resize the vector to hold per-view errors

    for (i = 0; i < (int)objectPoints.size(); ++i)
    {
        // Project 3D object points to 2D image points using the camera parameters
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
                      distCoeffs, imagePoints2);

        // Calculate the Euclidean distance between the observed image points and the reprojected image points
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

        int n = (int)objectPoints[i].size();
        // Calculate the per-view error as the root mean square error (RMSE) divided by the number of points
        perViewErrors[i] = (float)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    // Calculate the overall reprojection error as the root mean square error (RMSE) divided by the total number of points
    return std::sqrt(totalErr / totalPoints);
}


// The calcBoardCornerPositions function is used to calculate the 3D world coordinates 
// of the corners of a calibration pattern.

// The function takes as input the size of the calibration pattern (boardSize), 
// the size of each square in the pattern (squareSize), and the pattern type 
// (patternType). It initializes an empty corners vector to store the calculated 
// 3D corner positions.

// The function then switches based on the patternType to calculate 
// the corner positions. For the CHESSBOARD or CIRCLES_GRID patterns, it iterates 
// over the rows and columns of the pattern and calculates the 3D coordinates of 
// each corner as (j * squareSize, i * squareSize, 0), where j represents the 
// column index, i represents the row index, and squareSize is the size of each square.
static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners,
                                     Settings::Pattern patternType /*= Settings::CHESSBOARD*/)
{
    corners.clear(); // Clear the corners vector

    switch (patternType)
    {
    case Settings::CHESSBOARD:
    case Settings::CIRCLES_GRID:
        // Calculate corners for chessboard or circles grid pattern
        for (int i = 0; i < boardSize.height; ++i)
        {
            for (int j = 0; j < boardSize.width; ++j)
            {
                corners.push_back(Point3f(float(j * squareSize), float(i * squareSize), 0));
            }
        }
        break;

    case Settings::ASYMMETRIC_CIRCLES_GRID:
        // Calculate corners for asymmetric circles grid pattern
        for (int i = 0; i < boardSize.height; ++i)
        {
            for (int j = 0; j < boardSize.width; ++j)
            {
                corners.push_back(Point3f(float((2 * j + i % 2) * squareSize), float(i * squareSize), 0));
            }
        }
        break;

    default:
        break;
    }
}


// The runCalibration function is used to perform camera calibration using 
// a set of calibration images and corresponding detected image points.

// The function takes as input various calibration parameters (s) including the 
// board size, square size, calibration pattern type, and flags. It also takes 
// the imageSize of the calibration images, empty cameraMatrix and distCoeffs 
// matrices to store the intrinsic camera parameters, imagePoints which is a 
// vector of vectors containing the detected image points, and empty rvecs and 
// tvecs vectors to store the rotation and translation vectors for each image. 
// Additionally, it takes reprojErrs vector to store the reprojection errors 
// for each image, and a reference to totalAvgErr to store the average 
// reprojection error.

// The function initializes the cameraMatrix as an identity matrix, with the option 
// to fix the aspect ratio if specified by the flags. The distCoeffs matrix is 
// initialized as a zero matrix.

// It then calculates the 3D world coordinates of the calibration pattern corners 
// using the calcBoardCornerPositions function and stores them in the 
// objectPoints vector.

// The objectPoints vector is resized to match the number of imagePoints vectors. 
// This is done to ensure that the number of calibration pattern corners matches 
// the number of detected image points.

// The calibrateCamera function is called to estimate the intrinsic and extrinsic 
// camera parameters. It takes the objectPoints and imagePoints vectors, imageSize, 
// cameraMatrix, distCoeffs, and the flags as input. It outputs the root mean 
// square (RMS) re-projection error.

// After the calibration is performed, the function checks if the camera matrix 
// and distortion coefficients are within a valid range using the checkRange 
// function. It sets the ok variable accordingly.

// The function then computes the reprojection errors for each image using the 
// computeReprojectionErrors function. It takes the objectPoints, imagePoints, 
// rvecs, tvecs, cameraMatrix, distCoeffs, and reprojErrs as input, and outputs 
// the total average error.

// Finally, the function returns the ok variable indicating the success of the calibration process.
static bool runCalibration(Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                           vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
                           vector<float>& reprojErrs, double& totalAvgErr)
{
    // Initialize cameraMatrix as identity matrix with size 3x3
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if (s.flag & CV_CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = 1.0;

    // Initialize distCoeffs as zero matrix with size 8x1
    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    // Calculate 3D world coordinates of the corners of the calibration pattern
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);

    // Resize objectPoints vector to match the number of imagePoints vectors
    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    // Find intrinsic and extrinsic camera parameters using the calibration images and detected image points
    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                                 distCoeffs, rvecs, tvecs, s.flag | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);

    cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

    // Check if the camera matrix and distortion coefficients are within a valid range
    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    // Compute the reprojection errors
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                                            rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}


// Print camera parameters to the output file
// The saveCameraParams function is used to save the camera 
// calibration parameters and related information to a file 
// using the OpenCV FileStorage class. 

// It creates a FileStorage object to write the camera parameters to the specified output file.

// It saves various information such as calibration time, number of frames, image width and 
// height, board width and height, square size, and aspect ratio (if applicable).

// It saves the camera matrix and distortion coefficients.

// It saves the average reprojection error and per-view reprojection errors (if available).

// If rotation and translation vectors (rvecs and tvecs) are available, it saves them as 
// a set of 6-tuples representing the extrinsic parameters for each view.

// If image points (imagePoints) are available, it saves them as a matrix.

// The function uses comments to provide additional information in the file.
static void saveCameraParams( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                              const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                              const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
                              double totalAvgErr )
{


    FileStorage fs( s.outputFileName, FileStorage::WRITE );

    time_t tm;
    time( &tm );
    struct tm *t2 = localtime( &tm );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_Time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nrOfFrames" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_Width" << imageSize.width;
    fs << "image_Height" << imageSize.height;
    fs << "board_Width" << s.boardSize.width;
    fs << "board_Height" << s.boardSize.height;
    fs << "square_Size" << s.squareSize;

    if( s.flag & CV_CALIB_FIX_ASPECT_RATIO )
        fs << "FixAspectRatio" << s.aspectRatio;

    if( s.flag )
    {
        sprintf( buf, "flags: %s%s%s%s",
            s.flag & CV_CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "",
            s.flag & CV_CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "",
            s.flag & CV_CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "",
            s.flag & CV_CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "" );
        cvWriteComment( *fs, buf, 0 );

    }

    fs << "flagValue" << s.flag;

    fs << "Camera_Matrix" << cameraMatrix;
    fs << "Distortion_Coefficients" << distCoeffs;

    fs << "Avg_Reprojection_Error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "Per_View_Reprojection_Errors" << Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "Extrinsic_Parameters" << bigmat;
    }

    if( !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "Image_points" << imagePtMat;
    }
}




// Print camera parameters to the output file
// The readCameraParams function is used to read camera calibration 
// parameters and related information from a file using the OpenCV 
// FileStorage class.

// It creates a FileStorage object to read the camera parameters from the specified file.

// If the file cannot be opened, it returns false to indicate failure.

// It reads the image width and height from the file and assigns them to the imageSize parameter.

// It reads the camera matrix and distortion coefficients from the file and assigns them to the 
// cameraMatrix and distCoeffs parameters, respectively.

// It reads the extrinsic parameters (rvecs and tvecs) from the file and assigns them to the 
// respective output parameters. The extrinsic parameters are stored as a matrix, where each 
// row represents a set of 6-tuples (rotation vector + translation vector) for a view.

// After reading all the parameters, it returns true to indicate success.

// This function allows you to read the camera calibration parameters and associated data 
// from a file, which can be useful for loading pre-calibrated parameters for camera calibration 
// or other applications.
 bool readCameraParams( cv::String  s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                               vector<Mat>& rvecs,  vector<Mat>& tvecs
                               )
{
    FileStorage fs( s, FileStorage::READ );


	if ( !fs.isOpened())
		return false;

	fs["image_Width"]>>imageSize.width;
	fs["image_Height"]>>imageSize.height;

    fs["Camera_Matrix"]>> cameraMatrix ;
    fs["Distortion_Coefficients"] >> distCoeffs;

	 Mat bigmat;
	 fs["Extrinsic_Parameters"]>>bigmat;
	 int nrows=bigmat.rows;
	 
	
	 for( int i = 0 ; i < nrows; i++)
	 {
		    Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));
			rvecs.push_back(r);
            tvecs.push_back(t);
	 }

	 return true;
}


// The runCalibrationAndSave function performs camera calibration using the 
// provided settings and image points, and saves the calibration results 
// to a file.

// It declares variables to store the reprojection errors (reprojErrs) and 
// the total average error (totalAvgErr).

// It calls the runCalibration function to perform camera calibration, 
// passing in the provided settings, image size, camera matrix, distortion 
// coefficients, image points, and output vectors for rotation vectors 
// (rvecs) and translation vectors (tvecs).

// It prints whether the calibration succeeded or failed, along with the 
// average reprojection error.

// If the calibration succeeded (ok is true), it calls the saveCameraParams 
// function to save the camera calibration parameters, including the camera 
// matrix, distortion coefficients, rotation vectors, translation vectors, 
// reprojection errors, and image points, to a file specified in the settings.

// Finally, it returns the status of the calibration (ok).

// This function combines the calibration and saving steps into a 
// single function for convenience.
bool runCalibrationAndSave(Settings& s, Size imageSize, Mat&  cameraMatrix, Mat& distCoeffs,vector<vector<Point2f> > imagePoints,vector<Mat>& rvecs, vector<Mat> &tvecs )
{
   // vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(s,imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs,
                             reprojErrs, totalAvgErr);
    cout << (ok ? "Calibration succeeded" : "Calibration failed")
        << ". avg re projection error = "  << totalAvgErr ;

    if( ok )
        saveCameraParams( s, imageSize, cameraMatrix, distCoeffs, rvecs ,tvecs, reprojErrs,
                            imagePoints, totalAvgErr);
    return ok;
}
