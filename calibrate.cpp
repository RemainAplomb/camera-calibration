
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


struct AxisAngle4d
{
	double angle;
	double x;
	double y;
	double z;
};

//http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
 void matrixFromAxisAngle(AxisAngle4d a1 , Mat & r21 ) {

	 
	double c = cos(a1.angle);
	double s = sin(a1.angle);
	double t = 1.0 - c;
	//  if axis is not already normalised then uncomment this
	// double magnitude = Math.sqrt(a1.x*a1.x + a1.y*a1.y + a1.z*a1.z);
	// if (magnitude==0) throw error;
	// a1.x /= magnitude;
	// a1.y /= magnitude;
	// a1.z /= magnitude;

	double m00 = c + a1.x*a1.x*t;
	double m11 = c + a1.y*a1.y*t;
	double m22 = c + a1.z*a1.z*t;


	double tmp1 = a1.x*a1.y*t;
	double tmp2 = a1.z*s;
	double m10 = tmp1 + tmp2;
	double m01 = tmp1 - tmp2;
	 tmp1 = a1.x*a1.z*t;
	 tmp2 = a1.y*s;
	double 	m20 = tmp1 - tmp2;
	double 	m02 = tmp1 + tmp2;    
	tmp1 = a1.y*a1.z*t;
	tmp2 = a1.x*s;
	double m21 = tmp1 + tmp2;
	double m12 = tmp1 - tmp2;

	r21.at<double>(0, 0) =m00;
	 r21.at<double>(1, 0)=m01;
	r21.at<double>(2, 0)=m02;

	r21.at<double>(0, 1)= m10;
	 r21.at<double>(1, 1)= m11;
	 r21.at<double>(2, 1)=m12;


	r21.at<double>(0, 2)=m20;
	 r21.at<double>(1, 2)=m21;
	 r21.at<double>(2, 2)=m22;

}


void readPara(Camera & cam)
{

	char  buffer[1024];
	GetModuleFileName(NULL,(LPWSTR) buffer, 1024);
	TCHAR NPath[MAX_PATH];
	GetCurrentDirectory(MAX_PATH, NPath);

	string::size_type pos = string(buffer).find_last_of("\\/");
	string tmp= string(buffer).substr(0, pos);
   // char* pf= NPath;
	char nstring [1024];
	size_t convertedChars;

	size_t origsize = wcslen(NPath) + 1;
	size_t newsize = origsize*2;

	wcstombs_s(&convertedChars, nstring, newsize, NPath, _TRUNCATE);


//	MessageBox(0,(LPWSTR) tmp.c_str(), L"MessageBox caption", MB_OK);

//	FILE * outf = fopen("out.txt", "w+");
//	fprintf(outf, "%s\n", buffer );
//	fclose(outf);
//	MessageBox(0, L"OK", L"MessageBox caption", MB_OK);

	strstr(buffer, "/");



	const cv::String outputFileName = cv::String("C:\\james\\images_circle\\out_camera_data.xml");

	//Camera cam;

	vector<Mat> rvecs, tvecs;
	Size imageSize;

	Mat qview;
	Mat rot;
	
	if ( bshow)
	qview = imread("C:\\james\\images_circle\\test2.jpg");
	//MessageBox(0,(LPCWSTR)(outputFileName.c_str()), L"MessageBox caption", MB_OK);


	if (!readCameraParams(outputFileName, imageSize, cam.cameraMatrix, cam.distCoeffs, rvecs, tvecs))
	{

		return;

	}
//	MessageBox(0, L"read", L"MessageBox caption", MB_OK);


	cout << cam.cameraMatrix << endl;


//	MessageBox(0, L"out", L"MessageBox caption", MB_OK);

	//cam.init(rvecs[1], tvecs[1]);

//	MessageBox(0, L"init", L"MessageBox caption", MB_OK);


//	rot = rvecs[0];

	//cv::Mat RM(3, 3, cv::DataType<double>::type);

	Mat rot0;
	cv::Rodrigues(rvecs[0], rot0);

	Mat rot1;
	cv::Rodrigues(rvecs[1], rot1);

	cout << tvecs[0] << endl;
	cout << rot0 << endl;

	cout << tvecs[1] << endl;
	cout << rot1 << endl;


	Mat r21 = rot1*rot0.inv() ;

	
	AxisAngle4d a1;
	a1.angle = 22.5f*M_PI/180;
	a1.x = 0;
	a1.y = 0;
	a1.z = 1;

	matrixFromAxisAngle(a1, r21);

	//Mat t21 = -r21*tvecs[0].t() + tvecs[1].t(); 理論上的值.但如果轉點在圓心,可假設t 都一樣,故可取t[0]
	Mat t21 = -r21*tvecs[0].t() + tvecs[0].t();


	Mat pw=Mat::zeros(3,1,CV_64F);
	pw.at<double>(0) = 25.0f;
	
	Mat p1 = rot0*pw+ tvecs[0].t();
	Mat p2 = rot1*pw + tvecs[1].t();
	Mat p21 = r21 * p1 + t21;


	
	cout << p1 << endl;
	cout << p2 << endl;
	cout << p21 << endl;


	double R00 = r21.at<double>(0, 0);
	double R01 = r21.at<double>(1, 0);
	double R02 = r21.at<double>(2, 0);

	double R10 = r21.at<double>(0, 1);
	double R11 = r21.at<double>(1, 1);
	double R12 = r21.at<double>(2, 1);


	double R20 = r21.at<double>(0,2);
	double R21 = r21.at<double>(1, 2);
	double R22 = r21.at<double>(2, 2);


	double angle = acos((R00 + R11 + R22 - 1) / 2)*180/M_PI;
	double  ax, ay, az;
	ax = (R21 - R12) / sqrt((R21 - R12)*(R21 - R12) + (R02 - R20)*(R02 - R20) + (R10 - R01)*(R10 - R01));
	ay = (R02 - R20) / sqrt((R21 - R12)*(R21 - R12) + (R02 - R20)*(R02 - R20) + (R10 - R01)*(R10 - R01));
	az = (R10 - R01) / sqrt((R21 - R12)*(R21 - R12) + (R02 - R20)*(R02 - R20) + (R10 - R01)*(R10 - R01));

	cout << "angle:" << angle << endl;



	int nt =tvecs.size();

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

	Mat distorted_points2d;
	//projectPoints(Mat(newpoints), cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), cameraMatrix, distCoeffs, distorted_points2d);
	projectPoints(Mat(newpoints), rvecs[1], tvecs[1], cam.cameraMatrix,
		cam.distCoeffs, distorted_points2d);

	
	Scalar cc3 = cv::Scalar(0, 0, 255, 255); //BGRA
	int cBlockSize = 5;

	for (int i = 0; i < distorted_points2d.rows; i++)
	{
		Point2f pt = distorted_points2d.at<Point2f>(i);

		rectangle(qview, cv::Rect(pt.x - cBlockSize, pt.y - cBlockSize, cBlockSize * 2 + 1, cBlockSize * 2 + 1), cc3);
	}


//	MessageBox(0, L"ok", L"MessageBox caption", MB_OK);


	if (bshow)
	{
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
        x = default_value;
    else
        x.read(node);
}

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
    fs["Settings"] >> s;
    fs.release();                                         // close Settings file

    if (!s.goodInput)
    {
        cout << "Invalid input detected. Application stopping. " << endl;
        return -1;
    }

    vector<vector<Point2f> > imagePoints;
    Mat cameraMatrix, distCoeffs;
	vector<Mat> rvecs, tvecs;
    Size imageSize;
    int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;
	  clock_t prevTimestamp = 0;
    const Scalar RED(0,0,255), GREEN(0,255,0);
	const char ESC_KEY = 27;
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
                    imagePoints.push_back(pointBuf);
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
    help();
    Settings s;
    const string inputSettingsFile = argc > 1 ? argv[1] : "C:\\james\\images\\default.xml";



    FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
    if (!fs.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
        return -1;
    }
    fs["Settings"] >> s;
    fs.release();                                         // close Settings file

    if (!s.goodInput)
    {
        cout << "Invalid input detected. Application stopping. " << endl;
        return -1;
    }

    vector<vector<Point2f> > imagePoints;
    Mat cameraMatrix, distCoeffs;
	vector<Mat> rvecs, tvecs;
    Size imageSize;
    int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;
    clock_t prevTimestamp = 0;
    const Scalar RED(0,0,255), GREEN(0,255,0);
    const char ESC_KEY = 27;

	if (! readCameraParams(  s.outputFileName, imageSize,  cameraMatrix, distCoeffs,  rvecs,   tvecs ))
	{


	}

	//boost::timer::cpu_timer timer;

	//timer.start();



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

		float bw=169;//195;//169;//ss*5;
		float bh=230;//265;//228;//ss*7;
		float bt=31.5;//19;//32;//ss;

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
		
			 projectPoints( Mat(newpoints), rvecs[i], tvecs[i], newMatrix,
                       distCoeffs, imagePoints2);


			  nmap1.create( view.size(), CV_32FC1 );  
			  nmap2.create( view.size(), CV_32FC1 );  

			  float scale=4.0f;

			vector<Point3f>  map3d;		
		    vector<Point2f> mapp;


			//boost::timer::cpu_times elapsed = timer.elapsed(); // nano seconds...

			//float t = (elapsed.wall / 1000.0f / 1000.0f / 1000.0f) / (float)100;

			//float fps = 1.0f / t ;

			//if (t > 0.001) {

			//	std::cout << "t=" << t << ", fps=" << fps << std::endl;

			//}

			//timer.start();



			  for( int j = 0; j < nmap1.rows; j++ )  
			  { 
					for( int i = 0; i < nmap1.cols; i++ )     
					{
						  if ( (i < scale*bw +50) && (j < scale*bh +50)) 
						  {
			
						map3d.push_back(Point3d(bh-j/scale ,newy -bw+i/scale,-bt));
					

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


			  	
						 projectPoints( Mat(map3d), rvecs[0], tvecs[0], newMatrix,
								distCoeffs, mapp);

// elapsed = timer.elapsed(); // nano seconds...

	//		 t = (elapsed.wall / 1000.0f / 1000.0f / 1000.0f) / (float)100;

		/*	 fps = 1.0f / t ;

			if (t > 0.001) {

				std::cout << "t=" << t << ", fps=" << fps << std::endl;

			}

			timer.start();
*/


                int k = 0 ; 
			    for( int j = 0; j < nmap1.rows; j++ )  
			  { 
					for( int i = 0; i < nmap1.cols; i++ )     
					{
						  if ( (i < scale*bw +50) && (j < scale*bh +50)) 
						  {
				
						  nmap1.at<float>(j,i) = mapp[k].x ;  
						  nmap2.at<float>(j,i) = mapp[k].y ;
						  k++;

						}else
						{
							 nmap1.at<float>(j,i) = 0;  
							  nmap2.at<float>(j,i) = 0 ;  
						}
						 
					}
	            }

		//	 elapsed = timer.elapsed(); // nano seconds...

		/*	 t = (elapsed.wall / 1000.0f / 1000.0f / 1000.0f) / (float)100;

			 fps = 1.0f / t ;

			if (t > 0.001) {

				std::cout << "t=" << t << ", fps=" << fps << std::endl;

			}

			timer.start();*/


			// updatmap( nmap1,nmap2, imagePoints2 ,bw,bh,scale);
			  remap(rview, nview, nmap1, nmap2, INTER_LINEAR);


			  //在變正的封面上劃線
			   line(nview,Point2f(0,0*scale),Point2f(bw*scale,0*scale), Scalar( 0, 0, 255 ),2,8);
			  line(nview,Point2f(0,60*scale),Point2f(bw*scale,60*scale), Scalar( 0, 255, 0 ),2,8);
			  line(nview,Point2f(0,77*scale),Point2f(bw*scale,77*scale), Scalar( 0, 0, 255 ),2,8);
			  line(nview,Point2f(0,bh*scale),Point2f(bw*scale,bh*scale), Scalar( 0, 0, 255 ),2,8);

			 ellipse( rview, imagePoints2[8],Size( 5, 5 ),0,0,360,Scalar( 255, 0, 0 ),2,8 );
			 line(rview,imagePoints2[0],imagePoints2[1], Scalar( 0, 255, 0 ),2,8);
			 line(rview,imagePoints2[0],imagePoints2[2], Scalar( 0, 0, 255 ),2,8);
			 line(rview,imagePoints2[1],imagePoints2[3], Scalar( 0, 0, 255 ),2,8);
			 line(rview,imagePoints2[2],imagePoints2[3], Scalar( 0, 0, 255 ),2,8);
			 line(rview,imagePoints2[3],imagePoints2[4], Scalar( 0, 255, 255 ),2,8);
			 line(rview,imagePoints2[0],imagePoints2[5], Scalar( 0, 255, 255 ),2,8);
			 line(rview,imagePoints2[5],imagePoints2[4], Scalar( 0, 255, 255 ),2,8);
			 line(rview,imagePoints2[1],imagePoints2[6], Scalar( 0, 255, 255 ),2,8);
			 line(rview,imagePoints2[2],imagePoints2[7], Scalar( 0, 255, 255 ),2,8);

			 
			   imshow("map View", nview);
			   imwrite("map.png",nview);
            imshow("Image View", rview);
			imwrite("result.png",rview);
//
		 //elapsed = timer.elapsed(); // nano seconds...

			// t = (elapsed.wall / 1000.0f / 1000.0f / 1000.0f) / (float)100;

			// fps = 1.0f / t ;

			//if (t > 0.001) {

			//	std::cout << "t=" << t << ", fps=" << fps << std::endl;

			//}

			//timer.start();



            char c = (char)waitKey();
           /* if( c  == ESC_KEY || c == 'q' || c == 'Q' )
                break;*/
        }
    }


    return 0;
}

void Barycentric(Point2f p, Point2f a, Point2f b, Point2f c, float &u, float &v, float &w)
{
    Point2f v0 = b - a;
	Point2f v1 = c - a;
	Point2f v2 = p - a;
    float d00 = v0.dot( v0);
    float d01 = v0.dot(v1);
    float d11 = v1.dot( v1);
    float d20 = v2.dot( v0);
    float d21 = v2.dot( v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
}

Point2f  cartesian( Point2f p0, Point2f p1, Point2f p2, float u, float v, float w) 
{
      return u * p0 + v * p1 + w * p2;
}

//7,4,5,6
void  updatmap(Mat& nmap1, Mat& nmap2,vector<Point2f> & imagePoints2 , int bw,int bh, float scale )
{
	

	float u,v,w;
	Point2f ret ;

	

	for( int j = 0; j < nmap1.rows; j++ )  
    { 
		for( int i = 0; i < nmap1.cols; i++ )     
		{
			 ret= Point2f(0,0);
			  if ( (j < scale*bw) || (i < scale*bh)) 
			  {
				    Barycentric(Point2f(i,j),Point2f(0,0),Point2f(scale*bw,0),Point2f(0,scale*bh),u,v,w);
					if (( u+v+w)<=1.001 && ( u>=0 && v>=0 && w>=0)) 
					{
						ret = cartesian(imagePoints2[6],imagePoints2[4],imagePoints2[5],u,v,w);
				   
					}else
					{
					   
					 Barycentric(Point2f(i,j),Point2f(scale*bw,scale*bh),Point2f(scale*bw,0),Point2f(0,scale*bh),u,v,w);
					 if (( u+v+w)<=1.001 && ( u>=0 && v>=0 && w>=0)) 
						  ret = cartesian(imagePoints2[7],imagePoints2[4],imagePoints2[5],u,v,w);
				     
					}
				  

				  

			  }
			  nmap1.at<float>(j,i) = ret.x ;  
              nmap2.at<float>(j,i) = ret.y ;  
		}
	}
}

static double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                         const vector<vector<Point2f> >& imagePoints,
                                         const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                         const Mat& cameraMatrix , const Mat& distCoeffs,
                                         vector<float>& perViewErrors)
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); ++i )
    {
        projectPoints( Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
                       distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);
}

static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners,
                                     Settings::Pattern patternType /*= Settings::CHESSBOARD*/)
{
    corners.clear();

    switch(patternType)
    {
    case Settings::CHESSBOARD:
    case Settings::CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; ++i )
            for( int j = 0; j < boardSize.width; ++j )
                corners.push_back(Point3f(float( j*squareSize ), float( i*squareSize ), 0));
        break;

    case Settings::ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float((2*j + i % 2)*squareSize), float(i*squareSize), 0));
        break;
    default:
        break;
    }
}

static bool runCalibration( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                            vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
                            vector<float>& reprojErrs,  double& totalAvgErr)
{

    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if( s.flag & CV_CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = 1.0;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    //Find intrinsic and extrinsic camera parameters
    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                                 distCoeffs, rvecs, tvecs, s.flag|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

    cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                                             rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}

// Print camera parameters to the output file
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
