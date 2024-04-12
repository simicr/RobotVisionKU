#include <iostream>
#include <filesystem>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include "../include/assign1_task2.hpp"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

class Settings
{
public:
    Settings(string inputPath, string outputPath, Size boardSize, float squareSize) : input(inputPath), outputFileName(outputPath), boardSize(8,5), squareSize(30){

        bool ok = readStringList(input, imageList);
        if (!ok) 
        {
            cerr << "Error while reading input";
        } else {
            // Double check
            nrFrames = (int) imageList.size(); 
            fixK1 = false;
            fixK2 = false;
            fixK3 = false;
            fixK4 = true;
            fixK5 = true;
            showUndistorted = true;
            calibFixPrincipalPoint = false;
            calibZeroTangentDist = false;
            atImageList = 0;
            flipVertical = false;
            writeExtrinsics = false;
            aspectRatio = false; // Can be both removed?

            flag = 0;
            if(calibFixPrincipalPoint) flag |= CALIB_FIX_PRINCIPAL_POINT;
            if(calibZeroTangentDist)   flag |= CALIB_ZERO_TANGENT_DIST;
            if(aspectRatio)            flag |= CALIB_FIX_ASPECT_RATIO;
            if(fixK1)                  flag |= CALIB_FIX_K1;
            if(fixK2)                  flag |= CALIB_FIX_K2;
            if(fixK3)                  flag |= CALIB_FIX_K3;
            if(fixK4)                  flag |= CALIB_FIX_K4;
            if(fixK5)                  flag |= CALIB_FIX_K5;
        }
        
    };

    Mat nextImage()
    {
        Mat result;
        if( inputCapture.isOpened() )
        {
            Mat view0;
            inputCapture >> view0;
            view0.copyTo(result);
        }
        else if( atImageList < imageList.size() )
            result = imread(imageList[atImageList++], IMREAD_COLOR);

        return result;
    }

    static bool readStringList( const string& filename, vector<string>& l )
    {
        l.clear();
        for (const auto& entry : fs::directory_iterator(filename)) {
            if (fs::is_regular_file(entry)) {
                l.push_back(entry.path().string());
            }
        }
        return true;
    }

public:
    VideoCapture inputCapture; 
    Size boardSize;              // + The size of the board -> Number of items by width and height
    vector<string> imageList; 
    string outputFileName;       // + The name of the file where to write
    string input;                // + The input
    size_t atImageList; 
    int nrFrames;                // + The number of frames to use from the input for calibration
    int flag; 
    float squareSize;            // + The size of a square in your defined unit (point, millimeter,etc).
    float markerSize;            // ? The size of a marker in your defined unit (point, millimeter,etc).
    float aspectRatio;           // ? The aspect ratio
    bool ok;
    bool showUndistorted;        // + Show undistorted images after calibration
    bool calibZeroTangentDist;   // + Assume zero tangential distortion
    bool calibFixPrincipalPoint; // + Fix the principal point at the center
    bool fixK1;                  // + fix K1 distortion coefficient
    bool fixK2;                  // + fix K2 distortion coefficient
    bool fixK3;                  // + fix K3 distortion coefficient
    bool fixK4;                  // + fix K4 distortion coefficient
    bool fixK5;                  // + fix K5 distortion coefficient  
    bool flipVertical;           // ? Flip the captured images around the horizontal axis
    bool writeExtrinsics;        // ? Write extrinsic parameters
    bool writeGrid;              // ? Write refined 3D target grid points
};
enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };
bool runCalibrationAndSave(Settings& s, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                           vector<vector<Point2f> > imagePoints);
int calibrationSingleCamera(string inputPath, string outputPath);

int main(int argc, char* argv[]) {


    cout << "Left calibration:\n";
    string leftCameraList = "./data/CALIB_DATA/left";
    string leftCameraOutput = "./fileoutput/left_calib";
    // calibrationSingleCamera(leftCameraList, leftCameraOutput);

    cout << "Right calibration:\n";
    string rightCameraList = "./data/CALIB_DATA/right";
    string rightCameraOutput = "./fileoutput/right_calib";
    // calibrationSingleCamera(rightCameraList, rightCameraOutput);

    calculateExtrinsicParams(leftCameraList, rightCameraList, Size(8,5), 30, leftCameraOutput, rightCameraOutput, "./fileoutput/externals");

    return 0;
}

int calibrationSingleCamera(string inputPath, string outputPath)
{
    Settings s(inputPath, outputPath, Size(8,5), 30);
    vector<vector<Point2f>> imagePoints;
    Mat cameraMatrix, distCoeffs;
    Size imageSize;

    int mode = CAPTURING;
    int winSize = 11; // What is this?
    const char ESC_KEY = 27;

    for(;;)
    {
        Mat view;
        view = s.nextImage();
        bool blinkOutput = false; // Needed?
        
        if( mode == CAPTURING && imagePoints.size() >= (size_t)s.nrFrames )
        {
          if(runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints))
          {
            mode = CALIBRATED;
          } 
          else 
          {
            mode = DETECTION;
          }          
        }
        if(view.empty())  
        {
            if( mode != CALIBRATED && !imagePoints.empty() )
                runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints);
            break;
        }
        imageSize = view.size();
        if( s.flipVertical )    flip( view, view, 0 ); // Needed?

        vector<Point2f> pointBuf;
        bool found;
        int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

        found = findChessboardCorners( view, s.boardSize, pointBuf, chessBoardFlags);
        if (found)                
        {
            Mat viewGray;
            cvtColor(view, viewGray, COLOR_BGR2GRAY);
            cornerSubPix( viewGray, pointBuf, Size(winSize,winSize), Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001 ));
            imagePoints.push_back(pointBuf);
        }
    }

    if(s.showUndistorted && !cameraMatrix.empty())
    {
        Mat view, rview, map1, map2;
        initUndistortRectifyMap( cameraMatrix, distCoeffs, Mat()
        , getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0)
        , imageSize, CV_16SC2, map1, map2);

        for(size_t i = 0; i < s.imageList.size(); i++ )
        {
            view = imread(s.imageList[i], IMREAD_COLOR);
            if(!view.empty())
            {
                remap(view, rview, map1, map2, INTER_LINEAR);
                imshow("Image View", rview);
                char c = (char)waitKey();
                if( c  == ESC_KEY || c == 'q' || c == 'Q' )
                    break;
            }
        }
    }

    return 0;
}

static double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                         const vector<vector<Point2f> >& imagePoints,
                                         const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                         const Mat& cameraMatrix , const Mat& distCoeffs,
                                         vector<float>& perViewErrors)
{
    vector<Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for(size_t i = 0; i < objectPoints.size(); ++i )
    {   
        projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
        err = norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }
    return std::sqrt(totalErr/totalPoints);
}

static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.clear();
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            corners.push_back(Point3f(j*squareSize, i*squareSize, 0));
        }
    }    
}

static bool runCalibration( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                            vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
                            vector<float>& reprojErrs,  double& totalAvgErr, vector<Point3f>& newObjPoints)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if(s.flag & CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = s.aspectRatio; // ?     
    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0]);
    newObjPoints = objectPoints[0];
    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    double rms;
    rms = calibrateCamera(objectPoints, imagePoints, imageSize ,cameraMatrix, distCoeffs, rvecs, tvecs, s.flag | CALIB_USE_LU);

    cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;
    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
    objectPoints.clear();
    objectPoints.resize(imagePoints.size(), newObjPoints);
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
                                            distCoeffs, reprojErrs);
    return ok;
}

static void saveCameraParams( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                              const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                              const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
                              double totalAvgErr, const vector<Point3f>& newObjPoints )
{
    FileStorage fs( s.outputFileName, FileStorage::WRITE );
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "avg_reprojection_error" << totalAvgErr;

    if (s.writeExtrinsics && !reprojErrs.empty())
    {
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);
    }

    if(s.writeExtrinsics && !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, CV_MAKETYPE(rvecs[0].type(), 1));
        bool needReshapeR = rvecs[0].depth() != 1 ? true : false;
        bool needReshapeT = tvecs[0].depth() != 1 ? true : false;

        for( size_t i = 0; i < rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(int(i), int(i+1)), Range(0,3));
            Mat t = bigmat(Range(int(i), int(i+1)), Range(3,6));

            if(needReshapeR)
                rvecs[i].reshape(1, 1).copyTo(r);
            else
            {
                CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
                r = rvecs[i].t();
            }

            if(needReshapeT)
                tvecs[i].reshape(1, 1).copyTo(t);
            else
            {
                CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
                t = tvecs[i].t();
            }
        }
        fs.writeComment("a set of 6-tuples (rotation vector + translation vector) for each view");
        fs << "extrinsic_parameters" << bigmat;
    }
}

bool runCalibrationAndSave(Settings& s, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                           vector<vector<Point2f> > imagePoints)
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    vector<Point3f> newObjPoints;

    bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrs,
                             totalAvgErr, newObjPoints);
    cout << (ok ? "Calibration succeeded" : "Calibration failed")
         << ". avg re projection error = " << totalAvgErr << endl;

    if (ok)
        saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, imagePoints,
                         totalAvgErr, newObjPoints);
    return ok;
}