#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

static void StereoCalib(const vector<string>& imagelist, Size boardSize, float squareSize, 
                        string leftParamsInputPath, string rightParamsInputPath, string intristic_output, string extrinsicsOutputPath, 
                        bool displayCorners = false, bool useCalibrated=true, bool showRectified=true)
{
    if( imagelist.size() % 2 != 0 )
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    const int maxScale = 2;
    
    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;
    int i, j, k, nimages = (int)imagelist.size()/2;
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;
    for( i = j = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            const string& filename = imagelist[i*2+k];
            Mat img = imread(filename, 0);
            if(img.empty())
                break;
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale);
                found = findChessboardCorners(timg, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                
                if( found )
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }

            if( displayCorners )
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf);
                imshow("corners", cimg1);
                char c = (char)waitKey(500);
                if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            
            if( !found )
                break;
            cornerSubPix(img, corners, Size(11,11), Size(-1,-1), TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));
        }

        if( k == 2 )
        {
            goodImageList.push_back(imagelist[i*2]);
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    
    }
    cout << endl << j << " pairs have been successfully detected." << endl;
    nimages = j;
    if( nimages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for( i = 0; i < nimages; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
    }

    Mat cameraMatrix[2], distCoeffs[2];
    FileStorage leftParams(leftParamsInputPath, 0);
    FileStorage rightParams(rightParamsInputPath, 0);
    leftParams["camera_matrix"] >> cameraMatrix[0];
    leftParams["distortion_coefficients"] >> distCoeffs[0];
    rightParams["camera_matrix"] >> cameraMatrix[1]; 
    rightParams["distortion_coefficients"] >> distCoeffs[1];
    leftParams.release();
    rightParams.release();

    cout << "Running stereo calibration ...\n";
    Mat R, T, E, F;
    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,
                    CALIB_FIX_INTRINSIC,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    cout << "done with RMS error=" << rms << endl;


    // CALIBRATION QUALITY CHECK
    // because the output fundamental matrix implicitly
    // includes all the output information,
    // we can check the quality of calibration using the
    // epipolar geometry constraint: m2^t*F*m1=0
    
    // Needed? 
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for( i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for( k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for( j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "Avg. epipolar err = " <<  err/npoints << endl;

    FileStorage fs(intristic_output, FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }

    fs.open(extrinsicsOutputPath, FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "RMSE" << rms;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

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

int calculateExtrinsicParams(string leftCameraList, string rightCameraList, Size boardSize, float squareSize, string leftParamsInputPath, string rightParamsInputPath, string intristic_output, string extrinsicsOutputPath)
{
    vector<string> leftImageList;
    bool ok = readStringList(leftCameraList, leftImageList);
    if(!ok || leftImageList.empty())
    {
        cout << "can not open " << leftCameraList << " or the string list is empty" << endl;
        return 0;
    } else {
        cout << "Number of images of the left camera: " << leftImageList.size() << endl;
    }

    vector<string> rightImageList;
    ok = readStringList(rightCameraList, rightImageList);
    if(!ok || rightImageList.empty())
    {
        cout << "can not open " << rightCameraList << " or the string list is empty" << endl;
        return 0;
    } else {
        cout << "Number of images of the right camera: " << rightImageList.size() << endl;
    }

    vector<string> imageListPairs;
    for(int i = 0; i < leftImageList.size(); i++)
    {
        imageListPairs.push_back(leftImageList[i]);
        imageListPairs.push_back(rightImageList[i]);
    }

    StereoCalib(imageListPairs, boardSize, squareSize, leftParamsInputPath,
                rightParamsInputPath, intristic_output,extrinsicsOutputPath, 
                false, true, false);
    return 0;
}

int main()
{   
    string left_calibration = "fileoutput/left_calib.yaml";
    string right_calibration = "fileoutput/right_calib.yaml";
    string left_input = "data/CALIB_DATA/left";
    string right_input = "data/CALIB_DATA/right";
    string intristic_output = "fileoutput/intrinsics.yaml";
    string output_path = "fileoutput/extrinsic.yaml";

    calculateExtrinsicParams(left_input, right_input, Size(8,5), 30.0, left_calibration, right_calibration,intristic_output,output_path);

    return 0;
}
