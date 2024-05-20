
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

enum UseAlgorithm {
    FUNDAMENTAL8P,
    FUNDAMENTAL8P_RANSAC,
    ESSENTIAL_RANSAC,
};

enum CameraSetup
{
    LEFT_ONLY,
    LEFT_RIGHT
};

string getAlogorithmFromEnum(UseAlgorithm a)
{
    switch(a)
    {
        case FUNDAMENTAL8P: return "Fundamental8P";break;
        case FUNDAMENTAL8P_RANSAC: return "Fundamental8P_RANSAC";break;
        case ESSENTIAL_RANSAC: return "Essential_RANSAC";break;
    };
    return "This is not the string you are looking for";
};

void drawLines(Mat& inImg1, Mat& inImg2, Mat F, vector<Point2d> pts1, vector<Point2d> pts2, Mat inliers)
{
//  ''' img1 - image on which we draw the epilines for the points in img2
//  lines - corresponding epilines '''
    Mat img1, img2; 
    cvtColor(inImg1, img1, COLOR_GRAY2BGR);
    cvtColor(inImg2, img2, COLOR_GRAY2BGR);

    Mat points1 = Mat((int)pts1.size(),  2, CV_64F, pts1.data());
    Mat points2 = Mat((int)pts2.size(), 2, CV_64F, pts2.data());
    vconcat(points1.t(), Mat::ones(1, points1.rows, points1.type()), points1);
    vconcat(points2.t(), Mat::ones(1, points2.rows, points2.type()), points2);

    // cout << "Points 1: " << points1.row(0) << endl;

    RNG rng(rand());
    const int circle_sz = 4, line_sz = 1, max_lines = 300;

    //shuffle then choose first 20 points
    std::vector<int> pts_shuffle (points1.cols);
    for (int i = 0; i < points1.cols; i++)
        pts_shuffle[i] = i;

    std::srand(std::time(0));
    random_shuffle(pts_shuffle.begin(), pts_shuffle.end());
    
    for(int j = 0; j < 20; j++)
    {
        const int i = pts_shuffle[j];
        const Scalar col (rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));

        const Mat l2 = F  * points1.col(i);
        const Mat l1 = F.t() * points2.col(i);

        //from opencv samples - epipolar_lines.cpp
        double a1 = l1.at<double>(0), b1 = l1.at<double>(1), c1 = l1.at<double>(2);
        double a2 = l2.at<double>(0), b2 = l2.at<double>(1), c2 = l2.at<double>(2);
        const double mag1 = sqrt(a1*a1 + b1*b1), mag2 = (a2*a2 + b2*b2);
        a1 /= mag1; b1 /= mag1; c1 /= mag1; a2 /= mag2; b2 /= mag2; c2 /= mag2;


        line(img1, Point2d(0, -c1/b1),
                Point2d((double)img1.cols, -(a1*img1.cols+c1)/b1), col, line_sz);
        // line(img2, Point2d(0, -c2/b2),
        //         Point2d((double)img2.cols, -(a2*img2.cols+c2)/b2), col, line_sz);
        circle (img1, pts1[i], circle_sz, col, -1);
        circle (img2, pts2[i], circle_sz, col, -1);
    }
    inImg1 = img1;
    inImg2 = img2;
}

void processEpipolarLines(string matchesFilePath, string img1Path, string img2Path, UseAlgorithm alg, CameraSetup camSetup, string output, string detectorName)
{
    vector<DMatch> matches;
    vector<KeyPoint> kp1, kp2;
    vector<Point2d> pts1, pts2;
    vector<Point2d> normPoints1, normPoints2;

    vector<double> KLeftParams = {9.842439e+02, 0.000000e+00, 6.900000e+02, 0.000000e+00, 9.808141e+02, 2.331966e+02, 
    0.000000e+00, 0.000000e+00, 1.000000e+00};

    vector<double> KRightParams = {9.895267e+02, 0.000000e+00, 7.020000e+02, 0.000000e+00, 9.878386e+02, 2.455590e+02,
     0.000000e+00, 0.000000e+00, 1.000000e+00};

    Mat KLeft(3, 3, CV_64F, KLeftParams.data());
    Mat KRight(3, 3, CV_64F, KRightParams.data());

    FileStorage matchesFile(matchesFilePath, 0);
    matchesFile["Matches"] >> matches;
    matchesFile["KP1"] >> kp1;
    matchesFile["KP2"] >> kp2;

    Mat img1 = imread( samples::findFile( img1Path ), IMREAD_GRAYSCALE );
    Mat img2 = imread( samples::findFile( img2Path ), IMREAD_GRAYSCALE );


    cout << detectorName << endl;

    for(auto match : matches)
    {
        Point2f& p1 = kp1[match.queryIdx].pt;
        Point2f& p2 = kp2[match.trainIdx].pt;
        pts1.push_back(kp1[match.queryIdx].pt);
        normPoints1.push_back(Point2d( (2* p1.x / img1.cols) - 1, (2 * p1.y / img1.rows) - 1));
        pts2.push_back(kp2[match.trainIdx].pt);
        normPoints2.push_back(Point2d( (2* p2.x / img1.cols) - 1, (2 * p2.y / img1.rows) - 1));
    }

    Mat fundamentalMat, inliers;

    if(alg == ESSENTIAL_RANSAC)
    {
        Mat kLeftInv = KLeft.inv();
        Mat kRightInv = KRight.inv();

        if(camSetup == LEFT_ONLY)
        {
            Mat essentialMat;
            essentialMat = findEssentialMat(pts1, pts2, KLeft, RANSAC, 0.99, 1, inliers);
            fundamentalMat = kLeftInv.t() * essentialMat * kLeftInv;
        }
        else
        {
            Mat essentialMat, R, t;
            //gets essential matrix using both camera params
            recoverPose(pts1, pts2, KLeft, noArray(),  KRight, noArray(), essentialMat, R, t, RANSAC, 0.99, 1, inliers);
            fundamentalMat = kRightInv.t() * essentialMat * kLeftInv;
        }
    }
    else
    {
        //for normalized 8 point method
        vector<double> Kvector = {
        2.0 / img1.size[1], 0 , -1,
        0, 2.0 / img1.size[0] , -1,
        0, 0                  , 1 };

        Mat K(Size(3,3), CV_64F, Kvector.data());

        Mat fn; 
        if(alg == FUNDAMENTAL8P)
            fn = findFundamentalMat(normPoints1, normPoints2, FM_8POINT, 1., 0.99, inliers);
        else
            fn = findFundamentalMat(normPoints1, normPoints2, RANSAC, 1., 0.99, 2000, inliers);

        fundamentalMat = K.t() * fn * K;
    }

    cout << "\tFundamental: " << fundamentalMat.size() << endl<< "\t" << fundamentalMat << endl;
    cout << "\tDeterminant: " << determinant(fundamentalMat) << endl;

    vector<Point2d> inlierPts1, inlierPts2;
    for(int i = 0; i < inliers.rows ; i ++)
    {
        if(inliers.row(i).at<double>(0))
        {
            inlierPts1.push_back(pts1[i]);
            inlierPts2.push_back(pts2[i]);
        } 
    };

    drawLines(img1, img2, fundamentalMat, inlierPts1, inlierPts2, inliers);
    hconcat(img1, img2, img1);
    string fileName = output + detectorName + "_" + getAlogorithmFromEnum(alg) + ".png";
    cout << "\tSaving: " << fileName << endl;
    imwrite( fileName, img1);
};

int main() {
    vector<DMatch> matches;
    vector<KeyPoint> kp1, kp2;
    vector<Point2d> pts1, pts2;
    vector<Point2d> normPoints1, normPoints2;

    vector<double> KLeftParams = {9.842439e+02, 0.000000e+00, 6.900000e+02, 0.000000e+00, 9.808141e+02, 2.331966e+02, 
    0.000000e+00, 0.000000e+00, 1.000000e+00};

    vector<double> KRightParams = {9.895267e+02, 0.000000e+00, 7.020000e+02, 0.000000e+00, 9.878386e+02, 2.455590e+02,
     0.000000e+00, 0.000000e+00, 1.000000e+00};

    Mat KLeft(3, 3, CV_64F, KLeftParams.data());
    Mat KRight(3, 3, CV_64F, KRightParams.data());

    string left8 = "IMG_CAL_DATA/left08.png";
    string left10 = "IMG_CAL_DATA/left10.png";
    string right8 = "IMG_CAL_DATA/right08.png";
    string outputLeftOnly = "fileoutput/LL/estimation/";
    string outputLeftRight = "fileoutput/LR/estimation/";
    string inputLeftOnly = "fileoutput/LL/";
    string inputLeftRight = "fileoutput/LR/";
    string matchesFast = "matches_FastFeatureDetector.yaml";
    string matchesORB = "matches_ORB.yaml";
    string matchesSIFT = "matches_SIFT.yaml";
    string matchesSURF = "matches_SURF.yaml";

    //FAST =================================
    processEpipolarLines(inputLeftOnly + matchesFast, left8, left10, ESSENTIAL_RANSAC, LEFT_ONLY, outputLeftOnly,"FastFeatureDetector");
    processEpipolarLines(inputLeftOnly + matchesFast, left8, left10, FUNDAMENTAL8P, LEFT_ONLY, outputLeftOnly, "FastFeatureDetector");
    processEpipolarLines(inputLeftOnly + matchesFast, left8, left10, FUNDAMENTAL8P_RANSAC, LEFT_ONLY, outputLeftOnly,"FastFeatureDetector");

    processEpipolarLines(inputLeftRight + matchesFast, left8, right8, ESSENTIAL_RANSAC, LEFT_RIGHT, outputLeftRight, "FastFeatureDetector");
    processEpipolarLines(inputLeftRight + matchesFast, left8, right8, FUNDAMENTAL8P, LEFT_RIGHT, outputLeftRight, "FastFeatureDetector");
    processEpipolarLines(inputLeftRight + matchesFast, left8, right8, FUNDAMENTAL8P_RANSAC, LEFT_RIGHT, outputLeftRight,"FastFeatureDetector");

    //====================================

    //ORB =================================
    processEpipolarLines(inputLeftOnly + matchesORB, left8, left10, ESSENTIAL_RANSAC, LEFT_ONLY, outputLeftOnly,"ORB");
    processEpipolarLines(inputLeftOnly + matchesORB, left8, left10, FUNDAMENTAL8P, LEFT_ONLY, outputLeftOnly, "ORB");
    processEpipolarLines(inputLeftOnly + matchesORB, left8, left10, FUNDAMENTAL8P_RANSAC, LEFT_ONLY, outputLeftOnly,"ORB");

    processEpipolarLines(inputLeftRight + matchesORB, left8, right8, ESSENTIAL_RANSAC, LEFT_RIGHT, outputLeftRight, "ORB");
    processEpipolarLines(inputLeftRight + matchesORB, left8, right8, FUNDAMENTAL8P, LEFT_RIGHT, outputLeftRight, "ORB");
    processEpipolarLines(inputLeftRight + matchesORB, left8, right8, FUNDAMENTAL8P_RANSAC, LEFT_RIGHT, outputLeftRight,"ORB");

    //====================================

    //SURF =================================
    processEpipolarLines(inputLeftOnly + matchesSURF, left8, left10, ESSENTIAL_RANSAC, LEFT_ONLY, outputLeftOnly,"SURF");
    processEpipolarLines(inputLeftOnly + matchesSURF, left8, left10, FUNDAMENTAL8P, LEFT_ONLY, outputLeftOnly, "SURF");
    processEpipolarLines(inputLeftOnly + matchesSURF, left8, left10, FUNDAMENTAL8P_RANSAC, LEFT_ONLY, outputLeftOnly,"SURF");

    processEpipolarLines(inputLeftRight + matchesSURF, left8, right8, ESSENTIAL_RANSAC, LEFT_RIGHT, outputLeftRight, "SURF");
    processEpipolarLines(inputLeftRight + matchesSURF, left8, right8, FUNDAMENTAL8P, LEFT_RIGHT, outputLeftRight, "SURF");
    processEpipolarLines(inputLeftRight + matchesSURF, left8, right8, FUNDAMENTAL8P_RANSAC, LEFT_RIGHT, outputLeftRight,"SURF");

    //====================================

    //SIFT =================================
    processEpipolarLines(inputLeftOnly + matchesSIFT, left8, left10, ESSENTIAL_RANSAC, LEFT_ONLY, outputLeftOnly,"SIFT");
    processEpipolarLines(inputLeftOnly + matchesSIFT, left8, left10, FUNDAMENTAL8P, LEFT_ONLY, outputLeftOnly, "SIFT");
    processEpipolarLines(inputLeftOnly + matchesSIFT, left8, left10, FUNDAMENTAL8P_RANSAC, LEFT_ONLY, outputLeftOnly,"SIFT");

    processEpipolarLines(inputLeftRight + matchesSIFT, left8, right8, ESSENTIAL_RANSAC, LEFT_RIGHT, outputLeftRight, "SIFT");
    processEpipolarLines(inputLeftRight + matchesSIFT, left8, right8, FUNDAMENTAL8P, LEFT_RIGHT, outputLeftRight, "SIFT");
    processEpipolarLines(inputLeftRight + matchesSIFT, left8, right8, FUNDAMENTAL8P_RANSAC, LEFT_RIGHT, outputLeftRight,"SIFT");

    //====================================
    return 0;
}
