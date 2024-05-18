#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
 
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
 
void detection_and_matching(string& output, Mat img1, Mat img2, Ptr<Feature2D> detector, bool binary=false, Ptr<Feature2D> descriptor=nullptr) 
{
    const float ratio_thresh = 0.7f;

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    if (descriptor != nullptr) 
    {
        detector->detect(img1, keypoints1);
        detector->detect(img2, keypoints2);

        descriptor->compute(img1, keypoints1, descriptors1);
        descriptor->compute(img2, keypoints2, descriptors2);
    }
    else
    {
        detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
        detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );
    }
    
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(binary ? DescriptorMatcher::BRUTEFORCE_HAMMING : DescriptorMatcher::FLANNBASED);
    vector< vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
     
    Mat img_keypoints1, img_keypoints2;
    Mat img_matches;

    drawKeypoints(img1, keypoints1, img_keypoints1);
    drawKeypoints(img2, keypoints2, img_keypoints2);
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
    Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    string detector_name = detector->getDefaultName();
    string descriptor_name = descriptor == nullptr ? detector_name : descriptor->getDefaultName();
    int posdot = detector_name.find('.');
    detector_name = detector_name.substr(posdot+1);
    descriptor_name = descriptor_name.substr(posdot+1);

    imwrite( output + detector_name + "1.png", img_keypoints1);
    imwrite( output + detector_name + "2.png", img_keypoints2);
    imwrite( output + detector_name + "+" + descriptor_name +".png", img_matches);

    cout << "Good matches: " << good_matches.size() << endl;
    FileStorage fs(output + "matches_"+ detector_name +".yaml", FileStorage::WRITE | FileStorage::APPEND);
    fs << "Matches" << good_matches;
    fs << "KP1" << keypoints1;
    fs << "KP2" << keypoints2;
    fs.release();
    
}

int feature_matching(string& left_image, string& right_image, string& output) 
{
    Mat img1 = imread( samples::findFile( left_image ), IMREAD_GRAYSCALE );
    Mat img2 = imread( samples::findFile( right_image ), IMREAD_GRAYSCALE );
    if ( img1.empty() || img2.empty() )
    {
    cout << "Could not open or find the image!\n" << endl;
    return -1;
    }

    Ptr<SURF> surf = SURF::create();
    Ptr<ORB> orb = ORB::create();
    Ptr<SIFT> sift = SIFT::create();
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();
    Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create();

    detection_and_matching(output, img1, img2, surf, false, nullptr);
    detection_and_matching(output, img1, img2, sift, false, nullptr);
    detection_and_matching(output, img1, img2, orb, true, nullptr);
    detection_and_matching(output, img1, img2, fast, true, brief);

    return 0;
}

int main( int argc, char* argv[] )
{
    string left8 = "IMG_CAL_DATA/left08.png";
    string left10 = "IMG_CAL_DATA/left10.png";
    string right8 = "IMG_CAL_DATA/right08.png";

    string matching_out1 = "fileoutput/LR/";
    string matching_out2 = "fileoutput/LL/";

    feature_matching( left8, right8, matching_out1);
    feature_matching( left8, left10, matching_out2);
}
