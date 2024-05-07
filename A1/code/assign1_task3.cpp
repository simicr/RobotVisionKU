#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <regex>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

static void save_ply(const string& filename, const Mat& mat, const Mat& img) {

    FILE* fp = fopen(filename.c_str(), "wt");

    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", mat.rows * mat.cols);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");

    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            Vec3f point = mat.at<Vec3f>(y, x);
            Vec3b color = img.at<Vec3b>(y, x);
            fprintf(fp, "%f %f %f %d %d %d\n", point[0], point[1], point[2], color[2], color[1], color[0]);
        }
    }

    fclose(fp);
}

Mat read_pfm(const string& filename) 
{
    ifstream file(filename, ios::binary);

    bool color = false;
    int width = 0, height = 0;
    float scale = 0.0f;
    char endian = ' ';
    int chanels;

    string header;
    string line;

    getline(file, header);
    if (header == "PF") {
        color = true;
    } else if (header == "Pf") {
        color = false;
    } 

    getline(file, line);
    regex dim_regex(R"((\d+)\s(\d+))");
    smatch dim_match;

    if (regex_match(line, dim_match, dim_regex)) {
        width = stoi(dim_match[1]);
        height = stoi(dim_match[2]);
    }

    getline(file, line);
    scale = stof(line);
    if (scale < 0) {
        endian = '<';
        scale = -scale;
    } else {
        endian = '>';
    }

    if(color){
        chanels = 3;
    } else {
        chanels = 1;
    }
   

    vector<float> buffer(width * height * chanels);
    file.read(reinterpret_cast<char*>(buffer.data()), sizeof(float) * width * height * chanels);
    
    Mat data;
    if (color) {
        data = Mat(height, width, CV_32FC3, buffer.data()).clone();
    } else {
        data = Mat(height, width, CV_32FC1, buffer.data()).clone();
    }

    flip(data, data, 0);

    return data;
}


int main(int argc, char** argv)
{
    string img1_filename = "data/STEREO_DATA/stereo_data/left/picture_000001.png";
    string img2_filename = "data/STEREO_DATA/stereo_data/right/picture_000001.png";

    string intristic_filename = "fileoutput/intrinsics.yaml";
    string extrinsic_filename = "fileoutput/extrinsic.yaml";
    
    string disparity_filename = "fileoutput/img0_disp.pfm";
    string point_cloud_filename = "fileoutput/point_cloud.ply";
    string rectified_output1 = "fileoutput/rectify/img0.png";
    string rectified_output2 = "fileoutput/rectify/img1.png";

    bool calculate_disparity = true;
    int color_mode = IMREAD_UNCHANGED;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);
    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;

    // Loading the data
    FileStorage fs(intristic_filename, FileStorage::READ);

    Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;
    
    Mat R, T, R1, P1, R2, P2;
    fs.open(extrinsic_filename, FileStorage::READ);
    fs["R"] >> R;
    fs["T"] >> T;
    
    
    // Rectifying images. 
    stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0, img_size, &roi1, &roi2 );
    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
    Mat img1r, img2r;
    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);
    img1 = img1r;
    img2 = img2r;
    bool ok = imwrite(rectified_output1, img1);
    ok = ok & imwrite(rectified_output2, img2);

    // Calculate disparity via Unimatch
    // Read the pfm file, reproject the points and write them in the ply file.
    Mat disp, xyz;
    if (calculate_disparity & ok)
    {
        int returnCode = system("./gmstereo_demo.sh");
    }
    disp = read_pfm(disparity_filename);
    reprojectImageTo3D(disp, xyz, Q);
    save_ply(point_cloud_filename, xyz, img1);

    return 0;
}