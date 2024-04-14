#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

static void savePLY(const string& filename, const Mat& mat) {

    FILE* fp = fopen(filename.c_str(), "wt");

    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", mat.rows * mat.cols);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    // TODO: See how to get color information.
    //fprintf(fp, "property unchar red\n");
    //fprintf(fp, "property unchar green\n");
    //fprintf(fp, "property unchar blue\n");
    fprintf(fp, "end_header\n");

    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            Vec3f point = mat.at<Vec3f>(y, x);
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
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

    string header;
    getline(file, header);
    if (header == "PF") {
        color = true;
    } else if (header == "Pf") {
        color = false;
    } else {
        throw runtime_error("Not a PFM file.");
    }

    file >> width >> height;
    file >> scale;

    if (scale < 0) {
        endian = '<';
        scale = -scale;
    } else {
        endian = '>';
    }

    Mat data(height, width, CV_8UC1);
    vector<float> buffer(width * height * 3);
    file.read(reinterpret_cast<char*>(buffer.data()), sizeof(float) * width * height * 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float* pixel = data.ptr<float>(y, x);
            for (int c = 0; c < 3; ++c) {
                int index = (color ? 3 * (width * y + x) + c : width * y + x);
                pixel[c] = buffer[index];
            }
        }
    }
    flip(data, data, 0);

    return data;
}


int main(int argc, char** argv)
{
    string img1_filename = "data/STEREO_DATA/stereo_data/left/picture_000001.png";
    string img2_filename = "data/STEREO_DATA/stereo_data/right/picture_000001.png";
    string left_camera = "fileoutput/left_calib.yaml";
    string right_camera = "fileoutput/right_calib.yaml";
    string extrinsic_filename = "fileoutput/extrinsic.yaml";
    string disparity_filename = "fileoutput/img0_disp.pfm";
    string point_cloud_filename = "fileoutput/point_cloud.ply";
    string rectified_output1 = "fileoutput/rectify/img0.png";
    string rectified_output2 = "fileoutput/rectify/img1.png";

    int SADWindowSize = 0, numberOfDisparities = 0;
    bool no_display = true;
    bool calculate_disparity = false;
    float scale = 1.f;

    // Read and scale images
    // TODO: Is it okay to use grayscaled?
    int color_mode = 0;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);
    if( scale != 1.f )
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }
    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;

    // Loading the data
    FileStorage fs(left_camera, FileStorage::READ);

    Mat M1, D1, M2, D2;
    fs["camera_matrix"] >> M1;
    fs["distortion_coefficients"] >> D1;

    fs.open(right_camera, FileStorage::READ);
    fs["camera_matrix"] >> M2;
    fs["distortion_coefficients"] >> D2;
    
    fs.open(extrinsic_filename, FileStorage::READ);
    Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;
    
    
    // Rectifying images. 
    // TODO: Double check
    stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );
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

    if (calculate_disparity & ok)
    {
        int returnCode = system("./gmstereo_demo.sh");
    }

    Mat disp, xyz;
    disp = read_pfm(disparity_filename);
    reprojectImageTo3D(disp, xyz, Q, true);
    savePLY(point_cloud_filename, xyz);

    return 0;
}