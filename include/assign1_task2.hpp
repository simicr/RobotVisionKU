#include <string>
#include <opencv2/core.hpp>

int calculateExtrinsicParams(std::string leftCameraList, std::string rightCameraList, cv::Size boardSize, float squareSize, 
std::string leftParamsInputPath, std::string rightParamsInputPath, std::string extrinsicsOutputPath);