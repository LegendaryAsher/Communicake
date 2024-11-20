#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <opencv2/opencv.hpp>

float distance(const cv::Point& a, const cv::Point& b);
float angleBetweenPoints(const cv::Point& start, const cv::Point& farthest, const cv::Point& end);
int largestContour(const std::vector<std::vector<cv::Point>>& contours);

#endif // UTILITIES_HPP