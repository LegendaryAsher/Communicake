#include "Utilities.hpp"
#include <cmath>

float distance(const cv::Point& a, const cv::Point& b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

float angleBetweenPoints(const cv::Point& start, const cv::Point& farthest, const cv::Point& end) {
    float a = distance(start, farthest);
    float b = distance(end, farthest);
    float c = distance(start, end);
    return std::acos((a * a + b * b - c * c) / (2 * a * b)) * 180.0 / CV_PI;
}

int largestContour(const std::vector<std::vector<cv::Point>>& contours) {
    if (contours.empty()) return -1;
    int largestIndex = -1;
    double largestArea = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > largestArea) {
            largestArea = area;
            largestIndex = i;
        }
    }
    return largestArea > 400 ? largestIndex : -1;
}