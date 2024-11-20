#ifndef HAND_TRACKER_HPP //if not defined 
#define HAND_TRACKER_HPP //now define so that if this file is included multiple times, we only include once since second time HAND_TRACKER_HPP is now defined the below ifndef is not run

#include <opencv2/opencv.hpp>
#include <vector>

/*
#include <> syntax is used to include header files that are part of the system's
standard library or other predefined paths configured by the compiler. These paths are
specified during the installation of the compileror when configuring the build system.
*/

class HandTracker {
public:
    HandTracker();
    void processFrame(cv::Mat& frame);
    void setBackground(const cv::Mat& frame);
    void controlTrackbars();
    void hsvTrackbars();
    void set_number_of_fingertips(int fingers);
    void changeRoi(int, int, int, int);
    bool currentHsvWindow();
    void changeHsvWindow(bool);
    cv::Rect currentRoi();
    int get_number_of_fingertips();
    int getMode();
private:
    int SENSIB, fingertip_to_centroid_distance;
    int hmin, smin, vmin, hmax, smax, vmax, mode, number_of_fingers;
    int width, height; //of roi
    bool isHsvWindowOpen;
    cv::Rect roi;
    cv::Mat background;

    // Internal methods: are to used inside other function of the class(inside the class)
    void binaryMode(cv::Mat& frame);
    void hsvMode(cv::Mat& frame);
    void findFingertips(std::vector<cv::Point>& points, std::vector<cv::Point> hull,
        cv::Mat& frame, cv::Point centroid);
    cv::Point getCentroid(const std::vector<cv::Point>& contour);
};

#endif // HAND_TRACKER_HPP