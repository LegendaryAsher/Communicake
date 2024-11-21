#include "HandTracker.hpp" 
#include <cstdlib>
#include <algorithm>

int main() {
    std::cout << "Communicake - Communication made a piece of cake\nConnecting to the video capturing media, Please wait.....";
    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        std::cerr << "ERROR: Unable to open camera\n";
        return -1;
    }

    HandTracker tracker;
    int x = tracker.currentRoi().x, y = tracker.currentRoi().y;
    int w = tracker.currentRoi().width, h = tracker.currentRoi().height;

    cv::Mat frame;

    //creating trackers for thresholding controls and hsv values controls
    tracker.controlTrackbars();
    cam >> frame;
    cv::flip(frame, frame, 1);
    int frameWidth = frame.cols;
    int frameHeight = frame.rows;

    //ROI Controls
    cv::namedWindow("ROI Controls", cv::WINDOW_NORMAL);
    cv::resizeWindow("ROI Controls", 300, 70);
    cv::moveWindow("ROI Controls", 60, 100);
    cv::createTrackbar("X", "ROI Controls", nullptr, 1000);
    cv::createTrackbar("Y", "ROI Controls", nullptr, 1000);
    cv::createTrackbar("WIDTH", "ROI Controls", nullptr, 1000);
    cv::createTrackbar("HEIGHT", "ROI Controls", nullptr, 1000);

    cv::setTrackbarPos("X", "ROI Controls", x);
    cv::setTrackbarPos("Y", "ROI Controls", y);
    cv::setTrackbarPos("WIDTH", "ROI Controls", w);
    cv::setTrackbarPos("HEIGHT", "ROI Controls", h);
    tracker.setBackground(frame);
    while (true) {
        x = cv::getTrackbarPos("X", "ROI Controls");
        y = cv::getTrackbarPos("Y", "ROI Controls");
        w = cv::getTrackbarPos("WIDTH", "ROI Controls");
        h = cv::getTrackbarPos("HEIGHT", "ROI Controls");
        //Clamp trackbar values to valid ranges
        x = std::max(0, std::min(x, frameWidth - 1));
        y = std::max(0, std::min(y, frameHeight - 1));
        w = std::max(1, std::min(w, frameWidth - x));
        h = std::max(1, std::min(h, frameHeight - y));
        tracker.changeRoi(x, y, w, h);

        if (tracker.getMode() == 1) {
            tracker.hsvTrackbars();
        }
        else {
            if (tracker.currentHsvWindow()) {
                cv::destroyWindow("HSV VALUES");
                tracker.changeHsvWindow(false);
            }
        }

        cam >> frame;
        if (frame.empty()) break;
        cv::flip(frame, frame, 1);
        tracker.processFrame(frame);

        int key = cv::waitKey(1);
        if (key == 27) break; // ESC to exit
        if (key == 98) {    // 'b' to update background  
            if (frame.empty()) {
                return -1;
            }
            tracker.setBackground(frame);
        }
        if (key == 109) { //'m' to toggle mode
            if (tracker.getMode() == 0) {
                cv::setTrackbarPos("Mode", "Controls", 1);
            }
            else {
                cv::setTrackbarPos("Mode", "Controls", 0);
            }
        }
        if (key == 99) { // 'c' to capture the frame, get no. of fingers and open a particular app 
            if (tracker.get_number_of_fingertips() == 1) {
                system("code");
            }
            if (tracker.get_number_of_fingertips() == 2) {
                system("notepad");
            }
            if (tracker.get_number_of_fingertips() == 3) {
                system("start chrome");
            }
            if (tracker.get_number_of_fingertips() == 4) {
                system("start spotify");
            }
            if (tracker.get_number_of_fingertips() == 5) {
                system("calc");
            }
        }
    }
    //cleaning up
    cam.release();
    cv::destroyAllWindows();
    return 0;
}