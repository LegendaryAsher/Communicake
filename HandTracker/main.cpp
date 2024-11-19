#include "HandTracker.hpp" 

int main() {
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
    cv::namedWindow("ROI Controls", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("X", "ROI Controls", &x, 1000);
    cv::createTrackbar("Y", "ROI Controls", &y, 1000);
    cv::createTrackbar("WIDTH", "ROI Controls", &w, 1000);
    cv::createTrackbar("HEIGHT", "ROI Controls", &h, 1000);
    //after change roi controls, do press b to reset background because the background has dimension of 
    //previous roi, so absdiff function will throw an error when dimension do not match
    //hence i've only called the function absdiff in binaryMode only when dimension match
    //after resetting background the dimension will match then absdiff will always be called

    tracker.setBackground(frame);

    while (true) {
        // Clamp trackbar values to valid ranges
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
                tracker.setMode(1);
            }
            else {
                tracker.setMode(0);
            }
        }
        if (key == 114) { //'r' to get current roi dimensions
            cv::Rect roi = tracker.currentRoi();
            std::cout << "X: " << roi.x << std::endl;
            std::cout << "Y: " << roi.y << std::endl;
            std::cout << "Width: " << roi.width << std::endl;
            std::cout << "Height: " << roi.height << std::endl;
        }
    }
    std::cout << tracker.get_number_of_fingertips() << std::endl;
    //cleaning up
    cam.release();
    cv::destroyAllWindows();
    return 0;
}
