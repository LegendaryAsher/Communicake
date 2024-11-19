#include "HandTracker.hpp"
#include "Utilities.hpp"

HandTracker::HandTracker()
    : SENSIB(20), MAX_ANGLE(90), depth_threshold(10), fingertip_to_centroid_distance(100),
    hmin(0), smin(0), vmin(82), hmax(255), smax(36), vmax(255), mode(0),
    width(300), height(300), roi(300, 70, width, height), number_of_fingers(0), isHsvWindowOpen(false){} //default constructor

cv::Rect HandTracker::currentRoi() {
    return roi;
}

void HandTracker::changeRoi(int x, int y, int w, int h) {
    roi.x = x;
    roi.y = y;
    roi.width = w;
    roi.height = h;
}

void HandTracker::findFingertips(std::vector < cv::Point >& points, std::vector<cv::Point> hull, cv::Mat& img, cv::Point centroid) {
    for (int k = 0; k < hull.size(); ++k)
    {
        if (centroid.y + 20 > hull[k].y && distance(hull[k], centroid) > fingertip_to_centroid_distance && distance(hull[k], hull[(k + 1) % hull.size()]) > 30)
        {
            points.push_back(hull[k]);
        }
    }

    if (points.size() > 5) {
        int smallestIndex = 0;
        for (int i = 1; i < 5; i++) {
            if (points[smallestIndex].y > points[i].y) {
                smallestIndex = i;
            }
        }
        points.erase(points.begin() + smallestIndex);
    }

    for (int k = 0; k < points.size(); ++k)
    {
        cv::circle(img, points[k], 4, cv::Scalar(0, 0, 0), -4);
        cv::line(img, points[k], centroid, cv::Scalar(100, 100, 100), 2, 8);
    }

    cv::putText(img, std::to_string(points.size()), cv::Point(25, 25), cv::FONT_HERSHEY_COMPLEX, 0.75, cv::Scalar(255, 0, 255), 2);
}

void HandTracker::drawDefects(std::vector<cv::Point> contour, cv::Mat& img) {
    if (contour.empty()) return; //exit the function when empty

//this code crashes if in mode 1 hence we have try catch block so that even when it gives error it continues to run
//this is probably due to not finding anything in defects to draw 
//so drawContour must be the one giving error
//the error was due to convexityDefects function, the error in question:
/*  
    Bad argument(The convex hull indices are not monotonous,
    which can be in the case when the input contour contains 
    self-intersections) in cv::convexityDefects 
*/
//this error can be solved by using the function isContourConvex() meaning is the contour simple?(simple meaning no self-intersections)
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices, false, false);
    std::vector<cv::Vec4i> defects;
    if (!contour.empty() && hullIndices.size() > 3) {
        try {
            if (cv::isContourConvex(contour)) { //if no self-intersections
                cv::convexityDefects(contour, hullIndices, defects);
            }
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception in cv::convexityDefects: " << e.what() << std::endl;
        }
    }
    for (const auto& defect : defects) {
        cv::Point start = contour[defect[0]];
        cv::Point end = contour[defect[1]];
        cv::Point depth_point = contour[defect[2]];
        float depth = defect[3] / 256.0;

        // Filtering based on defect depth and position
        if (depth > depth_threshold) {
            cv::circle(img, depth_point, 5, cv::Scalar(255, 0, 0), -1);  // Marking defect points
        }
    }
}

void HandTracker::setBackground(const cv::Mat& frame) {
    frame.copyTo(background);
    background = background(roi);
}

void HandTracker::setMode(int newMode) {
    mode = newMode;
}

int HandTracker::getMode() {
    return mode;
}

void HandTracker::binaryMode(cv::Mat& frame) {
    cv::Mat bg;
    background.copyTo(bg);
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bg, bg, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0);
    cv::GaussianBlur(bg, bg, cv::Size(5, 5), 0);

    if (frame.cols == bg.cols && frame.rows == bg.rows) //checking dimensions match or not
    {
        cv::absdiff(frame, bg, frame);
    }

    cv::threshold(frame, frame, SENSIB, 255, 0);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(frame, frame, cv::MORPH_OPEN, kernel); //erosion followed by dilate

}

void HandTracker::hsvMode(cv::Mat& frame) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);
    cv::inRange(frame, cv::Scalar(hmin, smin, vmin), cv::Scalar(hmax, smax, vmax), frame);
}

void HandTracker::controlTrackbars() {
    cv::namedWindow("Controls", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Mode", "Controls", &mode, 2);
    cv::createTrackbar("Sensitivity", "Controls", &SENSIB, 255);
    cv::createTrackbar("Depth Threshold", "Controls", &depth_threshold, 255);
    cv::createTrackbar("MAX ANGLE", "Controls", &MAX_ANGLE, 255);
    cv::createTrackbar("Fingertip Distance", "Controls", &fingertip_to_centroid_distance, 255);
}

void HandTracker::hsvTrackbars() {
    //Trackers for hsv color detection
    if(!isHsvWindowOpen) 
    {
        cv::namedWindow("HSV VALUES", cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("HMIN", "HSV VALUES", &hmin, 255);
        cv::createTrackbar("SMIN", "HSV VALUES", &smin, 255);
        cv::createTrackbar("VMIN", "HSV VALUES", &vmin, 255);
        cv::createTrackbar("HMAX", "HSV VALUES", &hmax, 255);
        cv::createTrackbar("SMAX", "HSV VALUES", &smax, 255);
        cv::createTrackbar("VMAX", "HSV VALUES", &vmax, 255);

        isHsvWindowOpen = true;
    }
}

bool HandTracker::currentHsvWindow() {
    return isHsvWindowOpen;
}

void HandTracker::changeHsvWindow(bool a) {
    isHsvWindowOpen = a;
}

void HandTracker::set_number_of_fingertips(int fingers) {
    number_of_fingers = fingers;
}

int HandTracker::get_number_of_fingertips() {
    return number_of_fingers;
}

void HandTracker::processFrame(cv::Mat& frame) {
    cv::Mat mask, drawing;
    std::string toPut;
    frame.copyTo(mask);
    mask = mask(roi);
    mask.copyTo(drawing);

    if (mode == 0) { //background subtraction
        binaryMode(mask);
    }
    else if(mode == 1) { //hsv color space
        hsvMode(mask);
    } 
    else if (mode == 2) { //hybrid mode
        cv::Mat temp = mask;
        binaryMode(mask);
        hsvMode(temp);
        cv::addWeighted(mask, 0.5, temp, 1 - 0.5, 0.5, mask);
    }

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat temp;
    mask.copyTo(temp);
    cv::findContours(temp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 4);

    int largestContourIndex = largestContour(contours);
    if (largestContourIndex != -1) {
        cv::drawContours(drawing, contours, largestContourIndex, cv::Scalar(0, 0, 255), 1);  // draws outline (contour) in red (hand)

        std::vector< cv::Point > hull;
        cv::convexHull(cv::Mat(contours[largestContourIndex]), hull, false);   // saves the 'boundary' points of the contour without any inward dents to vector hull
        if (!hull.empty()) {
            //std::vector<std::vector<cv::Point>>{hull} equivalent to std::vector<std::vector<cv::Point>> hulls = {hull0, hull1, hull2, ..} where hulli is an vector of points
            cv::drawContours(drawing, std::vector<std::vector<cv::Point>>{hull}, -1, cv::Scalar(0, 215, 50), 2);

            cv::Point centroid = getCentroid(contours[largestContourIndex]);
            cv::circle(drawing, centroid, 3, cv::Scalar(255, 0, 0), -3);// drawing the centroid

            std::vector<cv::Point> fingertips;
            //finding fingertips point and pusing it into fingertips vector and drawing them and does some stuff ^_^
            findFingertips(fingertips, hull, drawing, centroid);
            set_number_of_fingertips(fingertips.size());
            switch (fingertips.size()) {
            case 1:
                toPut = "its a 1 right?";
                break;
            case 2:
                toPut = "Peace!";
                break;
            case 3:
                toPut = "maybe 3?";
                break;
            case 4:
                toPut = "4?";
                break;
            case 5:
                toPut = "Well Hello there";
                break;
            case 0:
                toPut = "A Fist?";
                break;
            }
            cv::putText(frame, toPut, cv::Point(100, 70), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 2, 2));
            //drawing depth points of defects
            drawDefects(contours[largestContourIndex], drawing);
            //findFingertipsUsingDefects(defects, contours[largestContourIndex], centroid, img);
        }
    }
    cv::imshow("Mask", mask);
    cv::imshow("Orignal", frame);
    cv::imshow("ROI", drawing);
}

cv::Point HandTracker::getCentroid(const std::vector<cv::Point>& contour) {
    cv::Moments moment = moments(contour);
    double x = moment.m10 / moment.m00;
    double y = moment.m01 / moment.m00;
    return cv::Point(x, y);
}
