#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame; // Capture a frame
        if (frame.empty())
            break;

        // Process frame...

        // Display the resulting frame
        cv::imshow("Frame", frame);

        // Break loop on 'q' key press
        if (cv::waitKey(1) == 'q')
            break;
    }

    // Release the video capture object
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
