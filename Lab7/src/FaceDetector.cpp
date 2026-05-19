#include "../include/FaceDetector.hpp"
#include <chrono>

FaceDetector::FaceDetector() : running(true), hasNewFrame(false) {
    net = cv::dnn::readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");
    worker = std::thread(&FaceDetector::run, this);
}

FaceDetector::~FaceDetector() {
    running = false;
    if (worker.joinable()) {
        worker.join();
    }
}

void FaceDetector::setFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(mtx);
    currentFrame = frame.clone();
    hasNewFrame = true;
}

std::vector<cv::Rect> FaceDetector::getFaces() {
    std::lock_guard<std::mutex> lock(mtx);
    return detectedFaces;
}

void FaceDetector::run() {
    while (running) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (!hasNewFrame || currentFrame.empty()) {
                continue; 
            }
            frame = currentFrame.clone();
            hasNewFrame = false;
        }

        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
        net.setInput(blob);
        
        cv::Mat detections = net.forward();

        std::vector<cv::Rect> faces;
        cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > 0.5) {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
                faces.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        {
            std::lock_guard<std::mutex> lock(mtx);
            detectedFaces = faces;
        }
    }
}
