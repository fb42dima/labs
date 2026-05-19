#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

class FaceDetector {
private:
    cv::dnn::Net net;
    std::thread worker;
    std::mutex mtx;
    std::atomic<bool> running;
    
    cv::Mat currentFrame;
    bool hasNewFrame;
    std::vector<cv::Rect> detectedFaces;

    void run();

public:
    FaceDetector();
    ~FaceDetector();
    
    void setFrame(const cv::Mat& frame);
    std::vector<cv::Rect> getFaces();
};
