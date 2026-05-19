#include "../include/CameraProvider.hpp"

CameraProvider::CameraProvider(int deviceId) {
    cap.open(deviceId);
}

CameraProvider::~CameraProvider() {
    if (cap.isOpened()) {
        cap.release();
    }
}

cv::Mat CameraProvider::getFrame() {
    cv::Mat frame;
    cap >> frame; // Читаємо кадр з камери
    return frame;
}
