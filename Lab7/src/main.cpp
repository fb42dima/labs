#include "CameraProvider.hpp"
#include "KeyProcessor.hpp"
#include "FrameProcessor.hpp"
#include "Display.hpp"
#include "FaceDetector.hpp"
#include <chrono>
#include <string>

int main() {
    CameraProvider camera(0);
    KeyProcessor keyProc;
    FrameProcessor frameProc;
    Display display("Lab 7");
    FaceDetector faceDetector;

    auto prevTime = std::chrono::high_resolution_clock::now();

    while (true) {
        cv::Mat frame = camera.getFrame();
        if (frame.empty()) break;

        Mode currentMode = keyProc.getCurrentMode();
        cv::Mat processedFrame = frameProc.process(frame, currentMode);

        if (currentMode == Mode::FACE_DETECT || currentMode == Mode::FACE_BLUR) {
            faceDetector.setFrame(frame); 
            auto faces = faceDetector.getFaces(); 
            
            for (const auto& rect : faces) {
                cv::Rect safeRect = rect & cv::Rect(0, 0, processedFrame.cols, processedFrame.rows);
                if (safeRect.area() == 0) continue;

                if (currentMode == Mode::FACE_DETECT) {
                    cv::rectangle(processedFrame, safeRect, cv::Scalar(0, 255, 0), 3);
                    cv::putText(processedFrame, "Face", cv::Point(safeRect.x, safeRect.y - 10), 
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                } 
                else if (currentMode == Mode::FACE_BLUR) {
                    cv::Mat faceROI = processedFrame(safeRect);
                    cv::GaussianBlur(faceROI, faceROI, cv::Size(51, 51), 0);
                    cv::rectangle(processedFrame, safeRect, cv::Scalar(0, 0, 255), 2);
                }
            }
        }

        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = currentTime - prevTime;
        prevTime = currentTime;
        float fps = 1000.0f / duration.count();

        cv::putText(processedFrame, "FPS: " + std::to_string(static_cast<int>(fps)), 
                    cv::Point(15, 35), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 0, 255), 2);

        display.show(processedFrame);

        int key = cv::waitKey(30);
        if (key == 27) break; // ESC
        keyProc.processKey(key);
    }
    return 0;
}
