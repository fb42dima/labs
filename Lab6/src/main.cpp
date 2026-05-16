#include "CameraProvider.hpp"
#include "KeyProcessor.hpp"
#include "FrameProcessor.hpp"
#include "Display.hpp"

int main() {
    CameraProvider camera(0);
    KeyProcessor keyProc;
    FrameProcessor frameProc;
    Display display("Lab 6 - OpenCV");

    while (true) {
        cv::Mat frame = camera.getFrame();
        if (frame.empty()) break;

        cv::Mat processedFrame = frameProc.process(frame, keyProc.getCurrentMode());
        display.show(processedFrame);

        int key = cv::waitKey(30);
        if (key == 27) break; // Вихід по клавіші ESC
        keyProc.processKey(key);
    }
    return 0;
}
