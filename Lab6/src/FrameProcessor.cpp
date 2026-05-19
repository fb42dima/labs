#include "../include/FrameProcessor.hpp"

cv::Mat FrameProcessor::process(const cv::Mat& input, Mode mode) {
    cv::Mat output;
    
    switch(mode) {
        case Mode::NORMAL:
            output = input.clone();
            break;
        case Mode::INVERT:
            cv::bitwise_not(input, output);
            break;
        case Mode::BLUR:
            // Розмиття з ядром 15x15
            cv::GaussianBlur(input, output, cv::Size(15, 15), 0);
            break;
        case Mode::CANNY:
            // Canny вимагає чорно-білого зображення
            cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
            cv::Canny(output, output, 50, 150);
            break;
        default:
            output = input.clone();
            break;
    }
    return output;
}
