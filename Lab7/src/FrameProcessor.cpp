#include "../include/FrameProcessor.hpp"
#include <vector>

cv::Mat FrameProcessor::process(const cv::Mat& input, Mode mode) {
    cv::Mat output;
    
    if (mode == Mode::INVERT) {
        cv::bitwise_not(input, output);
    } 
    else if (mode == Mode::GLITCH) {
        std::vector<cv::Mat> channels;
        cv::split(input, channels);
        
        int shift = 15;

        cv::Mat R = cv::Mat::zeros(channels[2].size(), channels[2].type());
        channels[2](cv::Rect(0, 0, input.cols - shift, input.rows)).copyTo(R(cv::Rect(shift, 0, input.cols - shift, input.rows)));
        channels[2] = R;
        
        cv::Mat B = cv::Mat::zeros(channels[0].size(), channels[0].type());
        channels[0](cv::Rect(shift, 0, input.cols - shift, input.rows)).copyTo(B(cv::Rect(0, 0, input.cols - shift, input.rows)));
        channels[0] = B;
        
        cv::merge(channels, output);
    }
    else {
        output = input.clone();
    }
    return output;
}
