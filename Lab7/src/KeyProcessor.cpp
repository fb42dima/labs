#include "../include/KeyProcessor.hpp"

KeyProcessor::KeyProcessor() : currentMode(Mode::NORMAL) {}

void KeyProcessor::processKey(int key) {
    if (key == '1') currentMode = Mode::NORMAL;
    else if (key == '2') currentMode = Mode::INVERT;
    else if (key == '3') currentMode = Mode::GLITCH;
    else if (key == '4') currentMode = Mode::FACE_DETECT;
    else if (key == '5') currentMode = Mode::FACE_BLUR;
}

Mode KeyProcessor::getCurrentMode() const {
    return currentMode;
}
