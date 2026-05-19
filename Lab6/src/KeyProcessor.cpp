#include "../include/KeyProcessor.hpp"

KeyProcessor::KeyProcessor() : currentMode(Mode::NORMAL) {}

void KeyProcessor::processKey(int key) {
    // Змінюємо режим за допомогою клавіш 1, 2, 3, 4
    if (key == '1') {
        currentMode = Mode::NORMAL;
    } else if (key == '2') {
        currentMode = Mode::INVERT;
    } else if (key == '3') {
        currentMode = Mode::BLUR;
    } else if (key == '4') {
        currentMode = Mode::CANNY;
    }
}

Mode KeyProcessor::getCurrentMode() const {
    return currentMode;
}
