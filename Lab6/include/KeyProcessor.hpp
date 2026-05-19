#pragma once

enum class Mode {
    NORMAL,
    INVERT,
    BLUR,
    CANNY
};

class KeyProcessor {
private:
    Mode currentMode;
public:
    KeyProcessor();
    void processKey(int key);
    Mode getCurrentMode() const;
};
