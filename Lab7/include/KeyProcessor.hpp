#pragma once

enum class Mode {
    NORMAL,
    INVERT,
    GLITCH,
    FACE_DETECT,
    FACE_BLUR
};

class KeyProcessor {
private:
    Mode currentMode;
public:
    KeyProcessor();
    void processKey(int key);
    Mode getCurrentMode() const;
};
