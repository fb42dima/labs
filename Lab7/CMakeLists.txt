cmake_minimum_required(VERSION 3.10)
project(Lab7)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp")
add_executable(Lab7 ${SOURCES})
target_link_libraries(Lab7 ${OpenCV_LIBS})
