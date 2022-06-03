find_package(OpenCV REQUIRED core objdetect highgui imgproc)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(face-detection)
add_subdirectory(find-lane)
add_subdirectory(number-plate-recognition)
