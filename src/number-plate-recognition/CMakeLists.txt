find_package(PkgConfig REQUIRED)
pkg_search_module(Tesseract 4.1 REQUIRED tesseract)

include_directories(${Tesseract_INCLUDE_DIRS})

add_executable(npr main.cpp)
set_property(TARGET npr PROPERTY CXX_STANDARD 17)
target_link_libraries(npr ${OpenCV_LIBS} ${Tesseract_LIBRARIES})
