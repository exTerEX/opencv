add_executable(fl main.cpp)
set_property(TARGET fl PROPERTY CXX_STANDARD 17)
target_link_libraries(fl ${OpenCV_LIBS})
