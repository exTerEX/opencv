add_executable(fd main.cpp)
set_property(TARGET fd PROPERTY CXX_STANDARD 17)
target_link_libraries(fd ${OpenCV_LIBS})
