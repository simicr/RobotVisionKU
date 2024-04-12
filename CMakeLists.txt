cmake_minimum_required(VERSION 3.12)
project(opencv_prvi)

find_package(OpenCV REQUIRED)
add_executable(out code/assign1_task1.cpp code/assign1_task2.cpp include/assign1_task2.hpp)
target_link_libraries(out ${OpenCV_LIBS})
