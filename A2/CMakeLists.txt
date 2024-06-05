cmake_minimum_required(VERSION 3.12)
project(opencv_drugi)

find_package(OpenCV REQUIRED)
add_executable(task1 code/assign2_task1.cpp)
add_executable(task2 code/assign2_task2.cpp)

target_link_libraries(task1 ${OpenCV_LIBS})
target_link_libraries(task2 ${OpenCV_LIBS})


