cmake_minimum_required(VERSION 3.14)


project(griffig VERSION 0.0.7 LANGUAGES CXX)
include(FetchContent)


find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(OpenGL REQUIRED COMPONENTS EGL)
find_package(Python3 REQUIRED COMPONENTS NumPy)
find_package(pybind11 REQUIRED)


add_subdirectory(affx)


add_library(griffig src/griffig.cpp)
target_compile_features(griffig PUBLIC cxx_std_17)
target_include_directories(griffig PUBLIC include)
target_link_libraries(griffig PUBLIC affx opencv_core opencv_imgproc OpenGL::EGL OpenGL::GLU GLEW::GLEW pybind11::embed Python3::NumPy)


pybind11_add_module(_griffig src/python.cpp)
target_link_libraries(_griffig PUBLIC griffig)
