cmake_minimum_required(VERSION 3.14)


project(griffig VERSION 0.0.1 LANGUAGES CXX)


find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS EGL)
find_package(Python3 REQUIRED COMPONENTS NumPy)

find_package(Eigen3)
if(NOT Eigen3_FOUND)
    add_subdirectory(eigen)
endif()

find_package(pybind11 2.6)
if(NOT pybind11_FOUND)
    add_subdirectory(pybind11)
endif()

find_package(OpenCV COMPONENTS core imgproc)
if(NOT OpenCV_FOUND)
    add_subdirectory(opencv)
endif()



add_subdirectory(affx)


add_library(griffig src/griffig.cpp)
target_compile_features(griffig PUBLIC cxx_std_17)
target_include_directories(griffig PUBLIC include)
target_link_libraries(griffig PUBLIC affx opencv_core opencv_imgproc OpenGL::EGL OpenGL::GLU GLEW::GLEW pybind11::embed Python3::NumPy)


pybind11_add_module(_griffig src/python.cpp)
target_link_libraries(_griffig PUBLIC griffig)
