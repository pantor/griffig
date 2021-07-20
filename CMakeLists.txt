cmake_minimum_required(VERSION 3.14)


project(griffig VERSION 0.0.1 LANGUAGES CXX)
include(FetchContent)


find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS EGL)
find_package(Python3 REQUIRED COMPONENTS NumPy)


find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    FetchContent_Declare(eigen GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git GIT_TAG 3.3.9)
    FetchContent_MakeAvailable(eigen)
endif()


find_package(pybind11 2.6 QUIET)
if(NOT pybind11_FOUND)
    FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.6.2)
    FetchContent_MakeAvailable(pybind11)
endif()


find_package(OpenCV QUIET COMPONENTS core imgproc)
if(NOT OpenCV_FOUND)
    FetchContent_Declare(opencv GIT_REPOSITORY https://github.com/opencv/opencv.git GIT_TAG 4.5.2)
    FetchContent_MakeAvailable(opencv)
endif()


add_subdirectory(affx)


add_library(griffig src/griffig.cpp)
target_compile_features(griffig PUBLIC cxx_std_17)
target_include_directories(griffig PUBLIC include)
target_link_libraries(griffig PUBLIC affx opencv_core opencv_imgproc OpenGL::EGL OpenGL::GLU GLEW::GLEW pybind11::embed Python3::NumPy)


pybind11_add_module(_griffig src/python.cpp)
target_link_libraries(_griffig PUBLIC griffig)
