cmake_minimum_required(VERSION 3.14)


project(griffig VERSION 0.0.1 LANGUAGES CXX)


set(PYTHON3_VERSION "" CACHE STRING "Python Version")


if (PYTHON3_VERSION)
    find_package(Python3 ${PYTHON3_VERSION} EXACT REQUIRED COMPONENTS NumPy)
else()
    find_package(Python3 REQUIRED COMPONENTS NumPy)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
    add_subdirectory(pybind11)
else()
    find_package(pybind11 2.6 REQUIRED)
endif()

find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(OpenGL REQUIRED COMPONENTS EGL)


add_subdirectory(affx)


add_library(griffig src/griffig.cpp)
target_compile_features(griffig PUBLIC cxx_std_17)
target_include_directories(griffig PUBLIC include)
target_link_libraries(griffig PUBLIC affx opencv_core opencv_imgproc OpenGL::EGL OpenGL::GLU GLEW::GLEW pybind11::embed Python3::NumPy)


pybind11_add_module(_griffig src/python.cpp)
target_link_libraries(_griffig PUBLIC griffig)
