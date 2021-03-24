cmake_minimum_required(VERSION 3.12)
project(griffig VERSION 0.0.1 LANGUAGES CXX)


find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Python 3.6 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 2.6 REQUIRED)


add_subdirectory(affx)


add_library(griffig src/griffig.cpp)
target_compile_features(griffig PUBLIC cxx_std_17)
target_include_directories(griffig PUBLIC ${GLFW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR} include)
target_link_libraries(griffig PUBLIC affx glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${OpenCV_LIBS})


pybind11_add_module(_griffig src/python.cpp)
target_link_libraries(_griffig PUBLIC griffig)
