cmake_minimum_required(VERSION 3.10)
project(griffig VERSION 0.0.1 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(PYTHON_EXECUTABLE python3.7)


find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python 3.7 EXACT REQUIRED COMPONENTS Interpreter Development)


pybind11_add_module(griffig SHARED src/python.cpp)
target_include_directories(griffig PUBLIC ${GLFW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR} include)
target_link_libraries(griffig PUBLIC glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${OpenCV_LIBS})
