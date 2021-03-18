cmake_minimum_required(VERSION 3.10)
project(griffig VERSION 0.0.1 LANGUAGES CXX)


set(PYTHON_EXECUTABLE python3.7)


find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)
find_package(OpenCV 4.5 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Python 3.7 EXACT REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 2.6 REQUIRED)


add_library(griffig SHARED src/griffig.cpp)
target_compile_features(griffig PUBLIC cxx_std_17)
target_include_directories(griffig PUBLIC ${GLFW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR} include)
target_link_libraries(griffig PUBLIC frankx::movex Eigen3::Eigen glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${OpenCV_LIBS})


pybind11_add_module(_griffig SHARED src/python.cpp)
target_link_libraries(_griffig PUBLIC griffig)
