/* #pragma once

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include <movex/affine.hpp>
#include <orthographic_image/orthographic_image.hpp>


class Window {
    GLFWwindow *win;

public:
    explicit Window(int width, int height, const char* title) {
        glfwInit();
        glfwWindowHint(GLFW_VISIBLE, 0);
        win = glfwCreateWindow(width, height, title, nullptr, nullptr);
        glfwMakeContextCurrent(win);
    }

    ~Window() {
        glfwDestroyWindow(win);
        glfwTerminate();
    }
};


class Checker {
    using Affine = movex::Affine;

    Window app {752, 480, ""};

    std::array<double, 3> camera_position {{0.0, 0.0, 0.0}};
    std::array<double, 3> finger_size {{0.04, 0.008, 0.12}};
    std::array<double, 3> gripper_size {{0.0, 0.0, 0.0}};

    void draw_cube(const Affine& pose, std::array<double, 3> size);

public:
    bool debug {false};

    Checker(const std::array<double, 3>& finger_size, const std::array<double, 3>& gripper_size): finger_size(finger_size), gripper_size(gripper_size) { }

    bool check_collision(const OrthographicImage& image, const Affine& pose, double stroke);
}; */
