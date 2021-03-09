#pragma once

#include <array>
#include <vector>

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>


struct BoxContour {
    std::vector<std::array<double, 3>> corners;

    explicit BoxContour(const std::vector<std::array<double, 3>>& corners): corners(corners) { }
    explicit BoxContour(const std::array<double, 3>& center, const std::array<double, 3>& size) {
        corners = {
            {center[0] + size[0] / 2, center[1] + size[1] / 2, size[2]},
            {center[0] + size[0] / 2, center[1] - size[1] / 2, size[2]},
            {center[0] - size[0] / 2, center[1] - size[1] / 2, size[2]},
            {center[0] - size[0] / 2, center[1] + size[1] / 2, size[2]},
        };
    }
};


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


class Griffig {
    Window app {752, 480, ""};
    BoxContour box_contour;

    cv::Mat box_rendered = cv::Mat::zeros(cv::Size(752, 480), CV_16UC4);
    cv::Mat depth = cv::Mat::zeros(cv::Size(752, 480), CV_32FC1);
    cv::Mat mask = cv::Mat::zeros(cv::Size(752, 480), CV_8UC1);

    void opengl_draw_box() const {
        glBegin(GL_QUADS);
        glColor3f(1, 0, 0);

        if (!box_contour.corners.size() == 4) {
            throw std::runtime_error("Box must have 4 corners currently.");
        }

        auto& c0 = box_contour.corners.at(0);
        auto& c1 = box_contour.corners.at(1);
        auto& c2 = box_contour.corners.at(2);
        auto& c3 = box_contour.corners.at(3);

        // Render contour
        glVertex3d(c2[0], -1, c2[2]);
        glVertex3d(c2[0], c2[1], c2[2]);
        glVertex3d(c1[0], c1[1], c1[2]);
        glVertex3d(c1[0], -1, c1[2]);

        glVertex3d(c3[0], 1, c3[2]);
        glVertex3d(c3[0], c3[1], c3[2]);
        glVertex3d(c0[0], c0[1], c0[2]);
        glVertex3d(c0[0], 1, c0[2]);

        glVertex3d(1, -1, c1[2]);
        glVertex3d(c1[0], -1, c1[2]);
        glVertex3d(c0[0], 1, c0[2]);
        glVertex3d(1, 1, c0[2]);

        glVertex3d(-1, -1, c2[2]);
        glVertex3d(c2[0], -1, c2[2]);
        glVertex3d(c3[0], 1, c3[2]);
        glVertex3d(-1, 1, c3[2]);
        glEnd();
    }

public:
    Griffig(const BoxContour& box_contour): box_contour(box_contour) {
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_STENCIL_TEST);

        glStencilMask(0xFF);
        glStencilFunc(GL_ALWAYS, 0xFF, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    }

    cv::Mat draw_box_on_image(cv::Mat& image) {
        const size_t width = image.cols;
        const size_t height = image.rows;

        const double plane_near = 0.22, plane_far = 0.41, pixel_size = 2000.0;
        const double alpha = 1.0 / (2 * pixel_size);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        glOrtho(-alpha * width, alpha * width, -alpha * height, alpha * height, plane_near, plane_far);
        gluLookAt(-0.002, -0.0015, 0.35, -0.002, -0.0015, 0, 0, 1, 0);

        opengl_draw_box();

        glReadPixels(0, 0, mask.cols, mask.rows, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, mask.data);
        glReadPixels(0, 0, depth.cols, depth.rows, GL_DEPTH_COMPONENT, GL_FLOAT, depth.data);
        glReadPixels(0, 0, box_rendered.cols, box_rendered.rows, GL_RGBA, GL_UNSIGNED_SHORT, box_rendered.data);

        depth = (1 - depth) * 255 * 255;
        depth.convertTo(depth, CV_16U);

        const int from_to[] = {0, 3};
        mixChannels(&depth, 1, &box_rendered, 1, from_to, 1);
        box_rendered.copyTo(image, mask);
        return image;
    }
};
