#pragma once

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include <movex/affine.hpp>
#include <griffig/box_data.hpp>
#include <griffig/gripper.hpp>
#include <griffig/orthographic_image.hpp>
#include <griffig/pointcloud.hpp>
#include <griffig/robot_pose.hpp>

using Affine = movex::Affine;


struct OrthographicData {
    double pixel_size;
    double min_depth;
    double max_depth;
};


class Renderer {
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

    Window app {752, 480, ""};
    BoxData box_contour;

    cv::Mat color = cv::Mat::zeros(cv::Size {752, 480}, CV_16UC4);
    cv::Mat depth = cv::Mat::zeros(cv::Size {752, 480}, CV_32FC1);
    cv::Mat mask = cv::Mat::zeros(cv::Size {752, 480}, CV_8UC1);

    void draw_affines(const std::array<Affine, 4>& affines) {
        for (auto affine: affines) {
            glVertex3d(-affine.y(), affine.x(), -affine.z());
        }
    }

    void draw_cube(const Affine& pose, std::array<double, 3> size) {
        auto tl = pose * Affine(-size[0] / 2, size[1] / 2, 0.0);
        auto tr = pose * Affine(size[0] / 2, size[1] / 2, 0.0);
        auto bl = pose * Affine(-size[0] / 2, -size[1] / 2, 0.0);
        auto br = pose * Affine(size[0] / 2, -size[1] / 2, 0.0);

        auto tlu = pose * Affine(-size[0] / 2, size[1] / 2, size[2]);
        auto tru = pose * Affine(size[0] / 2, size[1] / 2, size[2]);
        auto blu = pose * Affine(-size[0] / 2, -size[1] / 2, size[2]);
        auto bru = pose * Affine(size[0] / 2, -size[1] / 2, size[2]);

        draw_affines({tl, tr, br, bl});
        draw_affines({tl, tr, tru, tlu});
        draw_affines({bl, br, bru, blu});
        draw_affines({tl, bl, blu, tlu});
        draw_affines({tr, br, bru, tru});
    }

    void draw_gripper() const {

    }

    void draw_box() const {
        glBegin(GL_QUADS);
        glColor3f(0.8, 0, 0);

        if (!box_contour.contour.size() == 4) {
            throw std::runtime_error("Box must have 4 corners currently.");
        }

        auto& c0 = box_contour.contour.at(0);
        auto& c1 = box_contour.contour.at(1);
        auto& c2 = box_contour.contour.at(2);
        auto& c3 = box_contour.contour.at(3);

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
    Renderer() { }
    Renderer(const BoxData& box_contour): box_contour(box_contour) { }

    OrthographicImage draw_box_on_image(OrthographicImage& image) {
        const size_t width = image.mat.cols;
        const size_t height = image.mat.rows;
        const double alpha = 1.0 / (2 * image.pixel_size);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_STENCIL_TEST);

        glStencilMask(0xFF);
        glStencilFunc(GL_ALWAYS, 0xFF, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        glOrtho(-alpha * width, alpha * width, -alpha * height, alpha * height, image.min_depth, image.max_depth);
        gluLookAt(-0.002, -0.0015, 0.35, -0.002, -0.0015, 0, 0, 1, 0);

        draw_box();

        glReadPixels(0, 0, mask.cols, mask.rows, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, mask.data);
        glReadPixels(0, 0, depth.cols, depth.rows, GL_DEPTH_COMPONENT, GL_FLOAT, depth.data);
        glReadPixels(0, 0, color.cols, color.rows, GL_RGBA, GL_UNSIGNED_SHORT, color.data);

        depth = (1 - depth) * 255 * 255;
        depth.convertTo(depth, CV_16U);

        const int from_to[] = {0, 3};
        mixChannels(&depth, 1, &color, 1, from_to, 1);
        color.copyTo(image.mat, mask);
        return image;
    }

    OrthographicImage draw_gripper_on_image(OrthographicImage& image, const Gripper& gripper, const RobotPose& pose) {

    }

    template<bool draw_texture>
    cv::Mat draw_pointcloud(const Pointcloud& cloud, cv::Size size, const OrthographicData& ortho, const std::array<double, 3>& camera_position) {
        color = cv::Mat::zeros(size, CV_16UC4);
        depth = cv::Mat::zeros(size, CV_32FC1);

        if (!cloud.size) {
            if constexpr (draw_texture) {
                return color;
            } else {
                return depth;
            }
        }

        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glEnable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const double alpha = 1.0 / (2 * ortho.pixel_size);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glOrtho(-alpha * size.width, alpha * size.width, -alpha * size.height, alpha * size.height, ortho.min_depth, ortho.max_depth);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        gluLookAt(camera_position[0], camera_position[1], camera_position[2], 0, 0, 1, 0, -1, 0);

        if constexpr (draw_texture) {
            const float tex_border_color[] = { 0.8f, 0.8f, 0.8f, 0.8f };

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, cloud.tex.get_gl_handle());

            glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_border_color);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F);
        }

        glEnable(GL_POINT_SMOOTH);
        glPointSize((float)size.width / 640);
        glBegin(GL_POINTS);
        {
            for (size_t i = 0; i < cloud.size; ++i) {
                glVertex3fv(*((PointTypes::XYZ *)cloud.vertices + i));

                if constexpr (draw_texture) {
                    glTexCoord2fv(*((PointTypes::UV *)cloud.tex_coords + i));
                }
            }
        }
        glEnd();

        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glPopAttrib();

        glPixelStorei(GL_PACK_ALIGNMENT, (color.step & 3) ? 1 : 4);
        glReadPixels(0, 0, depth.cols, depth.rows, GL_DEPTH_COMPONENT, GL_FLOAT, depth.data);

        depth = (1 - depth) * 255 * 255;
        depth.convertTo(depth, CV_16U);

        if constexpr (!draw_texture)  {
            cv::cvtColor(depth, depth, cv::COLOR_RGB2GRAY);
            cv::flip(depth, depth, 1);
            return depth;
        }

        glReadPixels(0, 0, color.cols, color.rows, GL_BGRA, GL_UNSIGNED_SHORT, color.data);

        const int from_to[] = {0, 3};
        mixChannels(&depth, 1, &color, 1, from_to, 1);
        cv::flip(color, color, 1);
        return color;
    }
};