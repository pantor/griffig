#pragma once

#include <array>
#include <optional>
#include <iostream>

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include <affx/affine.hpp>
#include <griffig/box_data.hpp>
#include <griffig/gripper.hpp>
#include <griffig/orthographic_image.hpp>
#include <griffig/pointcloud.hpp>
#include <griffig/robot_pose.hpp>


struct OrthographicData {
    double pixel_density;
    double min_depth;
    double max_depth;

    OrthographicData(double pixel_density, double min_depth, double max_depth): pixel_density(pixel_density), min_depth(min_depth), max_depth(max_depth) { }
};


class Renderer {
    using Affine = affx::Affine;

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

    const int width {752}, height {480};
    Window app {width, height, ""};
    std::optional<BoxData> box_contour;

    double pixel_size;
    double depth_diff;

    cv::Mat color = cv::Mat::zeros(cv::Size {width, height}, CV_16UC4);
    cv::Mat depth = cv::Mat::zeros(cv::Size {width, height}, CV_32FC1);
    cv::Mat mask = cv::Mat::zeros(cv::Size {width, height}, CV_8UC1);

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

        if (!box_contour || !box_contour->contour.size() == 4) {
            throw std::runtime_error("Box must have 4 corners currently.");
        }

        auto& c0 = box_contour->contour.at(0);
        auto& c1 = box_contour->contour.at(1);
        auto& c2 = box_contour->contour.at(2);
        auto& c3 = box_contour->contour.at(3);

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
    std::array<double, 3> camera_position;
    double point_size {2.0}; // Point size for OpenGL rendering


    Renderer() { }
    Renderer(double pixel_size, double depth_diff, const std::optional<BoxData>& box_contour): pixel_size(pixel_size), depth_diff(depth_diff), box_contour(box_contour) { }
    Renderer(const std::array<double, 3>& position, double point_size): camera_position(position), point_size(point_size) { }

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
    cv::Mat draw_pointcloud(const Pointcloud& cloud, const std::array<size_t, 2>& size, const OrthographicData& ortho, const std::array<double, 3>& camera_position) {
        cv::Size cv_size {size[0], size[1]};
        color = cv::Mat::zeros(cv_size, CV_16UC4);
        depth = cv::Mat::zeros(cv_size, CV_32FC1);

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

        const double alpha = 1.0 / (2 * ortho.pixel_density);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glOrtho(-alpha * cv_size.width, alpha * cv_size.width, -alpha * cv_size.height, alpha * cv_size.height, ortho.min_depth, ortho.max_depth);

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
        glPointSize((float)cv_size.width / 640);
        glBegin(GL_POINTS);
        {
            for (size_t i = 0; i < cloud.size; ++i) {
                glVertex3fv(&((PointTypes::XYZ *)cloud.vertices + i)->x);

                if constexpr (draw_texture) {
                    glTexCoord2fv(&((PointTypes::UV *)cloud.tex_coords + i)->u);
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

    OrthographicImage render_pointcloud(const Pointcloud& cloud, double pixel_density, double min_depth, double max_depth) {
        cv::Mat color = cv::Mat::zeros(cv::Size(width, height), CV_16UC4);
        cv::Mat depth = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);

        if (cloud.size == 0) {
            return OrthographicImage(color, pixel_density, min_depth, max_depth);
        }

        glLoadIdentity();
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glEnable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const double alpha = 1.0 / (2 * pixel_density);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glOrtho(-alpha * width, alpha * width, -alpha * height, alpha * height, min_depth, max_depth);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        gluLookAt(camera_position[0], camera_position[1], camera_position[2], camera_position[0], camera_position[1], 1.0, 0.0, -1.0, 0.0);

        glEnable(GL_POINT_SMOOTH);
        glPointSize(point_size);
        glBegin(GL_POINTS);
        {
            for (size_t i = 0; i < cloud.size; ++i) {
                glVertex3fv(&((PointTypes::XYZWRGBA *)cloud.vertices + i)->x);
                glColor3ubv(&((PointTypes::XYZWRGBA *)cloud.vertices + i)->r);
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

        glReadPixels(0, 0, color.cols, color.rows, GL_RGBA, GL_UNSIGNED_SHORT, color.data);

        const int from_to[] = {0, 3};
        mixChannels(&depth, 1, &color, 1, from_to, 1);
        cv::flip(color, color, 1);
        return OrthographicImage(color, pixel_density, min_depth, max_depth);
    }
};