#pragma once

#include <array>
#include <optional>
#include <iostream>

#include <GL/glew.h>
#include <EGL/egl.h>
#include <opencv2/opencv.hpp>

// For Eigen3 compability
#ifdef Success
  #undef Success
#endif

#include <affx/affine.hpp>
#include <griffig/box_data.hpp>
#include <griffig/gripper.hpp>
#include <griffig/orthographic_image.hpp>
#include <griffig/pointcloud.hpp>
#include <griffig/robot_pose.hpp>


class Renderer {
    using Affine = affx::Affine;

    EGLDisplay eglDpy;
    GLuint fb, color, depth, stencil;

    void init_egl(int width, int height) {
        eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (eglDpy == EGL_NO_DISPLAY) {
            std::cout << "display: " << eglGetError() << std::endl;
            exit(EXIT_FAILURE);
        }

        EGLint major, minor;
        if (!eglInitialize(eglDpy, &major, &minor)) {
            std::cout << "init: " << eglGetError() << std::endl;
            exit(EXIT_FAILURE);
        }

        EGLint const configAttribs[] = {
            // EGL_RED_SIZE, 1,
            // EGL_GREEN_SIZE, 1,
            // EGL_BLUE_SIZE, 1,
            EGL_NONE
        };

        EGLint numConfigs;
        EGLConfig eglCfg;
        if (!eglChooseConfig(eglDpy, configAttribs, NULL, 0, &numConfigs)) {
            std::cout << "choose config: " << eglGetError() << std::endl;
            exit(EXIT_FAILURE);
        }

        // std::cout << "numConfigs: " << numConfigs << std::endl;
        eglBindAPI(EGL_OPENGL_API);

        EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
        if (eglCtx == NULL) {
            std::cout << "create context: " << eglGetError() << std::endl;
            exit(EXIT_FAILURE);
        }

        if (!eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx)) {
            std::cout << "make current: " << eglGetError() << std::endl;
            exit(EXIT_FAILURE);
        }

        GLenum glewinit = glewInit();
        if (glewinit != GLEW_OK) {
            std::cout << "glew init: " << glewinit << std::endl;
            exit(EXIT_FAILURE);
        }

        glGenFramebuffers(1, &fb);

        glGenTextures(1, &color);
        glGenRenderbuffers(1, &depth);
        glGenRenderbuffers(1, &stencil);

        glBindFramebuffer(GL_FRAMEBUFFER, fb);

        glBindTexture(GL_TEXTURE_2D, color);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16, width, height, 0, GL_RGBA, GL_UNSIGNED_SHORT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth);
    }

    void close_egl() {
        eglTerminate(eglDpy);
    }

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

    void draw_box(const BoxData& box_contour, const cv::Mat& image) const {
        if (box_contour.contour.size() != 4) {
            throw std::runtime_error("Box must have 4 corners currently.");
        }

        std::array<GLdouble, 16> projection, modelview;
        std::array<GLint, 4> viewport;
        glGetDoublev(GL_PROJECTION_MATRIX, projection.data());
        glGetDoublev(GL_MODELVIEW_MATRIX, modelview.data());
        glGetIntegerv(GL_VIEWPORT, viewport.data());

        std::array<GLdouble, 3> coord;
        std::vector<cv::Vec4f> colors;
        colors.resize(4);

        for (size_t i = 0; i < 4; ++i) {
            auto& c = box_contour.contour.at(i);
            gluProject(c[1], c[0], c[2], modelview.data(), projection.data(), viewport.data(), coord.data(), coord.data() + 1, coord.data() + 2);
            colors[i] = ((cv::Vec4f)image.at<cv::Vec4w>(coord[1], coord[0])) / (255 * 255);
        }

        float r_mean {0}, g_mean {0}, b_mean {0};
        for (auto c: colors) {
            r_mean += c[0] / colors.size();
            g_mean += c[1] / colors.size();
            b_mean += c[2] / colors.size();
        }

        auto& c0 = box_contour.contour.at(0);
        auto& c1 = box_contour.contour.at(1);
        auto& c2 = box_contour.contour.at(2);
        auto& c3 = box_contour.contour.at(3);

        glColor3f(r_mean, g_mean, b_mean);
        glBegin(GL_QUADS);
            glVertex3d(c0[1], 1.0, c0[2]);
            glVertex3d(c0[1], c0[0], c0[2]);
            glVertex3d(c1[1], c1[0], c1[2]);
            glVertex3d(c1[1], 1.0, c1[2]);

            glVertex3d(-1.0, c1[0], c1[2]);
            glVertex3d(c1[1], c1[0], c1[2]);
            glVertex3d(c2[1], c2[0], c2[2]);
            glVertex3d(-1.0, c2[0], c2[2]);

            glVertex3d(c2[1], -1.0, c2[2]);
            glVertex3d(c2[1], c2[0], c2[2]);
            glVertex3d(c3[1], c3[0], c3[2]);
            glVertex3d(c3[1], -1.0, c3[2]);

            glVertex3d(1.0, c3[0], c3[2]);
            glVertex3d(c3[1], c3[0], c3[2]);
            glVertex3d(c0[1], c0[0], c0[2]);
            glVertex3d(1.0, c0[0], c0[2]);
        glEnd();

        glBegin(GL_TRIANGLES);
            glVertex3d(c0[1], 1.0, c0[2]);
            glVertex3d(c0[1], c1[0], c1[2]);
            glVertex3d(-1.0, c1[0], c1[2]);

            glVertex3d(c1[1], -1.0, c1[2]);
            glVertex3d(c1[1], c2[0], c2[2]);
            glVertex3d(-1.0, c2[0], c2[2]);

            glVertex3d(c2[1], -1.0, c2[2]);
            glVertex3d(c2[1], c3[0], c3[2]);
            glVertex3d(1.0, c3[0], c3[2]);

            glVertex3d(c3[1], 1.0, c3[2]);
            glVertex3d(c3[1], c0[0], c0[2]);
            glVertex3d(1.0, c0[0], c0[2]);
        glEnd();
    }

    // std::unique_ptr<Window> app;

public:
    size_t width, height;
    double pixel_size;
    double typical_camera_distance;
    double depth_diff;

    std::optional<BoxData> box_contour;
    std::array<double, 3> camera_position {0.0, 0.0, 0.0};

    explicit Renderer() {
        init_egl(width, height);
    }

    explicit Renderer(const std::array<double, 2>& size): width(size[0]), height(size[1]) {
        init_egl(width, height);
    }

    explicit Renderer(const std::array<double, 2>& size, double pixel_size, double depth_diff, const std::optional<BoxData>& box_contour): width(size[0]), height(size[1]), pixel_size(pixel_size), depth_diff(depth_diff), box_contour(box_contour) {
        init_egl(width, height);
    }

    explicit Renderer(const std::array<double, 2>& size, const std::array<double, 3>& position): width(size[0]), height(size[1]), camera_position(position) {
        init_egl(width, height);
    }

    explicit Renderer(const BoxData& box_data, double typical_camera_distance, double pixel_size, double depth_diff): box_contour(box_data), typical_camera_distance(typical_camera_distance), pixel_size(pixel_size), depth_diff(depth_diff) {
        const auto size = box_data.get_rect(pixel_size, 5);
        width = size[0];
        height = size[1];
        init_egl(width, height);
    }

    ~Renderer() {
        close_egl();
    }

    OrthographicImage draw_box_on_image(OrthographicImage& image) {
        if (width != image.mat.cols || height != image.mat.rows) {
            throw std::runtime_error("Renderer size mismatch.");
        }

        const cv::Size size {(int)width, (int)height};

        cv::Mat color = cv::Mat::zeros(size, CV_16UC4);
        cv::Mat depth = cv::Mat::zeros(size, CV_32FC1);
        cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);

        glEnable(GL_TEXTURE_2D);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_STENCIL_TEST);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);

        glStencilMask(0xFF);
        glStencilFunc(GL_ALWAYS, 0xFF, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

        glViewport(0, 0, width, height);

        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        const double alpha = 1.0 / (2 * image.pixel_size);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glOrtho(alpha * width, -alpha * width, -alpha * height, alpha * height, image.min_depth, image.max_depth);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        gluLookAt(camera_position[0], camera_position[1], camera_position[2], camera_position[0], camera_position[1], 1.0, 0.0, -1.0, 0.0);

        if (box_contour) {
            draw_box(*box_contour, image.mat);
        }

        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glPopAttrib();

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
    cv::Mat draw_pointcloud(const Pointcloud& cloud, double pixel_density, double min_depth, double max_depth, const std::array<double, 3>& camera_position) {
        cv::Size size {(int)width, (int)height};
        cv::Mat color = cv::Mat::zeros(size, CV_16UC4);
        cv::Mat depth = cv::Mat::zeros(size, CV_32FC1);

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

        const double alpha = 1.0 / (2 * pixel_density);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glOrtho(alpha * size.width, -alpha * size.width, -alpha * size.height, alpha * size.height, min_depth, max_depth);

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
            return depth;
        }

        glReadPixels(0, 0, color.cols, color.rows, GL_BGRA, GL_UNSIGNED_SHORT, color.data);

        const int from_to[] = {0, 3};
        mixChannels(&depth, 1, &color, 1, from_to, 1);
        return color;
    }

    template<bool draw_texture>
    OrthographicImage render_pointcloud_easy(const Pointcloud& cloud) {
        const double min_depth = typical_camera_distance - depth_diff;
        const double max_depth = typical_camera_distance;

        return render_pointcloud<draw_texture>(cloud, pixel_size, min_depth, max_depth);
    }

    template<bool draw_texture>
    OrthographicImage render_pointcloud(const Pointcloud& cloud, double pixel_density, double min_depth, double max_depth) {
        const cv::Size size {(int)width, (int)height};
        cv::Mat color = cv::Mat::zeros(size, CV_16UC4);
        cv::Mat depth = cv::Mat::zeros(size, CV_32FC1);

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
        glOrtho(alpha * width, -alpha * width, -alpha * height, alpha * height, min_depth, max_depth);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        gluLookAt(camera_position[0], camera_position[1], camera_position[2], camera_position[0], camera_position[1], 1.0, 0.0, -1.0, 0.0);

        glEnable(GL_POINT_SMOOTH);
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
        return OrthographicImage(color, pixel_density, min_depth, max_depth);
    }
};