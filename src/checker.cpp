/*
#include <collisions/checker.hpp>


using Affine = movex::Affine;


inline void draw_affines(const std::array<Affine, 4>& affines) {
    for (auto affine: affines) {
        glVertex3d(-affine.y(), affine.x(), -affine.z());
    }
}

void Checker::draw_cube(const Affine& pose, std::array<double, 3> size) {
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

bool Checker::check_collision(const OrthographicImage& image, const Affine& pose, double stroke) {
    int width = image.mat.cols;
    int height = image.mat.rows;
    cv::Mat result = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);

    glClearDepth(0.0);

    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();

    double plane_near = image.min_depth, plane_far = image.max_depth;
    double alpha = 1.0 / (2 * image.pixel_size);
    glOrtho(-alpha * width, alpha * width, -alpha * height, alpha * height, plane_near, plane_far);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    gluLookAt(camera_position[0], camera_position[1], camera_position[2], 0, 0, 1, 0, -1, 0);

    glBegin(GL_QUADS);

    // Render fingers
    draw_cube(image.pose.inverse() * pose * Affine(0.0, stroke / 2, 0.0), finger_size);
    draw_cube(image.pose.inverse() * pose * Affine(0.0, -stroke / 2, 0.0), finger_size);

    // Render gripper
    if (gripper_size[0] > 0.0) {
        draw_cube(image.pose.inverse() * pose * Affine(0.0, 0.0, finger_size[2]), gripper_size);
    }

    glEnd();

    glPixelStorei(GL_PACK_ALIGNMENT, (result.step & 3) ? 1 : 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, result.step / result.elemSize());
    glReadPixels(0, 0, result.cols, result.rows, GL_DEPTH_COMPONENT, GL_FLOAT, result.data);

    result = plane_near + ( plane_far - plane_near ) * result;
    result = (result - image.max_depth) / (image.min_depth - image.max_depth);
    result = cv::max(cv::min(result, 1.0), 0.0);

    cv::Mat image_32f = cv::Mat::zeros(cv::Size(width, height), CV_32F);
    image.mat.convertTo(image_32f, CV_32F);
    image_32f /= 255 * 255;

    bool no_collision = cv::checkRange(result - image_32f, true, 0, 0.0, 1.01);

    if (debug) {
        cv::imwrite("/tmp/test.png", result * 255);
        cv::imwrite("/tmp/test2.png", image.mat);

        std::cout << "no_collision: " << no_collision << std::endl;
    }

    return no_collision;
}
*/