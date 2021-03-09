/* #pragma once

class Checker {
    std::array<double, 3> camera_position {{0.0, 0.0, 0.0}};
    std::array<double, 3> finger_size {{0.04, 0.008, 0.12}};
    std::array<double, 3> gripper_size {{0.0, 0.0, 0.0}};

    void draw_cube(const Affine& pose, std::array<double, 3> size);

    Checker(const std::array<double, 3>& finger_size, const std::array<double, 3>& gripper_size): finger_size(finger_size), gripper_size(gripper_size) { }

    bool check_collision(const OrthographicImage& image, const Affine& pose, double stroke);
}; */
