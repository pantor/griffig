#pragma once

#include <limits>
#include <optional>

#include <affx/affine.hpp>


struct Gripper {
    //! Transformation from end-effector to the fingertips
    affx::Affine robot_to_tip {};

    //! The min. and max. gripper width (distance between fingers)
    std::optional<std::array<double, 2>> width_interval;
    double min_stroke {0.0}, max_stroke {std::numeric_limits<double>::infinity()};

    //! A box around each finger
    std::optional<std::array<double, 3>> finger_size;
    double width, height;

    explicit Gripper(const affx::Affine& robot_to_tip, const std::optional<std::array<double, 2>>& width_interval, const std::optional<std::array<double, 3>>& finger_size): robot_to_tip(robot_to_tip), width_interval(width_interval), finger_size(finger_size) { }
    explicit Gripper(double min_stroke = 0.0, double max_stroke = std::numeric_limits<double>::infinity(), double width = 0.0, double height = 0.0, affx::Affine robot_to_tip = affx::Affine()):
        min_stroke(min_stroke), max_stroke(max_stroke), width(width), height(height), robot_to_tip(robot_to_tip) { }
};
