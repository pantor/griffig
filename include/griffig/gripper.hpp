#pragma once

#include <optional>

#include <movex/affine.hpp>


struct Gripper {
    //! Transformation from end-effector to the fingertips
    std::optional<movex::Affine> robot_to_tip;

    //! The min. and max. gripper width (distance between fingers)
    std::optional<std::array<double, 2>> width_interval;

    //! A box around each finger
    std::optional<std::array<double, 3>> finger_size;

    explicit Gripper(const std::optional<movex::Affine>& robot_to_tip, const std::optional<std::array<double, 2>>& width_interval, const std::optional<std::array<double, 3>>& finger_size): robot_to_tip(robot_to_tip), width_interval(width_interval), finger_size(finger_size) { }
};
