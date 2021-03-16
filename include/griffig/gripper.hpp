#pragma once

#include <movex/Affine.hpp>


struct Gripper {
    //! Transformation from end-effector to the fingertips
    Affine robot_to_tip;

    //! The min. and max. gripper width (distance between fingers)
    std::array<double, 2> width_interval;

    //! A box around each finger
    std::array<double, 3> finger_size;
};
