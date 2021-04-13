#pragma once

#include <limits>
#include <optional>
#include <vector>

#include <affx/affine.hpp>


struct Gripper {
    //! The min. and max. gripper width (distance between fingers)
    double min_stroke {0.0}, max_stroke {std::numeric_limits<double>::infinity()};

    //! A box around each finger
    double finger_width, finger_extent, finger_height;

    //! Offset transformation (local in the grippers reference frame)
    affx::Affine offset {};

    explicit Gripper(double min_stroke = 0.0, double max_stroke = std::numeric_limits<double>::infinity(), double finger_width = 0.0, double finger_extent = 0.0, double finger_height = 0.0, affx::Affine offset = affx::Affine()):
        min_stroke(min_stroke), max_stroke(max_stroke), finger_width(finger_width), finger_extent(finger_extent), finger_height(finger_height), offset(offset) { }

    std::vector<bool> consider_indices(const std::vector<double>& gripper_widths) const {
        std::vector<bool> result (gripper_widths.size());
        for (size_t i = 0; i < gripper_widths.size(); ++i) {
            result[i] = (min_stroke <= gripper_widths[i] && gripper_widths[i] <= max_stroke);
        }
        return result;
    }
};
