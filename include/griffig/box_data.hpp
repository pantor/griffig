#pragma once

#include <array>
#include <vector>
#include <optional>

#include <affx/affine.hpp>


struct BoxData {
    using Affine = affx::Affine;

    //! The contour of the box
    std::vector<std::array<double, 3>> contour;

    //! An optional pose of the center
    std::optional<Affine> pose;

    explicit BoxData() { }
    explicit BoxData(const std::vector<std::array<double, 3>>& contour, const std::optional<Affine>& pose = std::nullopt): contour(contour), pose(pose) { }
    explicit BoxData(const std::array<double, 3>& center, const std::array<double, 3>& size, const std::optional<Affine>& pose = std::nullopt): pose(pose) {
        contour = {
            {center[0] + size[0] / 2, center[1] + size[1] / 2, size[2]},
            {center[0] + size[0] / 2, center[1] - size[1] / 2, size[2]},
            {center[0] - size[0] / 2, center[1] - size[1] / 2, size[2]},
            {center[0] - size[0] / 2, center[1] + size[1] / 2, size[2]},
        };
    }
};
