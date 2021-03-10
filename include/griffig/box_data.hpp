#pragma once

#include <array>
#include <vector>
#include <optional>

#include <movex/affine.hpp>


struct BoxData {
    std::vector<std::array<double, 3>> contour;
    std::optional<movex::Affine> pose;

    explicit BoxData() { }
    explicit BoxData(const std::vector<std::array<double, 3>>& contour, const std::optional<movex::Affine>& pose = std::nullopt): contour(contour), pose(pose) { }
    explicit BoxData(const std::array<double, 3>& center, const std::array<double, 3>& size, const std::optional<movex::Affine>& pose = std::nullopt): pose(pose) {
        contour = {
            {center[0] + size[0] / 2, center[1] + size[1] / 2, size[2]},
            {center[0] + size[0] / 2, center[1] - size[1] / 2, size[2]},
            {center[0] - size[0] / 2, center[1] - size[1] / 2, size[2]},
            {center[0] - size[0] / 2, center[1] + size[1] / 2, size[2]},
        };
    }
};
