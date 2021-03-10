#pragma once

#include <array>
#include <vector>

#include <movex/affine.hpp>


struct BoxData {
    std::vector<std::array<double, 3>> contour;
    movex::Affine pose;

    explicit BoxData() { }
    explicit BoxData(const std::vector<std::array<double, 3>>& contour, const movex::Affine& pose): contour(contour), pose(pose) { }

    explicit BoxData(const std::array<double, 3>& center, const std::array<double, 3>& size, const movex::Affine& pose): pose(pose) {
        contour = {
            {center[0] + size[0] / 2, center[1] + size[1] / 2, size[2]},
            {center[0] + size[0] / 2, center[1] - size[1] / 2, size[2]},
            {center[0] - size[0] / 2, center[1] - size[1] / 2, size[2]},
            {center[0] - size[0] / 2, center[1] + size[1] / 2, size[2]},
        };
    }
};
