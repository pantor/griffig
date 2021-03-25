#pragma once

#include <array>
#include <optional>
#include <vector>

#include <opencv2/opencv.hpp>

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

    std::array<int, 2> get_rect(float pixel_size, int offset) const {
        std::vector<cv::Point2f> cont;
        for (auto e: contour) {
            cont.push_back({(float)e[0] * pixel_size, (float)e[1] * pixel_size});
        }
        auto rect = cv::boundingRect(cont);
        return {rect.height + offset, rect.width + offset};
    }
};
