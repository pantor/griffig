#pragma once

#include <array>
#include <optional>
#include <vector>

#include <opencv2/opencv.hpp>

#include <griffig/robot_pose.hpp>


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
        return {
            2*std::max(std::abs(rect.y), std::abs(rect.y + rect.height)) + offset,
            2*std::max(std::abs(rect.x), std::abs(rect.x + rect.width)) + offset
        };
    }

    bool is_pose_inside(const RobotPose& pose) const {
        double half_stroke = 0.5 * (pose.d + 0.002);  // [m]
        auto gripper_b1_trans = (pose * Affine(0.0, half_stroke, 0.0)).translation();
        auto gripper_b2_trans = (pose * Affine(0.0, -half_stroke, 0.0)).translation();
        cv::Point2f gripper_b1 {(float)gripper_b1_trans[0], (float)gripper_b1_trans[1]};
        cv::Point2f gripper_b2 {(float)gripper_b2_trans[0], (float)gripper_b2_trans[1]};

        std::vector<cv::Point2f> check_contour;
        for (auto c: contour) {
            check_contour.emplace_back(cv::Point2f(c[0], c[1]));
        }

        bool jaw1_inside_box = cv::pointPolygonTest(check_contour, gripper_b1, false) >= 0;
        bool jaw2_inside_box = cv::pointPolygonTest(check_contour, gripper_b2, false) >= 0;

        bool start_point_inside_box = true;
        if (!std::isnan(pose.z())) {
            auto helper_pose_trans = (pose * Affine(0.0, 0.0, 0.16)).translation();
            cv::Point2f helper_pose {(float)helper_pose_trans[0], (float)helper_pose_trans[1]};
            start_point_inside_box = cv::pointPolygonTest(check_contour, helper_pose, false) >= 0;
        }

        return jaw1_inside_box && jaw2_inside_box && start_point_inside_box;
    }
};
