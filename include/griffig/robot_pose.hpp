#pragma once

#include <affx/affine.hpp>


struct RobotPose: affx::Affine {
    double d;

    explicit RobotPose(double x, double y, double z, double a, double b, double c, double d): affx::Affine(x, y, z, a, b, c), d(d) { }
    explicit RobotPose(double x, double y, double z, double q_w, double q_x, double q_y, double q_z, double d): affx::Affine(x, y, z, q_w, q_x, q_y, q_z), d(d) { }
    explicit RobotPose(const affx::Affine& affine, double d): affx::Affine(affine), d(d) { }

    RobotPose operator *(const affx::Affine& a) const {
        affx::Affine result = Affine::operator *(a);
        return RobotPose(result, d);
    }

    friend RobotPose operator*(const affx::Affine& a, const RobotPose &p) {
        affx::Affine result = a.operator *(p);
        return RobotPose(result, p.d);
    }

    std::string toString() const {
        return affx::Affine::toString() + ", d: " + std::to_string(d);
    }
};
