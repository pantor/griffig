#pragma once

#include <movex/affine.hpp>


struct RobotPose: movex::Affine {
    double d;

    explicit RobotPose(double x, double y, double z, double a, double b, double c, double d): movex::Affine(x, y, z, a, b, c), d(d) { }
    explicit RobotPose(double x, double y, double z, double q_w, double q_x, double q_y, double q_z, double d): movex::Affine(x, y, z, q_w, q_x, q_y, q_z), d(d) { }
    explicit RobotPose(const movex::Affine& affine, double d): movex::Affine(affine), d(d) { }

    RobotPose operator *(const movex::Affine& a) const {
        movex::Affine result = Affine::operator *(a);
        return RobotPose(result, d);
    }

    friend RobotPose operator*(const movex::Affine& a, const RobotPose &p) {
        movex::Affine result = a.operator *(p);
        return RobotPose(result, p.d);
    }

    std::string toString() const {
        return movex::Affine::toString() + ", d: " + std::to_string(d);
    }
};
