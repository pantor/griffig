#pragma once

#include <vector>

#include <movex/affine.hpp>


struct Pointcloud {
    struct PointXYZRGB {
        float x, y, z;
        float r, g, b;
    };

    size_t width, height; // For image-based pointclouds

    std::vector<PointXYZRGB> points;
    movex::Affine pose;  // Sensor pose

    Pointcloud() { }
    Pointcloud(size_t width, size_t height): width(width), height(height) {
        resize(width * height);
    }

    void resize(size_t count) {
        points.resize(count);
        if (count != width * height) {
            width = count;
            height = 1;
        }
    }
};
