#pragma once

#include <cmath>
#include <limits>
#include <optional>

#include <opencv2/opencv.hpp>

#include <affx/affine.hpp>


class OrthographicImage {
  using Affine = affx::Affine;

  int min_value {std::numeric_limits<ushort>::min()};
  int max_value {std::numeric_limits<ushort>::max()};

  template<class T>
  T crop(double value, double min, double max) const {
    return std::min(std::max(value, min), max);
  }

public:
  double pixel_size; // [px/m]
  double min_depth, max_depth; // [m]

  cv::Mat mat;

  std::optional<std::string> camera;
  std::optional<Affine> pose;

  explicit OrthographicImage(const cv::Mat& mat, double pixel_size, double min_depth, double max_depth, const std::optional<std::string>& camera = std::nullopt, const std::optional<Affine>& pose = std::nullopt): mat(mat), pixel_size(pixel_size), min_depth(min_depth), max_depth(max_depth), camera(camera), pose(pose) { }
  OrthographicImage(const OrthographicImage& image) {
    pixel_size = image.pixel_size;
    min_depth = image.min_depth;
    max_depth = image.max_depth;

    min_value = image.min_value;
    max_value = image.max_value;

    mat = image.mat.clone();
    camera = image.camera;
    pose = image.pose;
  }

  double depthFromValue(double value) const {
    return max_depth + (value / max_value) * (min_depth - max_depth);
  }

  double valueFromDepth(double depth) const {
    double value = std::round((depth - max_depth) / (min_depth - max_depth) * max_value);
    return crop<double>(value, min_value, max_value);
  }

  std::array<int, 2> project(const Affine& point) {
    return {
      int(round(mat.size().width / 2 - pixel_size * point.y())),
      int(round(mat.size().height / 2 - pixel_size * point.x())),
    };
  }

  Affine inverse_project(const std::array<double, 2>& point) {
    // double d = depthFromValue(mat.at<ushort>(point[1], point[0]));

    double x = (mat.size().height / 2 - point[1]) / pixel_size;
    double y = (mat.size().width / 2 - point[0]) / pixel_size;
    double z = 0.0; // TODO Test
    return Affine(x, y, z, 1.0, 0.0, 0.0, 0.0);
  }

  double positionFromIndex(int idx, int length) const {
    return ((idx + 0.5) - double(length) / 2) / pixel_size;
  }

  int indexFromPosition(double position, int length) const {
    return ((position * pixel_size) + (length / 2) - 0.5);
  }
};
