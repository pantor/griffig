#pragma once

#include <cmath>
#include <limits>

#include <opencv2/opencv.hpp>

#include <movex/affine.hpp>


class OrthographicImage {
  using Affine = movex::Affine;

  int min_value {std::numeric_limits<ushort>::min()};
  int max_value {std::numeric_limits<ushort>::max()};

  template<class T>
  double crop(double value, double min, double max) const {
    return (T)std::min(std::max(value, min), max);
  }

public:
  double pixel_size; // [px/m]
  double min_depth, max_depth; // [m]

  cv::Mat mat;
  std::string camera;

  Affine pose;

  explicit OrthographicImage(cv::Mat mat, double pixel_size, double min_depth, double max_depth): mat(mat), pixel_size(pixel_size), min_depth(min_depth), max_depth(max_depth), camera(""), pose(Affine()) { }
  explicit OrthographicImage(cv::Mat mat, double pixel_size, double min_depth, double max_depth, const std::string& camera): mat(mat), pixel_size(pixel_size), min_depth(min_depth), max_depth(max_depth), camera(camera), pose(Affine()) { }
  explicit OrthographicImage(cv::Mat mat, double pixel_size, double min_depth, double max_depth, const std::string& camera, const Affine& pose): mat(mat), pixel_size(pixel_size), min_depth(min_depth), max_depth(max_depth), camera(camera), pose(pose) { }

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

  OrthographicImage translate(std::array<double, 3> vector) const {
    OrthographicImage result = OrthographicImage(cv::Mat::zeros(mat.size(), mat.type()), pixel_size, min_depth, max_depth);

    // Depth translation
    double depth_pixel_size = max_value / (max_depth - min_depth);
    for (int i = 0; i < mat.rows; i++) {
      for (int j = 0; j < mat.cols; j++) {
        result.mat.at<ushort>(i, j) = crop<ushort>(mat.at<ushort>(i, j) + depth_pixel_size * vector[2], min_value, max_value);
      }
    }

    // 2D translation
    cv::Mat trans_matrix = (cv::Mat_<double>(2, 3) << 1, 0, pixel_size * vector[0], 0, 1, pixel_size * vector[1]);
    cv::warpAffine(result.mat, result.mat, trans_matrix, mat.size());
    return result;
  }

  OrthographicImage rotateX(double angle, std::array<double, 2> vector) const {
    auto result = OrthographicImage(cv::Mat::zeros(mat.size(), mat.type()), pixel_size, min_depth, max_depth);

    for (int i = 0; i < mat.rows; i++) {
      double y = (i - mat.rows / 2) / pixel_size + vector[0];
      for (int j = 0; j < mat.cols; j++) {
        double d = depthFromValue(mat.at<ushort>(i, j)) - vector[1];

        double y_new = y * std::cos(angle) - d * std::sin(angle);
        double d_new = y * std::sin(angle) + d * std::cos(angle);

        int i_new = crop<int>(std::round((y_new - vector[0]) * pixel_size + mat.rows / 2), 0, mat.rows - 1);
        ushort val_new = valueFromDepth(d_new + vector[1]);

        result.mat.at<ushort>(i_new, j) = val_new;
      }
    }

    return result;
  }

  OrthographicImage rotateY(double angle, std::array<double, 2> vector) const {
    OrthographicImage result = OrthographicImage(cv::Mat::zeros(mat.size(), mat.type()), pixel_size, min_depth, max_depth);

    for (int j = 0; j < mat.cols; j++) {
      double x = (j - mat.cols / 2) / pixel_size + vector[0];
      for (int i = 0; i < mat.rows; i++) {
        double d = depthFromValue(mat.at<ushort>(i, j)) - vector[1];

        double x_new = x * std::cos(angle) - d * std::sin(angle);
        double d_new = x * std::sin(angle) + d * std::cos(angle);

        int j_new = crop<int>(std::round((x_new - vector[0]) * pixel_size + mat.cols / 2), 0, mat.cols - 1);
        ushort val_new = valueFromDepth(d_new + vector[1]);

        result.mat.at<ushort>(i, j_new) = val_new;
      }
    }

    return result;
  }

  OrthographicImage rotateZ(double angle, std::array<double, 2> vector) const {
    OrthographicImage result = OrthographicImage(cv::Mat::zeros(mat.size(), mat.type()), pixel_size, min_depth, max_depth);

    cv::Point2f pc(result.mat.cols / 2 + vector[0] * pixel_size, result.mat.rows / 2 + vector[1] * pixel_size);
    cv::Mat rot_matrix = cv::getRotationMatrix2D(pc, angle * 180.0 / CV_PI, 1.0);

    cv::warpAffine(mat, result.mat, rot_matrix, result.mat.size());
    return result;
  }

  OrthographicImage rescale(double new_pixel_size, double new_min_depth, double new_max_depth) const {
    OrthographicImage result = OrthographicImage(cv::Mat::zeros(mat.size(), mat.type()), new_pixel_size, new_min_depth, new_max_depth);

    // Depth rescale
    for (int i = 0; i < mat.rows; i++) {
      for (int j = 0; j < mat.cols; j++) {
        double d = depthFromValue(mat.at<ushort>(i, j));
        result.mat.at<ushort>(i, j) = result.valueFromDepth(d);
      }
    }

    // 2D rescale
    double pixel_scale = new_pixel_size / result.pixel_size;
    cv::Mat trans_matrix = (cv::Mat_<double>(2,3) << pixel_scale, 0, 0, 0, pixel_scale, 0);
    cv::warpAffine(result.mat, result.mat, trans_matrix, mat.size());

    return result;
  }
};
