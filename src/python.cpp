#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <griffig/ndarray_converter.hpp>
#include <griffig/orthographic_image.hpp>
#include <griffig/checker.hpp>
#include <griffig/griffig.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal


PYBIND11_MODULE(_griffig, m) {
    NDArrayConverter::init_numpy();

    py::class_<OrthographicImage>(m, "OrthographicImage")
        .def(py::init<const cv::Mat&, double, double, double>())
        .def(py::init<const cv::Mat&, double, double, double, const std::string&>())
        .def(py::init<const cv::Mat&, double, double, double, const std::string&, const movex::Affine&>())
        .def_readwrite("mat", &OrthographicImage::mat)
        .def_readwrite("pixel_size", &OrthographicImage::pixel_size)
        .def_readwrite("min_depth", &OrthographicImage::min_depth)
        .def_readwrite("max_depth", &OrthographicImage::max_depth)
        .def_readwrite("camera", &OrthographicImage::camera)
        .def_readwrite("pose", &OrthographicImage::pose)
        .def("depth_from_value", &OrthographicImage::depthFromValue)
        .def("value_from_depth", &OrthographicImage::valueFromDepth)
        .def("project", &OrthographicImage::project)
        .def("inverse_project", &OrthographicImage::inverse_project)
        .def("position_from_index", &OrthographicImage::positionFromIndex)
        .def("index_from_position", &OrthographicImage::indexFromPosition)
        .def("translate", &OrthographicImage::translate)
        .def("rotate_x", &OrthographicImage::rotateX)
        .def("rotate_y", &OrthographicImage::rotateY)
        .def("rotate_z", &OrthographicImage::rotateZ)
        .def("rescale", &OrthographicImage::rescale);

    py::class_<BoxData>(m, "BoxData")
        .def(py::init<const std::vector<std::array<double, 3>>&, const movex::Affine&>(), "contour"_a, "pose"_a)
        .def(py::init<const std::array<double, 3>&, const std::array<double, 3>&, const movex::Affine&>(), "center"_a, "size"_a, "pose"_a)
        .def_readwrite("contour", &BoxData::contour)
        .def_readwrite("pose", &BoxData::pose)
        .def("as_dict", [](BoxData self) {
            py::dict d;
            d["contour"] = self.contour;
            return d;
        });

    py::class_<Griffig>(m, "Griffig")
        .def(py::init<const BoxData&>(), "contour"_a)
        // .def("render_pointcloud", &Griffig::check_collision);
        .def("draw_box_on_image", &Griffig::draw_box_on_image);

    // py::class_<Checker>(m, "Checker")
    //     .def_readwrite("debug", &Checker::debug)
    //     .def(py::init<const std::array<double, 3>&, const std::array<double, 3>&>(), "finger_size"_a, "gripper_size"_a=std::array<double, 3>({0.0, 0.0, 0.0}))
    //     .def("check_collision", &Checker::check_collision);
}
