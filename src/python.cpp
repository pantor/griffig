#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <griffig/ndarray_converter.hpp>
#include <griffig/griffig.hpp>


namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal


PYBIND11_MODULE(_griffig, m) {
    NDArrayConverter::init_numpy();

    py::class_<Gripper>(m, "Gripper");

    py::class_<Pointcloud>(m, "Pointcloud")
        .def_readwrite("size", &Pointcloud::size);

    py::class_<BoxData>(m, "BoxData")
        .def(py::init<const std::vector<std::array<double, 3>>&, const std::optional<movex::Affine>&>(), "contour"_a, "pose"_a = std::nullopt)
        .def(py::init<const std::array<double, 3>&, const std::array<double, 3>&, const std::optional<movex::Affine>&>(), "center"_a, "size"_a, "pose"_a = std::nullopt)
        .def_readwrite("contour", &BoxData::contour)
        .def_readwrite("pose", &BoxData::pose)
        .def("as_dict", [](BoxData self) {
            py::dict d;
            d["contour"] = self.contour;
            return d;
        });

    py::class_<RobotPose, movex::Affine>(m, "RobotPose")
        .def(py::init<const movex::Affine&, double>(), "affine"_a, "d"_a)
        .def(py::init<double, double, double, double, double, double, double>(), "x"_a=0.0, "y"_a=0.0, "z"_a=0.0, "a"_a=0.0, "b"_a=0.0, "c"_a=0.0, "d"_a=0.0)
        .def(py::init<double, double, double, double, double, double, double, double>(), "x"_a, "y"_a, "z"_a, "q_w"_a, "q_x"_a, "q_y"_a, "q_z"_a, "d"_a)
        .def(py::init([](py::dict d) {
            if (d.contains("q_x")) { // Prefer quaternion construction
                return RobotPose(d["x"].cast<double>(), d["y"].cast<double>(), d["z"].cast<double>(), d["q_w"].cast<double>(), d["q_x"].cast<double>(), d["q_y"].cast<double>(), d["q_z"].cast<double>(), d["d"].cast<double>());
            }
            return RobotPose(d["x"].cast<double>(), d["y"].cast<double>(), d["z"].cast<double>(), d["a"].cast<double>(), d["b"].cast<double>(), d["c"].cast<double>(), d["d"].cast<double>());
        }))
        .def_readwrite("d", &RobotPose::d)
        .def(py::self * movex::Affine())
        .def(movex::Affine() * py::self)
        .def("__repr__", &RobotPose::toString)
        .def("as_dict", [](RobotPose self) {
            auto translation = self.translation();
            auto quaternion = self.quaternion();

            py::dict d;
            d["x"] = translation.x();
            d["y"] = translation.y();
            d["z"] = translation.z();
            d["a"] = self.a();
            d["b"] = self.b();
            d["c"] = self.c();
            d["q_w"] = quaternion.w();
            d["q_x"] = quaternion.x();
            d["q_y"] = quaternion.y();
            d["q_z"] = quaternion.z();
            d["d"] = self.d;
            return d;
        });

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

    py::class_<Renderer>(m, "Renderer")
        .def(py::init<const BoxData&>(), "contour"_a)
        .def("draw_pointcloud", &Renderer::draw_pointcloud<true>)
        .def("draw_depth_pointcloud", &Renderer::draw_pointcloud<false>)
        .def("draw_gripper_on_image", &Renderer::draw_gripper_on_image)
        .def("draw_box_on_image", &Renderer::draw_box_on_image);
}
