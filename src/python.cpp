#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <griffig/ndarray_converter.hpp>
#include <griffig/griffig.hpp>


namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal


PYBIND11_MODULE(_griffig, m) {
    NDArrayConverter::init_numpy();

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

    py::class_<Gripper>(m, "Gripper")
        .def(py::init<const std::optional<movex::Affine>&, const std::optional<std::array<double, 2>>&, const std::optional<std::array<double, 3>>&>(), "robot_to_tip"_a = std::nullopt, "width_interval"_a = std::nullopt, "finger_size"_a = std::nullopt)
        .def_readwrite("robot_to_tip", &Gripper::robot_to_tip)
        .def_readwrite("width_interval", &Gripper::width_interval)
        .def_readwrite("finger_size", &Gripper::finger_size);

    py::enum_<PointType>(m, "PointType")
        .value("XYZ", PointType::XYZ)
        .value("XYZRGB", PointType::XYZRGB)
        .value("UV", PointType::UV)
        .export_values();

    py::class_<Pointcloud>(m, "Pointcloud")
        .def(py::init([](py::object realsense_frames, py::object ros_message, py::object type, py::object data) {
            if (realsense_frames) {
                std::cout << "HERE" << std::endl;
                py::object depth = realsense_frames.attr("get_depth_frame")();
                py::object color = realsense_frames.attr("get_color_frame")();

                py::object pointcloud = py::module_::import("pyrealsense2").attr("pointcloud");
                // py::object points = pointcloud.attr("calculate")(pointcloud, depth);
                // pointcloud.attr("map_to")(color);
                // return std::make_unique<Pointcloud>(points.size(), color.get_width(), color.get_height(), points.get_vertices(), color.get_data(), points.get_texture_coordinates());

            } else if (ros_message) {

            } else if (type && data) {

            } else {

            }
            return std::make_unique<Pointcloud>();
        }), py::kw_only(), "realsense_frames"_a=py::none(), "ros_message"_a=py::none(), "type"_a=py::none(), "data"_a=py::none())
        .def_readwrite("size", &Pointcloud::size);

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

    py::class_<OrthographicData>(m, "OrthographicData")
        .def(py::init<double, double, double>(), "pixel_density"_a, "min_depth"_a, "max_depth"_a)
        .def_readwrite("pixel_density", &OrthographicData::pixel_density)
        .def_readwrite("min_depth", &OrthographicData::min_depth)
        .def_readwrite("max_depth", &OrthographicData::max_depth);

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
        .def("clone", [](py::object self) {
            py::object b = py::cast(OrthographicImage(self.cast<OrthographicImage>()));
            b.attr("mat") = self.attr("mat").attr("copy")();
            return b;
        })
        .def("depth_from_value", &OrthographicImage::depthFromValue)
        .def("value_from_depth", &OrthographicImage::valueFromDepth)
        .def("project", &OrthographicImage::project)
        .def("inverse_project", &OrthographicImage::inverse_project)
        .def("position_from_index", &OrthographicImage::positionFromIndex)
        .def("index_from_position", &OrthographicImage::indexFromPosition);

    py::class_<Renderer>(m, "Renderer")
        .def(py::init<double, double, const std::optional<BoxData>&>(), "pixel_size"_a, "depth_diff"_a, "contour"_a = std::nullopt)
        .def("draw_pointcloud", &Renderer::draw_pointcloud<true>, "pointcloud"_a, "size"_a, "ortho"_a, "camera_position"_a = (std::array<double, 3>){0.0, 0.0, 0.0})
        .def("draw_depth_pointcloud", &Renderer::draw_pointcloud<false>)
        .def("draw_gripper_on_image", &Renderer::draw_gripper_on_image, "image"_a, "gripper"_a, "pose"_a)
        .def("draw_box_on_image", &Renderer::draw_box_on_image, "image"_a);
}
