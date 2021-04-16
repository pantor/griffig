#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <griffig/ndarray_converter.hpp>
#include <griffig/griffig.hpp>


namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal
using Affine = affx::Affine;


PYBIND11_MODULE(_griffig, m) {
    NDArrayConverter::init_numpy();
    py::module::import("pyaffx");

    py::class_<BoxData>(m, "BoxData")
        .def(py::init<const std::vector<std::array<double, 3>>&, const std::optional<Affine>&>(), "contour"_a, "pose"_a = std::nullopt)
        .def(py::init<const std::array<double, 3>&, const std::array<double, 3>&, const std::optional<Affine>&>(), "center"_a, "size"_a, "pose"_a = std::nullopt)
        .def_readwrite("contour", &BoxData::contour)
        .def_readwrite("pose", &BoxData::pose)
        .def("get_rect", &BoxData::get_rect, "pixel_size"_a, "offset"_a=0)
        .def("as_dict", [](BoxData self) {
            py::dict d;
            d["contour"] = self.contour;
            return d;
        });

    py::class_<Gripper>(m, "Gripper")
        .def(py::init<double, double, double, double, double, Affine>(), "min_stroke"_a = 0.0, "max_stroke"_a = std::numeric_limits<double>::infinity(), "finger_width"_a = 0.0, "finger_extent"_a = 0.0, "finger_height"_a = 0.0, "offset"_a = Affine())
        .def_readwrite("min_stroke", &Gripper::min_stroke)
	    .def_readwrite("max_stroke", &Gripper::max_stroke)
	    .def_readwrite("finger_width", &Gripper::finger_width)
	    .def_readwrite("finger_extent", &Gripper::finger_extent)
        .def_readwrite("finger_height", &Gripper::finger_height)
        .def_readwrite("offset", &Gripper::offset)
        .def("consider_indices", &Gripper::consider_indices);

    py::class_<OrthographicImage>(m, "OrthographicImage")
        .def(py::init<const cv::Mat&, double, double, double, const std::optional<std::string>&, const Affine&>(), "mat"_a, "pixel_size"_a, "min_depth"_a, "max_depth"_a, "camera"_a=std::nullopt, "pose"_a=Affine())
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

    py::enum_<PointType>(m, "PointType")
        .value("XYZ", PointType::XYZ)
        .value("XYZRGB", PointType::XYZRGB)
        .value("XYZWRGBA", PointType::XYZWRGBA)
        .value("UV", PointType::UV)
        .export_values();

    py::class_<Pointcloud>(m, "Pointcloud")
        .def(py::init([](py::object frames) {
            py::object pc = py::module_::import("pyrealsense2").attr("pointcloud")();
            py::object depth = frames.attr("get_depth_frame")();
            py::object color = frames.attr("get_color_frame")();

            py::object points = pc.attr("calculate")(depth);
            pc.attr("map_to")(color);

            size_t size = points.attr("size")().cast<size_t>();
            int width = color.attr("get_width")().cast<int>();
            int height = color.attr("get_height")().cast<int>();

            py::buffer vertices_buf = points.attr("get_vertices")();
            py::buffer tex_buf = color.attr("get_data")();
            py::buffer tex_coords_buf = points.attr("get_texture_coordinates")();

            const void* vertices = vertices_buf.request().ptr;
            const void* texture = tex_buf.request().ptr;
            const void* tex_coords = tex_coords_buf.request().ptr;

            auto result = std::make_unique<Pointcloud>(size, width, height, vertices, texture, tex_coords);
            result->pc = std::move(pc);
            return result;
        }), py::kw_only(), "realsense_frames"_a=py::none())
        .def(py::init([](PointType type, py::object data) {
            size_t point_size;
	        switch (type) {
		        default:
		        case PointType::XYZWRGBA: { point_size = sizeof(PointTypes::XYZWRGBA); } break;
		        case PointType::XYZ: { point_size = sizeof(PointTypes::XYZ); } break;
		        case PointType::XYZRGB: { point_size = sizeof(PointTypes::XYZRGB); } break;
	        }

	        py::buffer_info info(py::buffer((py::bytes)data).request());
            return std::make_unique<Pointcloud>(static_cast<size_t>(info.size) / point_size, type, info.ptr);
        }), py::kw_only(), "type"_a=PointType::XYZWRGBA, "data"_a=py::none())
        .def_readonly("size", &Pointcloud::size)
        .def_readonly("point_type", &Pointcloud::point_type);

    py::class_<RobotPose, Affine>(m, "RobotPose")
        .def(py::init<const Affine&, double>(), "affine"_a, "d"_a)
        .def(py::init<double, double, double, double, double, double, double>(), "x"_a=0.0, "y"_a=0.0, "z"_a=0.0, "a"_a=0.0, "b"_a=0.0, "c"_a=0.0, "d"_a=0.0)
        .def(py::init<double, double, double, double, double, double, double, double>(), "x"_a, "y"_a, "z"_a, "q_w"_a, "q_x"_a, "q_y"_a, "q_z"_a, "d"_a)
        .def(py::init([](py::dict d) {
            if (d.contains("q_x")) { // Prefer quaternion construction
                return RobotPose(d["x"].cast<double>(), d["y"].cast<double>(), d["z"].cast<double>(), d["q_w"].cast<double>(), d["q_x"].cast<double>(), d["q_y"].cast<double>(), d["q_z"].cast<double>(), d["d"].cast<double>());
            }
            return RobotPose(d["x"].cast<double>(), d["y"].cast<double>(), d["z"].cast<double>(), d["a"].cast<double>(), d["b"].cast<double>(), d["c"].cast<double>(), d["d"].cast<double>());
        }))
        .def_readwrite("d", &RobotPose::d)
        .def(py::self * Affine())
        .def(Affine() * py::self)
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

    py::class_<Renderer>(m, "Renderer")
        .def(py::init<const std::array<int, 2>&>(), "size"_a)
        .def(py::init<const std::array<int, 2>&, double, double, const std::optional<BoxData>&>(), "size"_a, "pixel_size"_a, "depth_diff"_a, "contour"_a = std::nullopt)
        .def(py::init<const std::array<int, 2>&, const std::array<double, 3>&>(), "size"_a, "position"_a)
        .def(py::init<const BoxData&, double, double, double>(), "box_data"_a, "typical_camera_distance"_a, "pixel_size"_a, "depth_diff"_a)
        .def("draw_gripper_on_image", &Renderer::draw_gripper_on_image, "image"_a, "gripper"_a, "pose"_a)
        .def("draw_box_on_image", &Renderer::draw_box_on_image, "image"_a)
        .def("check_gripper_collision", &Renderer::check_gripper_collision, "image"_a, "gripper"_a, "pose"_a)
        .def("render_pointcloud", static_cast<OrthographicImage (Renderer::*)(const Pointcloud&)>(&Renderer::render_pointcloud<true>))
        .def("render_pointcloud", static_cast<OrthographicImage (Renderer::*)(const Pointcloud&, double, double, double)>(&Renderer::render_pointcloud<true>))
        .def("render_pointcloud_mat", &Renderer::render_pointcloud_mat<true>, "pointcloud"_a, "pixel_size"_a, "min_depth"_a, "max_depth"_a, "camera_position"_a = (std::array<double, 3>){0.0, 0.0, 0.0})
        .def("render_depth_pointcloud_mat", &Renderer::render_pointcloud_mat<false>)
	    .def_readwrite("camera_position", &Renderer::camera_position)
        .def_readwrite("box_data", &Renderer::box_contour)
        .def_readwrite("pixel_size", &Renderer::pixel_size)
        .def_readwrite("depth_diff", &Renderer::depth_diff)
        .def_readwrite("width", &Renderer::width)
        .def_readwrite("height", &Renderer::height);
}
