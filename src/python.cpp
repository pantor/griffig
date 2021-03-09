#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <griffig/ndarray_converter.hpp>
#include <griffig/griffig.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal


PYBIND11_MODULE(griffig, m) {
    NDArrayConverter::init_numpy();

    py::class_<BoxContour>(m, "BoxContour")
        .def(py::init<const std::vector<std::array<double, 3>>&>(), "corners"_a)
        .def(py::init<const std::array<double, 3>&, const std::array<double, 3>&>(), "center"_a, "size"_a)
        .def_readwrite("corners", &BoxContour::corners);

    py::class_<Griffig>(m, "Griffig")
        .def(py::init<const BoxContour&>(), "contour"_a)
        // .def("render_pointcloud", &Griffig::check_collision);
        .def("draw_box_on_image", &Griffig::draw_box_on_image);
}
