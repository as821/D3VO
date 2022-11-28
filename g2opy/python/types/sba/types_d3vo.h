#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <g2o/types/sba/types_d3vo.h>

#include <g2o/types/slam3d/se3quat.h>
//#include "python/core/base_vertex.h"
#include "python/core/base_unary_edge.h"
#include "python/core/base_binary_edge.h"
#include "python/core/base_multi_edge.h"


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareD3VO(py::module & m) {
    py::class_<EdgeProjectD3VO, BaseMultiEdge<2, Vector2D>>(m, "EdgeProjectD3VO")
        .def(py::init())
        .def("compute_error", &EdgeProjectD3VO::computeError)
        .def("linearize_oplus", &EdgeProjectD3VO::linearizeOplus)
    ;

    // constructor takes arguments --> VertexD3VOPointDepth(int u, int v) : _u(u), _v(v) {}
    py::class_<VertexD3VOPointDepth, BaseVertex<1, double>>(m, "VertexD3VOPointDepth")
        .def(py::init<const int, const int>())
        .def("set_to_origin_impl", &VertexD3VOPointDepth::setToOriginImpl)
        // .def("set_estimate", &VertexD3VOPointDepth::setEstimate)   // const VertexD3VOPointDepth& -> void
        .def("oplus_impl", &VertexD3VOPointDepth::oplusImpl)
        .def("set_estimate_data_impl", &VertexD3VOPointDepth::setEstimateDataImpl)
        .def("get_estimate_data", &VertexD3VOPointDepth::getEstimateData)
        .def("estimate_dimension", &VertexD3VOPointDepth::estimateDimension)
        .def("set_minimal_estimate_data_impl", &VertexD3VOPointDepth::setMinimalEstimateDataImpl)
        .def("get_minimal_estimate_data", &VertexD3VOPointDepth::getMinimalEstimateData)
        .def("minimal_estimate_dimension", &VertexD3VOPointDepth::minimalEstimateDimension)
    ;
}

}  // end namespace g2o