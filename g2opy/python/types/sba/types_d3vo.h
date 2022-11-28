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

}

}  // end namespace g2o