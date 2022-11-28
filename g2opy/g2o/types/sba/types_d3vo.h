// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// #ifndef G2O_SIX_DOF_TYPES_EXPMAP
// #define G2O_SIX_DOF_TYPES_EXPMAP

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/types/slam3d/se3_ops.h"
#include "types_sba.h"
#include "types_six_dof_expmap.h"
#include <Eigen/Geometry>

namespace g2o {
namespace types_six_dof_expmap {
}

// 3-way edge between two Frames and a Point
class G2O_TYPES_SBA_API EdgeProjectD3VO : public  g2o::BaseMultiEdge<2, Vector2D>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeProjectD3VO()  {
    resizeParameters(1);
    installParameter(_cam, 0);
  }

  virtual bool read  (std::istream& is);
  virtual bool write (std::ostream& os) const;
  void computeError  ();
  virtual void linearizeOplus ();
  CameraParameters * _cam;
};



// Vertex class for a single Point (represented by its depth)
class G2O_TYPES_SBA_API VertexD3VOPointDepth : public BaseVertex<1, double>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    VertexD3VOPointDepth(const int u, const int v) : _u(u), _v(v) {}
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() { _estimate = 0.; }

    virtual void oplusImpl(const double* update) {_estimate += *update;}

    virtual bool setEstimateDataImpl(const double* est){
        _estimate = *est;
        return true;
    }

    virtual bool getEstimateData(double* est) const{
        *est = _estimate;
        return true;
    }

    // Don't want to have to mess with copy constructors + inheritance, setEstimateDataImpl works for our uses
    // virtual void setEstimate(const VertexD3VOPointDepth& vert){
    //   self._estimate = vert._estimate
    // }

    virtual int estimateDimension() const {return 1;}

    virtual bool setMinimalEstimateDataImpl(const double* est){
        _estimate = *est;
        return true;
    }

    virtual bool getMinimalEstimateData(double* est) const{
        *est = _estimate;
        return true;
    }

    virtual int minimalEstimateDimension() const {return 1;}

  private:
    // Pixel coordinates in this Point's host frame
    int _u, _v;
};







} // end namespace

//#endif













