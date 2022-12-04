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

#ifndef G2O_SIX_DOF_TYPES_EXPMAP
#define G2O_SIX_DOF_TYPES_EXPMAP

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/types/slam3d/se3_ops.h"
#include "types_sba.h"
#include <Eigen/Geometry>

#include "g2o/config.h"
#include "g2o/core/hyper_graph_action.h"
#include "g2o/types/slam3d/isometry3d_mappings.h"
#include "g2o/types/slam3d/g2o_types_slam3d_api.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>



namespace g2o {
namespace types_six_dof_expmap {
void init();
}

typedef Eigen::Matrix<double, 6, 6, Eigen::ColMajor> Matrix6d;

class G2O_TYPES_SBA_API CameraParameters : public g2o::Parameter
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CameraParameters();

    CameraParameters(double focal_length,
        const Vector2D & principle_point,
        double baseline)
      : focal_length(focal_length),
      principle_point(principle_point),
      baseline(baseline){}

    Vector2D cam_map (const Vector3D & trans_xyz) const;

    Vector3D cam_unmap(const Vector2D & trans_uv, const double depth) const;

    Vector3D stereocam_uvu_map (const Vector3D & trans_xyz) const;

    virtual bool read (std::istream& is){
      is >> focal_length;
      is >> principle_point[0];
      is >> principle_point[1];
      is >> baseline;
      return true;
    }

    virtual bool write (std::ostream& os) const {
      os << focal_length << " ";
      os << principle_point.x() << " ";
      os << principle_point.y() << " ";
      os << baseline << " ";
      return true;
    }

    double focal_length;
    Vector2D principle_point;
    double baseline;
};

/**
 * \brief SE3 Vertex parameterized internally with a transformation matrix
 and externally with its exponential map
 */
class G2O_TYPES_SBA_API VertexSE3Expmap : public BaseVertex<6, SE3Quat>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexSE3Expmap();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  virtual void setToOriginImpl() {
    _estimate = SE3Quat();
  }

  virtual void oplusImpl(const double* update_)  {
    Eigen::Map<const Vector6d> update(update_);
    setEstimate(SE3Quat::exp(update)*estimate());
  }
};


/**
 * \brief 6D edge between two Vertex6
 */
class G2O_TYPES_SBA_API EdgeSE3Expmap : public BaseBinaryEdge<6, SE3Quat, VertexSE3Expmap, VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
      EdgeSE3Expmap();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
      const VertexSE3Expmap* v2 = static_cast<const VertexSE3Expmap*>(_vertices[1]);

      SE3Quat C(_measurement);
      SE3Quat error_= v2->estimate().inverse()*C*v1->estimate();
      _error = error_.log();
    }

    virtual void linearizeOplus();
};


class G2O_TYPES_SBA_API EdgeProjectXYZ2UV : public  BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZ2UV();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
      const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
      const CameraParameters * cam
        = static_cast<const CameraParameters *>(parameter(0));
      Vector2D obs(_measurement);
      _error = obs-cam->cam_map(v1->estimate().map(v2->estimate()));
    }

    virtual void linearizeOplus();

    CameraParameters * _cam;
};


class G2O_TYPES_SBA_API EdgeProjectPSI2UV : public  g2o::BaseMultiEdge<2, Vector2D>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeProjectPSI2UV()  {
    resizeParameters(1);
    installParameter(_cam, 0);
  }

  virtual bool read  (std::istream& is);
  virtual bool write (std::ostream& os) const;
  void computeError  ();
  virtual void linearizeOplus ();
  CameraParameters * _cam;
};




//Stereo Observations
// U: left u
// V: left v
// U: right u
class G2O_TYPES_SBA_API EdgeProjectXYZ2UVU : public  BaseBinaryEdge<3, Vector3D, VertexSBAPointXYZ, VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZ2UVU();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError(){
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
      const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
      const CameraParameters * cam
        = static_cast<const CameraParameters *>(parameter(0));
      Vector3D obs(_measurement);
      _error = obs-cam->stereocam_uvu_map(v1->estimate().map(v2->estimate()));
    }
    //  virtual void linearizeOplus();
    CameraParameters * _cam;
};

// Projection using focal_length in x and y directions
class EdgeSE3ProjectXYZ : public BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZ();

  bool read(std::istream &is);

  bool write(std::ostream &os) const;

  void computeError() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[1]);
    const VertexSBAPointXYZ *v2 = static_cast<const VertexSBAPointXYZ *>(_vertices[0]);
    Vector2D obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(v2->estimate()));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[1]);
    const VertexSBAPointXYZ *v2 = static_cast<const VertexSBAPointXYZ *>(_vertices[0]);
    return (v1->estimate().map(v2->estimate()))(2) > 0.0;
  }

  virtual void linearizeOplus();

  Vector2D cam_project(const Vector3D &trans_xyz) const;

  double fx, fy, cx, cy;
};

// Edge to optimize only the camera pose
class EdgeSE3ProjectXYZOnlyPose : public BaseUnaryEdge<2, Vector2D, VertexSE3Expmap> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZOnlyPose() {}

  bool read(std::istream &is);

  bool write(std::ostream &os) const;

  void computeError() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
    Vector2D obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
    return (v1->estimate().map(Xw))(2) > 0.0;
  }

  virtual void linearizeOplus();

  Vector2D cam_project(const Vector3D &trans_xyz) const;

  Vector3D Xw;
  double fx, fy, cx, cy;
};

// Projection using focal_length in x and y directions stereo
class EdgeStereoSE3ProjectXYZ : public BaseBinaryEdge<3, Vector3D, VertexSBAPointXYZ, VertexSE3Expmap> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeStereoSE3ProjectXYZ();

  bool read(std::istream &is);

  bool write(std::ostream &os) const;

  void computeError() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[1]);
    const VertexSBAPointXYZ *v2 = static_cast<const VertexSBAPointXYZ *>(_vertices[0]);
    Vector3D obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(v2->estimate()), bf);
  }

  bool isDepthPositive() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[1]);
    const VertexSBAPointXYZ *v2 = static_cast<const VertexSBAPointXYZ *>(_vertices[0]);
    return (v1->estimate().map(v2->estimate()))(2) > 0.0;
  }

  virtual void linearizeOplus();

  Vector3D cam_project(const Vector3D &trans_xyz, const float &bf) const;

  double fx, fy, cx, cy, bf;
};

// Edge to optimize only the camera pose stereo
class EdgeStereoSE3ProjectXYZOnlyPose : public BaseUnaryEdge<3, Vector3D, VertexSE3Expmap> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeStereoSE3ProjectXYZOnlyPose() {}

  bool read(std::istream &is);

  bool write(std::ostream &os) const;

  void computeError() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
    Vector3D obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
    return (v1->estimate().map(Xw))(2) > 0.0;
  }

  virtual void linearizeOplus();

  Vector3D cam_project(const Vector3D &trans_xyz) const;

  Vector3D Xw;
  double fx, fy, cx, cy, bf;
};







/*
An Edge to implement the D3VO photometric error. Connects to a VertexD3VOPointDepth and two VertexD3VOFramePose.
*/
class G2O_TYPES_SBA_API EdgeProjectD3VO : public  g2o::BaseMultiEdge<3, Vector3D>
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeProjectD3VO()  {
            resizeParameters(1);
            installParameter(_cam, 0);
            out_of_bounds = false;
        }

        virtual bool read(std::istream& is) {return false;}
        virtual bool write(std::ostream& os) const {return false;}
        void computeError();
        virtual void linearizeOplus();
        CameraParameters * _cam;
        bool out_of_bounds;
};


/*
A Vertex type to represent a single point depth estimate. The depth estimate for this vertex comes from its host frame.
*/
class G2O_TYPES_SBA_API VertexD3VOPointDepth : public BaseVertex<1, double>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Pixel coordinate in this Point's host frame
        Vector2D uv;

        VertexD3VOPointDepth(const int u, const int v) {
            uv = Vector2D(u, v);
        }

        virtual void setToOriginImpl() {
            _estimate = 0.;
        }

        virtual void oplusImpl(const double* update_)  {
            // std::cout << "VertexD3VOPointDepth oplusImpl update: (" << *update_ << ")" << std::endl;
            _estimate += (*update_);
        }

        virtual bool setEstimateDataImpl(const double* est){
            // std::cout << "setting (" << uv(0) << ", "<< uv(1) << ") estimate to: " << *est << std::endl;
            _estimate = *est;
            return true;
        }

        virtual bool getEstimateData(double* est) const{
            *est = _estimate;
            return true;
        }

        virtual bool read (std::istream& is){return false;}

        virtual bool write (std::ostream& os) const {return false;}
};



/*
A SE3 Vertex parameterized internally with a transformation matrix and externally with its exponential map used to 
represent the pose of a single Frame. This vertex also stores the image of the frame it represents in order to 
implement a photometric error.
 
Inspired by VertexSE3Expmap. SE3Quat slightly worse performance than using Isometry3D (inspired by VertexSE3), 
but optimization is much more stable and matches the exponential map we expect to see in the pose update.
*/
class G2O_TYPES_SLAM3D_API VertexD3VOFramePose : public BaseVertex<6, SE3Quat>
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        pybind11::buffer_info pixel_inten;
       
        VertexD3VOFramePose(pybind11::array_t<double> pixel_intensity) : BaseVertex<6, SE3Quat>()
        {
            setToOriginImpl();
            updateCache();

            // Process + store input numpy array
            pixel_inten = pixel_intensity.request();
        }

        virtual void setToOriginImpl() {
            _estimate = SE3Quat();
        }

        virtual void oplusImpl(const double* update_)  {
            Eigen::Map<const Vector6d> update(update_);
            setEstimate(SE3Quat::exp(update)*estimate());
        }

    virtual bool read (std::istream& is){return false;}

    virtual bool write (std::ostream& os) const {return false;}

};
} // end namespace

#endif

