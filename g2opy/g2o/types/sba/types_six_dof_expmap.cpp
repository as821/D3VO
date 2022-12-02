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

#include "types_six_dof_expmap.h"

#include "g2o/core/factory.h"
#include "g2o/stuff/macros.h"

namespace g2o {

using namespace std;
using namespace Eigen;

G2O_REGISTER_TYPE_GROUP(expmap);
G2O_REGISTER_TYPE(VERTEX_SE3:EXPMAP, VertexSE3Expmap);
G2O_REGISTER_TYPE(EDGE_SE3:EXPMAP, EdgeSE3Expmap);
G2O_REGISTER_TYPE(EDGE_PROJECT_XYZ2UV:EXPMAP, EdgeProjectXYZ2UV);
G2O_REGISTER_TYPE(EDGE_PROJECT_XYZ2UVU:EXPMAP, EdgeProjectXYZ2UVU);
G2O_REGISTER_TYPE(PARAMS_CAMERAPARAMETERS, CameraParameters);

CameraParameters
::CameraParameters()
  : focal_length(1.),
    principle_point(Vector2D(0., 0.)),
    baseline(0.5)  {
}

Vector2D project2d(const Vector3D& v)  {
  Vector2D res;
  res(0) = v(0)/v(2);
  res(1) = v(1)/v(2);
  return res;
}

Vector3D unproject2d(const Vector2D& v)  {
  Vector3D res;
  res(0) = v(0);
  res(1) = v(1);
  res(2) = 1;
  return res;
}

inline Vector3D invert_depth(const Vector3D & x){
  return unproject2d(x.head<2>())/x[2];
}

Vector2D  CameraParameters::cam_map(const Vector3D & trans_xyz) const {
    // Project 3D point onto 2D pixel coordinate
    Vector2D proj = project2d(trans_xyz);
    Vector2D res;
    res[0] = proj[0]*focal_length + principle_point[0];
    res[1] = proj[1]*focal_length + principle_point[1];
    return res;
}

Vector3D  CameraParameters::cam_unmap(const Vector2D & trans_uv, const double depth) const {
    // Unproject 2D pixel coordinate onto 3D point
    Vector3D res;
    res(0) = (trans_uv(0) - principle_point[0]) / focal_length * depth;
    res(1) = (trans_uv(1) - principle_point[1]) / focal_length * depth;
    res(2) = depth;
    return res;
}


Vector3D CameraParameters::stereocam_uvu_map(const Vector3D & trans_xyz) const {
  Vector2D uv_left = cam_map(trans_xyz);
  double proj_x_right = (trans_xyz[0]-baseline)/trans_xyz[2];
  double u_right = proj_x_right*focal_length + principle_point[0];
  return Vector3D(uv_left[0],uv_left[1],u_right);
}


VertexSE3Expmap::VertexSE3Expmap() : BaseVertex<6, SE3Quat>() {
}

bool VertexSE3Expmap::read(std::istream& is) {
  Vector7d est;
  for (int i=0; i<7; i++)
    is  >> est[i];
  SE3Quat cam2world;
  cam2world.fromVector(est);
  setEstimate(cam2world.inverse());
  return true;
}

bool VertexSE3Expmap::write(std::ostream& os) const {
  SE3Quat cam2world(estimate().inverse());
  for (int i=0; i<7; i++)
    os << cam2world[i] << " ";
  return os.good();
}

EdgeSE3Expmap::EdgeSE3Expmap() :
  BaseBinaryEdge<6, SE3Quat, VertexSE3Expmap, VertexSE3Expmap>() {
}





bool EdgeSE3Expmap::read(std::istream& is)  {
  Vector7d meas;
  for (int i=0; i<7; i++)
    is >> meas[i];
  SE3Quat cam2world;
  cam2world.fromVector(meas);
  setMeasurement(cam2world.inverse());
  //TODO: Convert information matrix!!
  for (int i=0; i<6; i++)
    for (int j=i; j<6; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3Expmap::write(std::ostream& os) const {
  SE3Quat cam2world(measurement().inverse());
  for (int i=0; i<7; i++)
    os << cam2world[i] << " ";
  for (int i=0; i<6; i++)
    for (int j=i; j<6; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

EdgeProjectXYZ2UV::EdgeProjectXYZ2UV() : BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>() {
  _cam = 0;
  resizeParameters(1);
  installParameter(_cam, 0);
}

bool EdgeProjectPSI2UV::write(std::ostream& os) const  {
  os << _cam->id() << " ";
  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

bool EdgeProjectPSI2UV::read(std::istream& is) {
  int paramId;
  is >> paramId;
  setParameterId(0, paramId);

  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

void EdgeProjectPSI2UV::computeError(){
  const VertexSBAPointXYZ * psi = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
  const VertexSE3Expmap * T_p_from_world = static_cast<const VertexSE3Expmap*>(_vertices[1]);
  const VertexSE3Expmap * T_anchor_from_world = static_cast<const VertexSE3Expmap*>(_vertices[2]);
  const CameraParameters * cam = static_cast<const CameraParameters *>(parameter(0));

  Vector2D obs(_measurement);
  _error = obs - cam->cam_map(T_p_from_world->estimate()
        *T_anchor_from_world->estimate().inverse()
        *invert_depth(psi->estimate()));
}

inline Matrix<double,2,3,Eigen::ColMajor> d_proj_d_y(const double & f, const Vector3D & xyz){
  double z_sq = xyz[2]*xyz[2];
  Matrix<double,2,3,Eigen::ColMajor> J;
  J << f/xyz[2], 0,           -(f*xyz[0])/z_sq,
      0,           f/xyz[2], -(f*xyz[1])/z_sq;
  return J;
}

inline Matrix<double,3,6,Eigen::ColMajor> d_expy_d_y(const Vector3D & y){
  Matrix<double,3,6,Eigen::ColMajor> J;
  J.topLeftCorner<3,3>() = -skew(y);
  J.bottomRightCorner<3,3>().setIdentity();

  return J;
}

inline Matrix3D d_Tinvpsi_d_psi(const SE3Quat & T, const Vector3D & psi){
  Matrix3D R = T.rotation().toRotationMatrix();
  Vector3D x = invert_depth(psi);
  Vector3D r1 = R.col(0);
  Vector3D r2 = R.col(1);
  Matrix3D J;
  J.col(0) = r1;
  J.col(1) = r2;
  J.col(2) = -R*x;
  J*=1./psi.z();
  return J;
}

void EdgeProjectPSI2UV::linearizeOplus(){
  VertexSBAPointXYZ* vpoint = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
  Vector3D psi_a = vpoint->estimate();
  VertexSE3Expmap * vpose = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T_cw = vpose->estimate();
  VertexSE3Expmap * vanchor = static_cast<VertexSE3Expmap *>(_vertices[2]);
  const CameraParameters * cam
      = static_cast<const CameraParameters *>(parameter(0));

  SE3Quat A_aw = vanchor->estimate();
  SE3Quat T_ca = T_cw*A_aw.inverse();
  Vector3D x_a = invert_depth(psi_a);
  Vector3D y = T_ca*x_a;
  Matrix<double,2,3,Eigen::ColMajor> Jcam
      = d_proj_d_y(cam->focal_length, y);
  _jacobianOplus[0] = -Jcam*d_Tinvpsi_d_psi(T_ca, psi_a);
  _jacobianOplus[1] = -Jcam*d_expy_d_y(y);
  _jacobianOplus[2] = Jcam*T_ca.rotation().toRotationMatrix()*d_expy_d_y(x_a);
}



EdgeProjectXYZ2UVU::EdgeProjectXYZ2UVU() : BaseBinaryEdge<3, Vector3D, VertexSBAPointXYZ, VertexSE3Expmap>()
{
  _cam = 0;
  resizeParameters(1);
  installParameter(_cam, 0);
}

bool EdgeProjectXYZ2UV::read(std::istream& is){
  int paramId;
  is >> paramId;
  setParameterId(0, paramId);

  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeProjectXYZ2UV::write(std::ostream& os) const {
  os << _cam->id() << " ";
  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeSE3Expmap::linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  SE3Quat Ti(vi->estimate());

  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat Tj(vj->estimate());

  const SE3Quat & Tij = _measurement;
  SE3Quat invTij = Tij.inverse();

  SE3Quat invTj_Tij = Tj.inverse()*Tij;
  SE3Quat infTi_invTij = Ti.inverse()*invTij;

  _jacobianOplusXi = invTj_Tij.adj();
  _jacobianOplusXj = -infTi_invTij.adj();
}

void EdgeProjectXYZ2UV::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
  Vector3D xyz = vi->estimate();
  Vector3D xyz_trans = T.map(xyz);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;

  const CameraParameters * cam = static_cast<const CameraParameters *>(parameter(0));

  Matrix<double,2,3,Eigen::ColMajor> tmp;
  tmp(0,0) = cam->focal_length;
  tmp(0,1) = 0;
  tmp(0,2) = -x/z*cam->focal_length;

  tmp(1,0) = 0;
  tmp(1,1) = cam->focal_length;
  tmp(1,2) = -y/z*cam->focal_length;

  _jacobianOplusXi =  -1./z * tmp * T.rotation().toRotationMatrix();

  _jacobianOplusXj(0,0) =  x*y/z_2 *cam->focal_length;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *cam->focal_length;
  _jacobianOplusXj(0,2) = y/z *cam->focal_length;
  _jacobianOplusXj(0,3) = -1./z *cam->focal_length;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *cam->focal_length;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *cam->focal_length;
  _jacobianOplusXj(1,1) = -x*y/z_2 *cam->focal_length;
  _jacobianOplusXj(1,2) = -x/z *cam->focal_length;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *cam->focal_length;
  _jacobianOplusXj(1,5) = y/z_2 *cam->focal_length;
}

bool EdgeProjectXYZ2UVU::read(std::istream& is){
  for (int i=0; i<3; i++){
    is  >> _measurement[i];
  }
  for (int i=0; i<3; i++)
    for (int j=i; j<3; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeProjectXYZ2UVU::write(std::ostream& os) const {
  for (int i=0; i<3; i++){
    os  << measurement()[i] << " ";
  }

  for (int i=0; i<3; i++)
    for (int j=i; j<3; j++){
      os << " " << information()(i,j);
    }
  return os.good();
}

EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() : BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>() {
}

bool EdgeSE3ProjectXYZ::read(std::istream &is) {
  for (int i = 0; i < 2; i++) {
    is >> _measurement[i];
  }
  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      is >> information()(i, j);
      if (i != j)
        information()(j, i) = information()(i, j);
    }
  return true;
}

bool EdgeSE3ProjectXYZ::write(std::ostream &os) const {

  for (int i = 0; i < 2; i++) {
    os << measurement()[i] << " ";
  }

  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      os << " " << information()(i, j);
    }
  return os.good();
}

void EdgeSE3ProjectXYZ::linearizeOplus() {
  VertexSE3Expmap *vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAPointXYZ *vi = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
  Vector3d xyz = vi->estimate();
  Vector3d xyz_trans = T.map(xyz);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z * z;

  Matrix<double, 2, 3> tmp;
  tmp(0, 0) = fx;
  tmp(0, 1) = 0;
  tmp(0, 2) = -x / z * fx;

  tmp(1, 0) = 0;
  tmp(1, 1) = fy;
  tmp(1, 2) = -y / z * fy;

  _jacobianOplusXi = -1. / z * tmp * T.rotation().toRotationMatrix();

  _jacobianOplusXj(0, 0) = x * y / z_2 * fx;
  _jacobianOplusXj(0, 1) = -(1 + (x * x / z_2)) * fx;
  _jacobianOplusXj(0, 2) = y / z * fx;
  _jacobianOplusXj(0, 3) = -1. / z * fx;
  _jacobianOplusXj(0, 4) = 0;
  _jacobianOplusXj(0, 5) = x / z_2 * fx;

  _jacobianOplusXj(1, 0) = (1 + y * y / z_2) * fy;
  _jacobianOplusXj(1, 1) = -x * y / z_2 * fy;
  _jacobianOplusXj(1, 2) = -x / z * fy;
  _jacobianOplusXj(1, 3) = 0;
  _jacobianOplusXj(1, 4) = -1. / z * fy;
  _jacobianOplusXj(1, 5) = y / z_2 * fy;
}

Vector2d EdgeSE3ProjectXYZ::cam_project(const Vector3d &trans_xyz) const {
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0] * fx + cx;
  res[1] = proj[1] * fy + cy;
  return res;
}

Vector3d EdgeStereoSE3ProjectXYZ::cam_project(const Vector3d &trans_xyz, const float &bf) const {
  const float invz = 1.0f / trans_xyz[2];
  Vector3d res;
  res[0] = trans_xyz[0] * invz * fx + cx;
  res[1] = trans_xyz[1] * invz * fy + cy;
  res[2] = res[0] - bf * invz;
  return res;
}

EdgeStereoSE3ProjectXYZ::EdgeStereoSE3ProjectXYZ() : BaseBinaryEdge<3, Vector3d, VertexSBAPointXYZ, VertexSE3Expmap>() {
}

bool EdgeStereoSE3ProjectXYZ::read(std::istream &is) {
  for (int i = 0; i <= 3; i++) {
    is >> _measurement[i];
  }
  for (int i = 0; i <= 2; i++)
    for (int j = i; j <= 2; j++) {
      is >> information()(i, j);
      if (i != j)
        information()(j, i) = information()(i, j);
    }
  return true;
}

bool EdgeStereoSE3ProjectXYZ::write(std::ostream &os) const {

  for (int i = 0; i <= 3; i++) {
    os << measurement()[i] << " ";
  }

  for (int i = 0; i <= 2; i++)
    for (int j = i; j <= 2; j++) {
      os << " " << information()(i, j);
    }
  return os.good();
}

void EdgeStereoSE3ProjectXYZ::linearizeOplus() {
  VertexSE3Expmap *vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAPointXYZ *vi = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
  Vector3d xyz = vi->estimate();
  Vector3d xyz_trans = T.map(xyz);

  const Matrix3d R = T.rotation().toRotationMatrix();

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z * z;

  _jacobianOplusXi(0, 0) = -fx * R(0, 0) / z + fx * x * R(2, 0) / z_2;
  _jacobianOplusXi(0, 1) = -fx * R(0, 1) / z + fx * x * R(2, 1) / z_2;
  _jacobianOplusXi(0, 2) = -fx * R(0, 2) / z + fx * x * R(2, 2) / z_2;

  _jacobianOplusXi(1, 0) = -fy * R(1, 0) / z + fy * y * R(2, 0) / z_2;
  _jacobianOplusXi(1, 1) = -fy * R(1, 1) / z + fy * y * R(2, 1) / z_2;
  _jacobianOplusXi(1, 2) = -fy * R(1, 2) / z + fy * y * R(2, 2) / z_2;

  _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - bf * R(2, 0) / z_2;
  _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) - bf * R(2, 1) / z_2;
  _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2) - bf * R(2, 2) / z_2;

  _jacobianOplusXj(0, 0) = x * y / z_2 * fx;
  _jacobianOplusXj(0, 1) = -(1 + (x * x / z_2)) * fx;
  _jacobianOplusXj(0, 2) = y / z * fx;
  _jacobianOplusXj(0, 3) = -1. / z * fx;
  _jacobianOplusXj(0, 4) = 0;
  _jacobianOplusXj(0, 5) = x / z_2 * fx;

  _jacobianOplusXj(1, 0) = (1 + y * y / z_2) * fy;
  _jacobianOplusXj(1, 1) = -x * y / z_2 * fy;
  _jacobianOplusXj(1, 2) = -x / z * fy;
  _jacobianOplusXj(1, 3) = 0;
  _jacobianOplusXj(1, 4) = -1. / z * fy;
  _jacobianOplusXj(1, 5) = y / z_2 * fy;

  _jacobianOplusXj(2, 0) = _jacobianOplusXj(0, 0) - bf * y / z_2;
  _jacobianOplusXj(2, 1) = _jacobianOplusXj(0, 1) + bf * x / z_2;
  _jacobianOplusXj(2, 2) = _jacobianOplusXj(0, 2);
  _jacobianOplusXj(2, 3) = _jacobianOplusXj(0, 3);
  _jacobianOplusXj(2, 4) = 0;
  _jacobianOplusXj(2, 5) = _jacobianOplusXj(0, 5) - bf / z_2;
}

bool EdgeSE3ProjectXYZOnlyPose::read(std::istream &is) {
  for (int i = 0; i < 2; i++) {
    is >> _measurement[i];
  }
  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      is >> information()(i, j);
      if (i != j)
        information()(j, i) = information()(i, j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream &os) const {

  for (int i = 0; i < 2; i++) {
    os << measurement()[i] << " ";
  }

  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      os << " " << information()(i, j);
    }
  return os.good();
}

void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
  VertexSE3Expmap *vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double invz = 1.0 / xyz_trans[2];
  double invz_2 = invz * invz;

  _jacobianOplusXi(0, 0) = x * y * invz_2 * fx;
  _jacobianOplusXi(0, 1) = -(1 + (x * x * invz_2)) * fx;
  _jacobianOplusXi(0, 2) = y * invz * fx;
  _jacobianOplusXi(0, 3) = -invz * fx;
  _jacobianOplusXi(0, 4) = 0;
  _jacobianOplusXi(0, 5) = x * invz_2 * fx;

  _jacobianOplusXi(1, 0) = (1 + y * y * invz_2) * fy;
  _jacobianOplusXi(1, 1) = -x * y * invz_2 * fy;
  _jacobianOplusXi(1, 2) = -x * invz * fy;
  _jacobianOplusXi(1, 3) = 0;
  _jacobianOplusXi(1, 4) = -invz * fy;
  _jacobianOplusXi(1, 5) = y * invz_2 * fy;
}

Vector2d EdgeSE3ProjectXYZOnlyPose::cam_project(const Vector3d &trans_xyz) const {
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0] * fx + cx;
  res[1] = proj[1] * fy + cy;
  return res;
}

Vector3d EdgeStereoSE3ProjectXYZOnlyPose::cam_project(const Vector3d &trans_xyz) const {
  const float invz = 1.0f / trans_xyz[2];
  Vector3d res;
  res[0] = trans_xyz[0] * invz * fx + cx;
  res[1] = trans_xyz[1] * invz * fy + cy;
  res[2] = res[0] - bf * invz;
  return res;
}

bool EdgeStereoSE3ProjectXYZOnlyPose::read(std::istream &is) {
  for (int i = 0; i <= 3; i++) {
    is >> _measurement[i];
  }
  for (int i = 0; i <= 2; i++)
    for (int j = i; j <= 2; j++) {
      is >> information()(i, j);
      if (i != j)
        information()(j, i) = information()(i, j);
    }
  return true;
}

bool EdgeStereoSE3ProjectXYZOnlyPose::write(std::ostream &os) const {

  for (int i = 0; i <= 3; i++) {
    os << measurement()[i] << " ";
  }

  for (int i = 0; i <= 2; i++)
    for (int j = i; j <= 2; j++) {
      os << " " << information()(i, j);
    }
  return os.good();
}

void EdgeStereoSE3ProjectXYZOnlyPose::linearizeOplus() {
  VertexSE3Expmap *vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double invz = 1.0 / xyz_trans[2];
  double invz_2 = invz * invz;

  _jacobianOplusXi(0, 0) = x * y * invz_2 * fx;
  _jacobianOplusXi(0, 1) = -(1 + (x * x * invz_2)) * fx;
  _jacobianOplusXi(0, 2) = y * invz * fx;
  _jacobianOplusXi(0, 3) = -invz * fx;
  _jacobianOplusXi(0, 4) = 0;
  _jacobianOplusXi(0, 5) = x * invz_2 * fx;

  _jacobianOplusXi(1, 0) = (1 + y * y * invz_2) * fy;
  _jacobianOplusXi(1, 1) = -x * y * invz_2 * fy;
  _jacobianOplusXi(1, 2) = -x * invz * fy;
  _jacobianOplusXi(1, 3) = 0;
  _jacobianOplusXi(1, 4) = -invz * fy;
  _jacobianOplusXi(1, 5) = y * invz_2 * fy;

  _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - bf * y * invz_2;
  _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) + bf * x * invz_2;
  _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2);
  _jacobianOplusXi(2, 3) = _jacobianOplusXi(0, 3);
  _jacobianOplusXi(2, 4) = 0;
  _jacobianOplusXi(2, 5) = _jacobianOplusXi(0, 5) - bf * invz_2;
}









bool EdgeProjectD3VO::write(std::ostream& os) const  {
    // os << _cam->id() << " ";
    // for (int i=0; i<2; i++){
    //     os << measurement()[i] << " ";
    // }

    // for (int i=0; i<2; i++)
    //     for (int j=i; j<2; j++){
    //     os << " " <<  information()(i,j);
    //     }
    return os.good();
}

bool EdgeProjectD3VO::read(std::istream& is) {
    // int paramId;
    // is >> paramId;
    // setParameterId(0, paramId);

    // for (int i=0; i<2; i++){
    //     is >> _measurement[i];
    // }
    // for (int i=0; i<2; i++)
    //     for (int j=i; j<2; j++) {
    //     is >> information()(i,j);
    //     if (i!=j)
    //         information()(j,i)=information()(i,j);
    //     }
    return true;
}

void EdgeProjectD3VO::computeError(){
    const VertexD3VOPointDepth* pt = static_cast<const VertexD3VOPointDepth*>(_vertices[0]);
    const VertexD3VOFramePose* dest_frame = static_cast<const VertexD3VOFramePose*>(_vertices[1]);
    const VertexD3VOFramePose* host_frame = static_cast<const VertexD3VOFramePose*>(_vertices[2]);
    const CameraParameters* cam = static_cast<const CameraParameters*>(parameter(0));

    Vector2D p_prime = cam->cam_map(dest_frame->estimate() * host_frame->estimate().inverse() * cam->cam_unmap(pt->uv, pt->estimate()));

    // Obtain pixel intensity for host and destination frames for points p and p'
    double* host_img = (double*) host_frame->pixel_inten.ptr;
    double* dest_img = (double*) dest_frame->pixel_inten.ptr;

    // Have to index into array manually...
    int X = host_frame->pixel_inten.shape[0];
    int Y = host_frame->pixel_inten.shape[1];
    int Z = host_frame->pixel_inten.shape[2];
    int host_base_idx = (int)pt->uv(0) * Y * Z + Z * (int)pt->uv(1);
    int dest_base_idx = (int)p_prime(0) * Y * Z + Z * (int)p_prime(1);

    if(host_base_idx + 2 >= X*Y*Z || dest_base_idx + 2 >= X*Y*Z || host_base_idx < 0 || dest_base_idx < 0) {
        _error.setZero();
        out_of_bounds = true;
        return;
    }
    else{
        out_of_bounds = false;
    }

    Vector3D host_inten(host_img[host_base_idx], host_img[host_base_idx+1], host_img[host_base_idx+2]);
    Vector3D dest_inten(dest_img[dest_base_idx], dest_img[dest_base_idx+1], dest_img[dest_base_idx+2]);
    _error = dest_inten - host_inten;
}



void EdgeProjectD3VO::linearizeOplus(){
    // General resource for DSO Jacobian derivation
    // https://openaccess.thecvf.com/content_ECCV_2018/papers/David_Schubert_Direct_Sparse_Odometry_ECCV_2018_paper.pdf (pg8) has better Jacobian breakdown
    // https://github.com/edward0im/stereo-dso-g2o/blob/master/KR_dso_review_with_codes.pdf detailed walk through, but in Korean
    
    if(out_of_bounds) {
        // Out of bounds reprojection detected, cause optimizer to ignore this edge
        Eigen::Matrix<double,1,6> frame_error;
        Eigen::Matrix<double,1,1> depth_error;
        depth_error.setZero();
        frame_error.setZero(); 
        _jacobianOplus[0] = depth_error;
        _jacobianOplus[1] = frame_error;
        _jacobianOplus[2] = frame_error;
        out_of_bounds = false;
        return;
    }
    
    const VertexD3VOPointDepth* pt = static_cast<const VertexD3VOPointDepth*>(_vertices[0]);
    const VertexD3VOFramePose* dest_frame = static_cast<const VertexD3VOFramePose*>(_vertices[1]);
    const VertexD3VOFramePose* host_frame = static_cast<const VertexD3VOFramePose*>(_vertices[2]);
    const CameraParameters* cam = static_cast<const CameraParameters*>(parameter(0));


    // Finite difference approximation to the image gradient (J_I = (\partial I_j) / (\partial p'))
    int X = dest_frame->pixel_inten.shape[0];
    int Y = dest_frame->pixel_inten.shape[1];
    int Z = host_frame->pixel_inten.shape[2];
    Isometry3D T_host_dest = dest_frame->estimate() * host_frame->estimate().inverse();
    Vector3D unprojected_X = T_host_dest * cam->cam_unmap(pt->uv, pt->estimate());
    Vector2D p_prime = cam->cam_map(unprojected_X);
    int p_prime_u = (int) p_prime(0);
    int p_prime_v = (int) p_prime(1);
    
    // Validate point not near the edge of the image
    Vector2D J_Ij;
    if(p_prime_u + 1 < X && p_prime_u - 1 >= 0 && p_prime_v - 1 >= 0 && p_prime_v + 1 < Y) {
        // Note: top left of image is (0, 0)
        double* dest_img = (double*) dest_frame->pixel_inten.ptr;
        int dest_base_idx = p_prime_u * Y * Z + Z * p_prime_v;
        double dx_r = dest_img[dest_base_idx + Z] - dest_img[dest_base_idx - Z];        // right - left;
        double dy_r = dest_img[dest_base_idx + Y*Z] - dest_img[dest_base_idx - Y*Z];    // bottom - top;
        double dx_g = dest_img[dest_base_idx + Z + 1] - dest_img[dest_base_idx - Z + 1];  
        double dy_g = dest_img[dest_base_idx + Y*Z + 1] - dest_img[dest_base_idx - Y*Z + 1];   
        double dx_b = dest_img[dest_base_idx + Z + 2] - dest_img[dest_base_idx - Z + 2]; 
        double dy_b = dest_img[dest_base_idx + Y*Z + 2] - dest_img[dest_base_idx - Y*Z + 2];  

        // Gradient of RGB image is not well-defined, just take the average over gradients from all channels
        J_Ij = Vector2D((dx_r + dx_g + dx_b) / 3, (dy_r + dy_g + dy_b) / 3);
    }
    else {
        // Out of bounds reprojection detected / failure to calculate image gradient, cause optimizer to ignore this edge
        Eigen::Matrix<double,1,6> frame_error;
        Eigen::Matrix<double,1,1> depth_error;
        depth_error.setZero();
        frame_error.setZero(); 
        _jacobianOplus[0] = depth_error;
        _jacobianOplus[1] = frame_error;
        _jacobianOplus[2] = frame_error;
        return;
    }

    // d p' / dX   (where X is the unprojected and rotated point in 3D space)
    double depth = pt->estimate();
    Eigen::Matrix<double,2,3> dprime_dX = d_proj_d_y(cam->focal_length, unprojected_X);

    // d X / d depth
    Vector3D unprojected = cam->cam_unmap(pt->uv, depth) / depth;
    Vector3D dx_dd = unprojected_X / depth;          // rather than recomputing, just subtract by depth since it is a scalar and commutes through matrix-vector operations

    // d p' / d depth
    Vector2D dprime_ddepth = dprime_dX * dx_dd;

    // Full depth Jacobian (d I_J / d depth) = (d I_j / d p') * (d p' / d depth) --> need a dot product
    Eigen::Matrix<double,1,1> J_depth = J_Ij.transpose() * dprime_ddepth;

    // Calculate relative pose Jacobian wrt T_j * T_i^{-1}
    // https://github.com/edward0im/stereo-dso-g2o/blob/master/KR_dso_review_with_codes.pdf (slide 50)
    // http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf Chapter 7
    Eigen::Matrix<double,3,6> dX_drelative = d_expy_d_y(unprojected_X);

    // d X / d (T_j * T_i^{-1})
    Eigen::Matrix<double, 2, 6> J_relative_pose = dprime_dX * dX_drelative;

    // Separate this relative pose Jacobian (T_j * T_i^{-1}) into Jacobians for T_i and T_j using the adjoint transformation
    // https://ethaneade.com/lie.pdf (pg4/5)
    // https://www.cnblogs.com/JingeTU/p/8306727.html 
    // https://www.ethaneade.com/latex2html/lie/node17.html --> Adjoint of a matrix in SE(3)
    Eigen::Matrix<double, 3, 3> R_th = T_host_dest.rotation().matrix();  
    Eigen::Matrix<double, 1, 3> t_th = T_host_dest.translation();
    Eigen::Matrix<double, 3, 3> cross_t_th = skew(t_th);     // Eigen has no unary cross product operator??
    Eigen::Matrix<double, 3, 3> zero;
    zero.setZero();

    // Assemble Ad_{T_{th}}
    Eigen::Matrix<double,6,6> host_adjoint;
    host_adjoint << R_th, cross_t_th * R_th,
                    zero, R_th;

    // Calculate separated Jacobians for host and target frames
    Eigen::Matrix<double,2,6> J_host_p_prime = J_relative_pose * -host_adjoint;   // (d p_prime / d(target -> host)) (d(target -> host) / d host)
    Eigen::Matrix<double,2,6> J_dest_p_prime = J_relative_pose;   // adjoint of target frame is the identity

    // Calculate full host and target frame Jacobians by incorporating image pixel gradients
    Eigen::Matrix<double,1,6> J_host_T = J_Ij.transpose() * J_host_p_prime;
    Eigen::Matrix<double,1,6> J_dest_T = J_Ij.transpose() * J_dest_p_prime;

    // Order of these Jacobians is the same as the order of _vertices (depth, dest frame, host frame)
    _jacobianOplus[0] = J_depth;
    _jacobianOplus[1] = J_dest_T;
    _jacobianOplus[2] = J_host_T;
}


} // end namespace
