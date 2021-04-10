// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <IsometricPatternMatcher/LocalParamSe3.h>

#include <sophus/se3.hpp>
#include <Eigen/Core>

namespace surreal_opensource {

bool LocalParamSe3::Plus(const double* T_raw, const double* delta_raw, double* T_plus_delta_raw)
    const {
  const Eigen::Map<const Sophus::SE3d> T(T_raw);
  const Eigen::Map<const Sophus::SE3d::Tangent> delta(delta_raw);
  Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
  T_plus_delta = T * Sophus::SE3d::exp(delta);
  return true;
}

bool LocalParamSe3::ComputeJacobian(const double* T_raw, double* jacobian_raw) const {
  using JacobianMatrix =
      Eigen::Matrix<double, Sophus::SE3d::num_parameters, Sophus::SE3d::DoF, Eigen::RowMajor>;

  Eigen::Map<const Sophus::SE3d> T(T_raw);
  Eigen::Map<JacobianMatrix> jacobian(jacobian_raw);
  jacobian = T.Dx_this_mul_exp_x_at_0();
  return true;
}

int LocalParamSe3::GlobalSize() const {
  return Sophus::SE3d::num_parameters;
}

int LocalParamSe3::LocalSize() const {
  return Sophus::SE3d::DoF;
}

} // namespace surreal_opensource
