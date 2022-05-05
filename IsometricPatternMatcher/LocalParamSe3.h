/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ceres/local_parameterization.h>

namespace surreal_opensource {

// A Ceres local parameterization for SE3 parameter blocks that are arranged in
// the same order as Sophus::SE3d types with a unit-norm quaternion and
// translation as:
//
//   qx qy qz qw tx ty tz
//
// By providing this local parameterization to Ceres the parameters are
// optimized on the SE3 manifold. This is desirable since it ensures each update
// to the parameters remains a valid SE3, and it is better for numeric and
// computationally more efficient during optimziation. To use this local
// parameterization, you can do the following:
//
//   ceres::Problem problem;
//   Sophus::SE3d T_bar_foo = ...;
//   problem.AddParameterBlock(T_bar_foo.data(),
//                             Sophus::SE3d::num_parameters,
//                             new surreal::math::LocalParamSE3());
class LocalParamSe3 : public ceres::LocalParameterization {
 public:
  LocalParamSe3() = default;

  // SE3 plus operation for Ceres
  //
  //  T * exp(x)
  //
  bool Plus(const double* T, const double* delta,
            double* T_plus_delta) const override;

  // Jacobian of SE3 plus operation for Ceres
  //
  // Dx T * exp(x)  with  x=0
  //
  bool ComputeJacobian(const double* T, double* jacobian) const override;

  int GlobalSize() const override;
  int LocalSize() const override;
};
}  // namespace surreal_opensource
