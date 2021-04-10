// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <sophus/se3.hpp>
#include <Eigen/Core>

#pragma once

namespace surreal_opensource {
// Represents a ray in 3d using an origin point `P` and a direction vector `dir` from the origin
// along the line.

template <typename T, int Dim>
struct Ray {
  Ray() {}
  template <typename Deriv1, typename Deriv2>
  Ray(const Eigen::MatrixBase<Deriv1>& P, const Eigen::MatrixBase<Deriv2>& dir) : P(P), dir(dir) {}

  template <typename Deriv>
  Ray(const Eigen::MatrixBase<Deriv>& dir) : P(Eigen::Matrix<T, Dim, 1>::Zero()), dir(dir) {}

  Ray normalized() const {
    return Ray(P, dir.normalized());
  }

  Eigen::Matrix<T, 3, 1> point(T t) const {
    return P + t * dir;
  }

  template <typename TNew>
  Ray<TNew, Dim> cast() const {
    return Ray<TNew, Dim>(P.template cast<TNew>(), dir.template cast<TNew>());
  }

  Eigen::ParametrizedLine<T, Dim> line() const {
    return Eigen::ParametrizedLine<T, Dim>(P, dir.normalized());
  }

  static Ray<T, Dim> Zero() {
    return Ray<T, Dim>(Eigen::Matrix<T, Dim, 1>::Zero(), Eigen::Matrix<T, Dim, 1>::Zero());
  }

  // origin
  Eigen::Matrix<T, Dim, 1> P;

  // direction
  Eigen::Matrix<T, Dim, 1> dir;
};

template <typename T>
Ray<T, 3> operator*(const Sophus::SE3<T>& T_a_b, const Ray<T, 3>& ray_b) {
  Ray<T, 3> ray_a;
  ray_a.P = T_a_b * ray_b.P;
  ray_a.dir = T_a_b.so3() * ray_b.dir;
  return ray_a;
}

template <typename T>
Ray<T, 3> operator*(const Sophus::SO3<T>& R_a_b, const Ray<T, 3>& ray_b) {
  Ray<T, 3> ray_a;
  ray_a.P = R_a_b * ray_b.P;
  ray_a.dir = R_a_b * ray_b.dir;
  return ray_a;
}

// Transforms ray in parent frame to ray in frame `bar`.
template <typename Derived, int Dim>
Ray<typename Sophus::SE3Base<Derived>::Scalar, Dim> operator*(
    const Sophus::SE3Base<Derived>& T_bar_parent,
    const Ray<typename Sophus::SE3Base<Derived>::Scalar, Dim>& ray_a) {
  using T = typename Sophus::SE3Base<Derived>::Scalar;
  return Ray<T, Dim>(T_bar_parent * ray_a.P, T_bar_parent.so3() * ray_a.dir);
}

template <class T>
using Ray3 = Ray<T, 3>;
using Ray3f = Ray<float, 3>;
using Ray3d = Ray<double, 3>;
} // namespace surreal_opensource
