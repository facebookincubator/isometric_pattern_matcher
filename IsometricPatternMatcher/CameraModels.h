/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace surreal_opensource {

// Helper Newton functions for camera unprojection
namespace {

constexpr float kFloatTolerance = 1e-5;
constexpr float kDoubleTolerance = 1e-7;

template <typename T>
inline double IgnoreJetInfinitesimal(const T& j) {
  static_assert(std::is_same<decltype(j.a), double>::value ||
                    std::is_same<decltype(j.a), float>::value,
                "T should be a ceres jet");
  return j.a;
}

inline double IgnoreJetInfinitesimal(double j) { return j; }

inline float IgnoreJetInfinitesimal(float j) { return j; }

template <typename T>
constexpr float getConvergenceTolerance() {
  if (std::is_same<T, ceres::Jet<float, T::DIMENSION>>::value) {
    return kFloatTolerance;
  }
  if (std::is_same<T, ceres::Jet<double, T::DIMENSION>>::value) {
    // largest number that passes project / unproject test to within 1e-8 pixels
    // for all models.
    return kDoubleTolerance;
  }
}

template <>
constexpr float getConvergenceTolerance<float>() {
  return kFloatTolerance;
}

template <>
constexpr float getConvergenceTolerance<double>() {
  return kDoubleTolerance;
}

template <typename T>
inline bool hasConverged(const T& step) {
  using std::abs;
  return abs(IgnoreJetInfinitesimal(step)) < getConvergenceTolerance<T>();
}

template <typename T>
inline T initTheta(const T& r) {
  using std::sqrt;
  return sqrt(r);
}

// Dehomogenize / project input
template <typename Derived>
EIGEN_STRONG_INLINE static Eigen::Matrix<typename Derived::Scalar,
                                         Derived::RowsAtCompileTime - 1, 1>
Project(const Eigen::MatrixBase<Derived>& v) {
  return v.template head<Derived::RowsAtCompileTime - 1>() /
         v[Derived::RowsAtCompileTime - 1];
}

}  // namespace

// Pinhole Projection Model
//
// parameters = fx, fy, cx, cy
class PinholeProjection {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int kNumParams = 4;
  static constexpr char kName[] = "Pinhole";
  static constexpr char kDescription[] = "fx, fy, cx, cy";
  static constexpr int kNumDistortionParams = 0;
  static constexpr int kFocalXIdx = 0;
  static constexpr int kFocalYIdx = 1;
  static constexpr int kPrincipalPointColIdx = 2;
  static constexpr int kPrincipalPointRowIdx = 3;
  static constexpr bool kIsFisheye = false;
  static constexpr bool kHasAnalyticalProjection = true;

  // Takes in 3-point ``pointOptical`` in the local reference frame of the
  // camera and projects it onto the image plan.
  //
  // Precondition: pointOptical.z() != 0.
  //
  // Return 2-point in the image plane.
  //
  template <class D, class DP,
            class DJ1 = Eigen::Matrix<typename D::Scalar, 2, 3>,
            class DJ2 = Eigen::Matrix<typename D::Scalar, 2, kNumParams>>
  static Eigen::Matrix<typename D::Scalar, 2, 1> project(
      const Eigen::MatrixBase<D>& pointOptical,
      const Eigen::MatrixBase<DP>& params,
      Eigen::MatrixBase<DJ1>* d_point = nullptr,
      Eigen::MatrixBase<DJ2>* d_params = nullptr) {
    using T = typename D::Scalar;

    static_assert(D::RowsAtCompileTime == 3 && D::ColsAtCompileTime == 1,
                  "THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE");
    static_assert(
        DP::ColsAtCompileTime == 1 && (DP::RowsAtCompileTime == kNumParams ||
                                       DP::RowsAtCompileTime == Eigen::Dynamic),
        "THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE");
    static_assert((DJ1::ColsAtCompileTime == 3 ||
                   DJ1::ColsAtCompileTime == Eigen::Dynamic) &&
                      (DJ1::RowsAtCompileTime == 2 ||
                       DJ1::RowsAtCompileTime == Eigen::Dynamic),
                  "THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE");
    static_assert((DJ2::ColsAtCompileTime == kNumParams ||
                   DJ2::ColsAtCompileTime == Eigen::Dynamic) &&
                      (DJ2::RowsAtCompileTime == 2 ||
                       DJ2::RowsAtCompileTime == Eigen::Dynamic),
                  "THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE");

    SOPHUS_ENSURE(pointOptical.z() != T(0), "z(%) must not be zero.",
                  pointOptical.z());

    // Focal length and principal point
    const Eigen::Matrix<T, 2, 1> ff = params.template head<2>();
    const Eigen::Matrix<T, 2, 1> pp = params.template segment<2>(2);

    const Eigen::Matrix<T, 2, 1> px =
        (Project(pointOptical).array() * ff.array()).matrix() + pp;

    if (d_point) {
      const T oneOverZ = T(1) / pointOptical(2);

      (*d_point)(0, 0) = ff(0) * oneOverZ;
      (*d_point)(0, 1) = 0.0;
      (*d_point)(0, 2) = -(*d_point)(0, 0) * pointOptical(0) * oneOverZ;
      (*d_point)(1, 0) = 0.0;
      (*d_point)(1, 1) = ff(1) * oneOverZ;
      (*d_point)(1, 2) = -(*d_point)(1, 1) * pointOptical(1) * oneOverZ;
    }
    if (d_params) {
      (*d_params)(0, 0) = pointOptical(0) / pointOptical(2);
      (*d_params)(0, 1) = T(0.0);
      (*d_params)(0, 2) = T(1.0);
      (*d_params)(0, 3) = T(0.0);
      (*d_params)(1, 0) = T(0.0);
      (*d_params)(1, 1) = pointOptical(1) / pointOptical(2);
      (*d_params)(1, 2) = T(0.0);
      (*d_params)(1, 3) = T(1.0);
    }

    return px;
  }

  // Takes in 2-point ``uv`` in the image plane of the camera and unprojects it
  // into the reference frame of the camera.
  //
  // This function is the inverse of ``project``. In particular it holds that
  //
  // X = unproject(project(X))     [for X=(x,y,z) in R^3, z>0]
  //
  //  and
  //
  // x = project(unproject(s*x))   [for s!=0 and x=(u,v) in R^2]
  //
  // Return 3-point in the camera frame with z = 1.
  //
  template <typename D, typename DP>
  static Eigen::Matrix<typename D::Scalar, 3, 1> unproject(
      const Eigen::MatrixBase<D>& uvPixel,
      const Eigen::MatrixBase<DP>& params) {
    EIGEN_STATIC_ASSERT(D::RowsAtCompileTime == 2 && D::ColsAtCompileTime == 1,
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    EIGEN_STATIC_ASSERT(
        DP::ColsAtCompileTime == 1 && (DP::RowsAtCompileTime == kNumParams ||
                                       DP::RowsAtCompileTime == Eigen::Dynamic),
        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    using T = typename D::Scalar;

    // Unprojection
    const T fu = params[0];
    const T fv = params[1];
    const T u0 = params[2];
    const T v0 = params[3];

    const T un = (uvPixel(0) - u0) / fu;
    const T vn = (uvPixel(1) - v0) / fv;

    return Eigen::Matrix<T, 3, 1>(un, vn, T(1.0));
  }
};

// Kannala and Brandt Like 'Generic' Projection Model
// http://cs.iupui.edu/~tuceryan/pdf-repository/Kannala2006.pdf
// https://april.eecs.umich.edu/wiki/Camera_suite
// NOTE, our implementation presents some important differences wrt the original
// paper:
// - k1 in eq(6) in the paper is fixed to 1.0, so k0 here is k2 in the paper
//   (for the same reason we have only x4 k parameters instead of x5 in the
//   paper, for order theta^9)
//
// In this projection model the points behind the camera are projected in a way
// that the optimization cost function is continuous, therefore the optimization
// problem can be nicely solved. This option should be used during calibration.
//
// parameters = fx, fy, cx, cy, kb0, kb1, kb2, kb3
class KannalaBrandtK3Projection {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int kNumParams = 8;
  static constexpr char kName[] = "KannalaBrandtK3";
  static constexpr char kDescription[] = "fx, fy, cx, cy, kb0, kb1, kb2, kb3";
  static constexpr int kNumDistortionParams = 4;
  static constexpr int kFocalXIdx = 0;
  static constexpr int kFocalYIdx = 1;
  static constexpr int kPrincipalPointColIdx = 2;
  static constexpr int kPrincipalPointRowIdx = 3;
  static constexpr bool kIsFisheye = true;
  static constexpr bool kHasAnalyticalProjection = true;

  // Takes in 3-point ``pointOptical`` in the local reference frame of the
  // camera and projects it onto the image plan.
  //
  // Precondition: pointOptical.z() != 0.
  //
  // Return 2-point in the image plane.
  //
  template <class D, class DP,
            class DJ1 = Eigen::Matrix<typename D::Scalar, 2, 3>,
            class DJ2 = Eigen::Matrix<typename D::Scalar, 2, kNumParams>>
  static Eigen::Matrix<typename D::Scalar, 2, 1> project(
      const Eigen::MatrixBase<D>& pointOptical,
      const Eigen::MatrixBase<DP>& params,
      Eigen::MatrixBase<DJ1>* d_point = nullptr,
      Eigen::MatrixBase<DJ2>* d_params = nullptr) {
    using T = typename D::Scalar;

    static_assert(D::RowsAtCompileTime == 3 && D::ColsAtCompileTime == 1,
                  "THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE");
    static_assert(
        DP::ColsAtCompileTime == 1 && (DP::RowsAtCompileTime == kNumParams ||
                                       DP::RowsAtCompileTime == Eigen::Dynamic),
        "THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE");
    static_assert((DJ1::ColsAtCompileTime == 3 ||
                   DJ1::ColsAtCompileTime == Eigen::Dynamic) &&
                      (DJ1::RowsAtCompileTime == 2 ||
                       DJ1::RowsAtCompileTime == Eigen::Dynamic),
                  "THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE");
    static_assert((DJ2::ColsAtCompileTime == kNumParams ||
                   DJ2::ColsAtCompileTime == Eigen::Dynamic) &&
                      (DJ2::RowsAtCompileTime == 2 ||
                       DJ2::RowsAtCompileTime == Eigen::Dynamic),
                  "THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE");

    SOPHUS_ENSURE(pointOptical.z() != T(0), "z(%) must not be zero.",
                  pointOptical.z());

    // Focal length and principal point
    const Eigen::Matrix<T, 2, 1> ff = params.template head<2>();
    const Eigen::Matrix<T, 2, 1> pp = params.template segment<2>(2);

    const T k0 = params[4];
    const T k1 = params[5];
    const T k2 = params[6];
    const T k3 = params[7];

    const T radiusSquared =
        pointOptical(0) * pointOptical(0) + pointOptical(1) * pointOptical(1);
    using std::atan2;
    using std::sqrt;

    if (radiusSquared > Sophus::Constants<T>::epsilon()) {
      const T radius = sqrt(radiusSquared);
      const T radiusInverse = T(1.0) / radius;
      const T theta = atan2(radius, pointOptical(2));
      const T theta2 = theta * theta;
      const T theta4 = theta2 * theta2;
      const T theta6 = theta4 * theta2;
      const T theta8 = theta4 * theta4;
      const T rDistorted = theta * (T(1.0) + k0 * theta2 + k1 * theta4 +
                                    k2 * theta6 + k3 * theta8);
      const T scaling = rDistorted * radiusInverse;

      if (d_point) {
        const T xSquared = pointOptical(0) * pointOptical(0);
        const T ySquared = pointOptical(1) * pointOptical(1);
        const T normSquared = pointOptical(2) * pointOptical(2) + radiusSquared;
        const T rDistortedDerivative =
            T(1.0) + T(3.0) * k0 * theta2 + T(5.0) * k1 * theta4 +
            T(7.0) * k2 * theta6 + T(9.0) * k3 * theta8;
        const T x13 =
            pointOptical(2) * rDistortedDerivative / normSquared - scaling;
        const T rDistortedDerivativeNormalized =
            rDistortedDerivative / normSquared;
        const T x20 = pointOptical(2) * rDistortedDerivative /
                      (normSquared)-radiusInverse * rDistorted;

        (*d_point)(0, 0) = xSquared / radiusSquared * x20 + scaling;
        (*d_point)(0, 1) =
            pointOptical(1) * x13 * pointOptical(0) / radiusSquared;
        (*d_point)(0, 2) = -pointOptical(0) * rDistortedDerivativeNormalized;
        (*d_point)(1, 0) = (*d_point)(0, 1);
        (*d_point)(1, 1) = ySquared / radiusSquared * x20 + scaling;
        (*d_point)(1, 2) = -pointOptical(1) * rDistortedDerivativeNormalized;

        // toDenseMatrix() is needed for CUDA to explicitly know the matrix
        // dimensions
        (*d_point) = ff.asDiagonal().toDenseMatrix() * (*d_point);
      }
      using std::pow;
      if (d_params) {
        const T xScaled = pointOptical[0] * params[0] * radiusInverse;
        const T yScaled = pointOptical[1] * params[1] * radiusInverse;

        const T theta3 = theta * theta2;
        const T theta5 = theta3 * theta2;
        const T theta7 = theta5 * theta2;
        const T theta9 = theta7 * theta2;

        (*d_params)(0, 0) = pointOptical[0] * scaling;
        (*d_params)(0, 1) = T(0.0);
        (*d_params)(0, 2) = T(1.0);
        (*d_params)(0, 3) = T(0.0);
        (*d_params)(0, 4) = xScaled * theta3;
        (*d_params)(0, 5) = xScaled * theta5;
        (*d_params)(0, 6) = xScaled * theta7;
        (*d_params)(0, 7) = xScaled * theta9;
        (*d_params)(1, 0) = T(0.0);
        (*d_params)(1, 1) = pointOptical[1] * scaling;
        (*d_params)(1, 2) = T(0.0);
        (*d_params)(1, 3) = T(1.0);
        (*d_params)(1, 4) = yScaled * theta3;
        (*d_params)(1, 5) = yScaled * theta5;
        (*d_params)(1, 6) = yScaled * theta7;
        (*d_params)(1, 7) = yScaled * theta9;
      }

      const Eigen::Matrix<T, 2, 1> px =
          scaling * ff.cwiseProduct(pointOptical.template head<2>()) + pp;

      return px;
    } else {
      // linearize r around radius=0
      if (d_point) {
        const T z2 = pointOptical(2) * pointOptical(2);
        // clang-format off
        (*d_point) << ff.x() / pointOptical(2), T(0.0), -ff.x() * pointOptical(0) / z2,
                      T(0.0), ff.y() / pointOptical(2), -ff.y() * pointOptical(1) / z2;
        // clang-format on
      }
      if (d_params) {
        (*d_params)(0, 0) = pointOptical(0) / pointOptical(2);
        (*d_params)(0, 1) = T(0.0);
        (*d_params)(0, 2) = T(1.0);
        (*d_params)(0, 3) = T(0.0);

        (*d_params)(1, 0) = T(0.0);
        (*d_params)(1, 1) = pointOptical(1) / pointOptical(2);
        (*d_params)(1, 2) = T(0.0);
        (*d_params)(1, 3) = T(1.0);
        (*d_params).template rightCols<4>().setZero();
      }
      const Eigen::Matrix<T, 2, 1> px =
          ff.cwiseProduct(pointOptical.template head<2>()) / pointOptical(2) +
          pp;

      return px;
    }
  }

  // Takes in 2-point ``uv`` in the image plane of the camera and unprojects it
  // into the reference frame of the camera.
  //
  // This function is the inverse of ``project``. In particular it holds that
  //
  // X = unproject(project(X))     [for X=(x,y,z) in R^3, z>0]
  //
  //  and
  //
  // x = project(unproject(s*x))   [for s!=0 and x=(u,v) in R^2]
  //
  // Return 3-point in the camera frame with z = 1.
  //
  template <typename D, typename DP>
  static Eigen::Matrix<typename D::Scalar, 3, 1> unproject(
      const Eigen::MatrixBase<D>& uvPixel,
      const Eigen::MatrixBase<DP>& params) {
    EIGEN_STATIC_ASSERT(D::RowsAtCompileTime == 2 && D::ColsAtCompileTime == 1,
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    EIGEN_STATIC_ASSERT(
        DP::ColsAtCompileTime == 1 && (DP::RowsAtCompileTime == kNumParams ||
                                       DP::RowsAtCompileTime == Eigen::Dynamic),
        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    using T = typename D::Scalar;

    // Unprojection
    const T fu = params[0];
    const T fv = params[1];
    const T u0 = params[2];
    const T v0 = params[3];

    const T k0 = params[4];
    const T k1 = params[5];
    const T k2 = params[6];
    const T k3 = params[7];

    const T un = (uvPixel(0) - u0) / fu;
    const T vn = (uvPixel(1) - v0) / fv;
    const T rth2 = un * un + vn * vn;

    if (rth2 <
        Sophus::Constants<T>::epsilon() * Sophus::Constants<T>::epsilon()) {
      return Eigen::Matrix<T, 3, 1>(un, vn, T(1.0));
    }

    const T rth = sqrt(rth2);

    // Use Newtons method to solve for theta, 50 iterations max
    T th = initTheta(rth);
    for (int i = 0; i < 50; ++i) {
      const T th2 = th * th;
      const T th4 = th2 * th2;
      const T th6 = th4 * th2;
      const T th8 = th4 * th4;

      const T thd = th * (T(1.0) + k0 * th2 + k1 * th4 + k2 * th6 + k3 * th8);

      const T d_thd_wtr_th = T(1.0) + T(3.0) * k0 * th2 + T(5.0) * k1 * th4 +
                             T(7.0) * k2 * th6 + T(9.0) * k3 * th8;

      const T step = (thd - rth) / d_thd_wtr_th;
      th -= step;
      if (hasConverged(step)) {
        break;
      }
    }

    using std::tan;
    T radiusUndistorted = tan(th);

    if (radiusUndistorted < T(0.0)) {
      return Eigen::Matrix<T, 3, 1>(-radiusUndistorted * un / rth,
                                    -radiusUndistorted * vn / rth, T(-1.0));
    }
    return Eigen::Matrix<T, 3, 1>(radiusUndistorted * un / rth,
                                  radiusUndistorted * vn / rth, T(1.0));
  }
};

}  // namespace surreal_opensource
