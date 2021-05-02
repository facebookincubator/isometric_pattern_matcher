// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <IsometricPatternMatcher/CameraModels.h>
#include <IsometricPatternMatcher/ray.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <Eigen/Core>
#include <sophus/se3.hpp>

#pragma once
// to estimate T_camera_target to transfer the image points to target space that
// the distance between neighrous = 1 please refer to this doc for more details:
// https://docs.google.com/document/d/1qLtr-ibhi1K-JtlnfyqryOUAPMmp5dpB1b3hoIzhtXA/edit
namespace surreal_opensource {

class isometricPatternPoseCost {
 public:
  isometricPatternPoseCost(const Ray3d rayInCamera,
                           const Ray3d rayNeighbourInCamera,
                           const Sophus::Plane3d& plane,
                           const double& transferSpacing)
      : rayInCamera_(rayInCamera),
        rayNeighbourInCamera_(rayNeighbourInCamera),
        plane_(plane),
        transferSpacing_(transferSpacing) {}

  template <typename Scalar>
  bool operator()(const Scalar* const paramTargetPose,
                  Scalar* residuals) const {
    const Eigen::Map<const Sophus::SE3<Scalar>> T_camera_target(
        paramTargetPose);
    Ray3<Scalar> rayInTarget =
        T_camera_target.inverse() * rayInCamera_.template cast<Scalar>();
    Ray3<Scalar> rayNeighbourInTarget =
        T_camera_target.inverse() *
        rayNeighbourInCamera_.template cast<Scalar>();

    Eigen::Vector<Scalar, 3> ptTarget3d =
        rayInTarget.line().intersectionPoint(plane_.template cast<Scalar>());
    Eigen::Vector<Scalar, 3> ptNeighbourTarget3d =
        rayNeighbourInTarget.line().intersectionPoint(
            plane_.template cast<Scalar>());
    Eigen::Vector<Scalar, 3> delta = ptTarget3d - ptNeighbourTarget3d;

    residuals[0] = delta.squaredNorm() - Scalar(transferSpacing_);
    return true;
  }

 private:
  const Ray3d rayInCamera_;
  const Ray3d rayNeighbourInCamera_;
  const Sophus::Plane3d plane_;
  const double transferSpacing_;
};

class isometricPatternDistortionCost {
 public:
  isometricPatternDistortionCost(const Eigen::Vector2d& imageDots,
                                 const Eigen::Vector2d& neighbourDots,
                                 const Eigen::Vector2d& centerXY,
                                 const double& focalLength,
                                 const Sophus::Plane3d& plane)
      : imageDots_(imageDots),
        neighbourDots_(neighbourDots),
        centerXY_(centerXY),
        focalLength_(focalLength),
        plane_(plane) {}

  template <typename Scalar>
  bool operator()(const Scalar* const paramTargetPose,
                  const Scalar* const paramDistortion,
                  Scalar* residuals) const {
    Eigen::Vector<Scalar, KannalaBrandtK3Projection::kNumParams> intrinsics;
    intrinsics << Scalar(focalLength_), Scalar(focalLength_),
        Scalar(centerXY_(0)), Scalar(centerXY_(1)), paramDistortion[0],
        paramDistortion[1], paramDistortion[2], paramDistortion[3];

    Ray3<Scalar> rayInCamera = KannalaBrandtK3Projection::unproject(
        imageDots_.template cast<Scalar>(),
        intrinsics);  //.template cast<Scalar>();
    Ray3<Scalar> rayNeighbourInCamera = KannalaBrandtK3Projection::unproject(
        neighbourDots_.template cast<Scalar>(),
        intrinsics);  //.template cast<Scalar>();

    const Eigen::Map<const Sophus::SE3<Scalar>> T_camera_target(
        paramTargetPose);
    Ray3<Scalar> rayInTarget =
        T_camera_target.inverse() * rayInCamera.template cast<Scalar>();
    Ray3<Scalar> rayNeighbourInTarget =
        T_camera_target.inverse() *
        rayNeighbourInCamera.template cast<Scalar>();

    Eigen::Vector<Scalar, 3> ptTarget3d =
        rayInTarget.line().intersectionPoint(plane_.template cast<Scalar>());
    Eigen::Vector<Scalar, 3> ptNeighbourTarget3d =
        rayNeighbourInTarget.line().intersectionPoint(
            plane_.template cast<Scalar>());
    Eigen::Vector<Scalar, 3> delta = ptTarget3d - ptNeighbourTarget3d;

    residuals[0] = delta.squaredNorm() - Scalar(1.0);
    return true;
  }

 private:
  const Eigen::Vector2d imageDots_;
  const Eigen::Vector2d neighbourDots_;
  const Eigen::Vector2d centerXY_;
  const double focalLength_;
  const Sophus::Plane3d plane_;
};

}  // namespace surreal_opensource
