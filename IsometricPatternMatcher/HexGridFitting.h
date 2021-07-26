// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ceres/solver.h>
#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace surreal_opensource {
class HexGridFitting {
  // generate the storage map of the pattern with detected image points
 public:
  HexGridFitting() {}
  // this constructor is used for one image input isometric pattern detection
  HexGridFitting(const Eigen::Matrix2Xd& imageDots,
                 const Eigen::Vector2d& centerXY, double focalLength,
                 const Eigen::VectorXd& intensity, bool ifDistort,
                 bool ifTwoShot = false, bool ifPoseMerge = false,
                 double spacing = 1.0, int numNeighboursForPoseEst = 3,
                 int numberBlock = 3, double perPointSearchRadius = 0.5,
                 int numNeighbourLayer = 2);
  // this constructor is used for two-shot isometric pattern detection
  HexGridFitting(const Eigen::Matrix2Xd& imageDots,
                 const Eigen::Vector2d& centerXY, double focalLength,
                 const Eigen::VectorXi& dotLabels, bool ifDistort,
                 bool ifTwoShot = true, bool ifPoseMerge = false,
                 double spacing = 1.0, int numNeighboursForPoseEst = 3,
                 int numberBlock = 3, double perPointSearchRadius = 0.5,
                 int numNeighbourLayer = 2);

  void Clear();

  void setParams(const Eigen::Vector2d& centerXY, double focalLength,
                 bool ifDistort, bool ifTwoShot = false,
                 bool ifPoseMerge = false, double spacing = 1.0,
                 int numNeighboursForPoseEst = 3, int numberBlock = 3,
                 double perPointSearchRadius = 0.5, int numNeighbourLayer = 2);

  void setImageDots(const Eigen::Matrix2Xd& imageDots);
  void setTransferedDots(const Eigen::Matrix2Xd& transferedDots);
  void setIntensity(const Eigen::VectorXd& intensity);
  void setIndexMap(const Eigen::MatrixXi& indexMap);

  // Each matrix2Xd stores the ith dot and its numberNeighbour-1 closest
  // neighbours found in images
  std::vector<Eigen::Matrix2Xd> imageNeighbourMatrix(int numberNeighours);

  template <typename T>
  void getSortIndx(const T& coords, std::vector<int>& idx);

  // selectIndx is used to select poses as final camera pose for merge grid
  // grow algorithm, if selectIndx = -1, we will use best pose with most inliers
  // as final camera pose
  void findPoseAndCamModel(const ceres::Solver::Options& solverOption,
                           int selectIndx = -1,
                           const Sophus::SE3d& initT_camera_target =
                               Sophus::SE3d::trans(0.1, 0.1, 0.3));

  bool findT_camera_target(const ceres::Solver::Options& solverOption,
                           const std::vector<Eigen::Matrix2Xd>& neighbourDots,
                           const std::vector<int>& sampleIndx,
                           double inlierThreshold, const Sophus::Plane3d& plane,
                           Sophus::SE3d& T_camera_target,
                           std::vector<int>& inliersIndx);

  // For KB3 camera distortion
  bool findKb3DistortionParams(
      const ceres::Solver::Options& solverOption,
      const std::vector<Eigen::Matrix2Xd>& neighbourDots,
      const std::vector<int>& sampleIndx, double inlierThreshold,
      const Sophus::Plane3d& plane, Sophus::SE3d& T_camera_target,
      std::vector<double>& distortionParams, std::vector<int>& inliersIndx);

  std::vector<int> findInliers(
      const std::vector<Eigen::Matrix2Xd>& neighbourDots,
      const Sophus::SE3d& T_camera_target,
      const Eigen::Vector4d& distortionParams, double inlierThreshold);

  Eigen::Matrix2Xd reprojectDots(const Sophus::SE3d& T_camera_target,
                                 const Eigen::Vector4d& distortionParams,
                                 const Eigen::Matrix2Xd& imageDots);

  void getStorageMap();

  // merge results from multiple poses
  void buildBinaryCode(const Eigen::Matrix3Xi& cubeCoor, int maxX, int minX,
                       int maxZ, int minZ);
  int getCubeCoordinate(const Eigen::Matrix2Xd& transferDots, int& minX,
                        int& maxX, int& minZ, int& maxZ,
                        Eigen::Matrix3Xi& cubeCoor,
                        std::vector<int>& bfsProcessSeq);
  Eigen::Matrix3Xi rotateRight60(const Eigen::Matrix3Xi& coord);
  Eigen::Matrix3Xi rotateLeft60(const Eigen::Matrix3Xi& coord);
  void getStorageMapFromPoseSeq(const Eigen::Matrix2Xd& transferDots1,
                                const Eigen::Matrix2Xd& transferDots2);

  bool neighboursIdxInArea(const Eigen::Matrix2Xd& dotMatrix,
                           const Eigen::Vector2d& center, double searchRadius,
                           Eigen::VectorXi& result);

  Eigen::Matrix2Xd getDirections(int startIndx);

  int findOffset(const std::array<Eigen::MatrixXi, 6>& patternReference,
                 const Eigen::MatrixXi& pattern, Eigen::Vector2i& bestOffset,
                 int& bestIndx) const;

  int getBinarycode(const Eigen::Vector2i& centerRQ, int layer);

  // Used for 2-shot detection
  int getBinarycodeFor2Shot(const Eigen::Vector2i& centerRQ, int index,
                            int layer);

  inline Sophus::SE3d T_camera_target() const { return T_camera_target_; }

  inline Eigen::Vector4d distortionParams() const { return distortionParams_; }

  inline Eigen::Matrix2Xd transferDots() const { return transferDots_; }

  inline Eigen::MatrixXi detectPattern() const { return detectPattern_; }

  // 2D map stores the index of column of detected dots
  inline Eigen::MatrixXi indexMap() const { return indexMap_; }

  inline Eigen::VectorXi binaryCode() const { return binaryCode_; }

  inline Eigen::Matrix2Xd searchDirectionsOnPattern() const {
    return searchDirectionsOnPattern_;
  }

  inline std::vector<int> bfsProcessSeq() const { return bfsProcessSeq_; }

  Eigen::Matrix2Xd inlierPose;
  Eigen::Matrix2Xd inlierDistortion;

 private:
  double spacing_;
  int numNeighboursForPoseEst_;  // needs to be <=6
  int numberBlock_;
  Eigen::Matrix2Xd imageDots_;
  Eigen::VectorXd intensity_;
  Eigen::VectorXi dotLabels_;
  Eigen::VectorXi binaryCode_;
  double perPointSearchRadius_;
  int numNeighbourLayer_;
  Sophus::SE3d T_camera_target_;
  double focalLength_;
  Eigen::Vector2d centerXY_;
  Eigen::Vector4d distortionParams_;
  Eigen::Matrix2Xd transferDots_;
  Eigen::MatrixXi detectPattern_;
  Eigen::MatrixXi indexMap_;
  Eigen::Matrix2Xd searchDirectionsOnPattern_;
  std::vector<int> bfsProcessSeq_;
  bool ifDistort_;
  bool ifTwoShot_;
  bool ifPoseMerge_;
};
}  // namespace surreal_opensource
