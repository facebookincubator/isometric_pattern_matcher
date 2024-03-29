/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <IsometricPatternMatcher/CameraModels.h>
#include <IsometricPatternMatcher/HexGridCostFunction.h>
#include <IsometricPatternMatcher/HexGridFitting.h>
#include <IsometricPatternMatcher/IsometricPattern.h>
#include <IsometricPatternMatcher/LocalParamSe3.h>
#include <fmt/format.h>
#include <glog/logging.h>
#include <Eigen/Core>
#include <limits>
#include <numeric>
#include <queue>

namespace surreal_opensource {

HexGridFitting::HexGridFitting(const Eigen::Matrix2Xd& imageDots,
                               const Eigen::Vector2d& centerXY,
                               double focalLength,
                               const Eigen::VectorXd& intensity, bool ifDistort,
                               bool ifTwoShot, bool ifPoseMerge, double spacing,
                               int numNeighboursForPoseEst, int numberSegX,
                               int numberSegY, double perPointSearchRadius,
                               int numNeighbourLayer)
    : spacing_(spacing),
      numNeighboursForPoseEst_(numNeighboursForPoseEst),
      numberSegX_(numberSegX),
      numberSegY_(numberSegY),
      imageDots_(imageDots),
      intensity_(intensity),
      perPointSearchRadius_(perPointSearchRadius),
      numNeighbourLayer_(numNeighbourLayer),
      focalLength_(focalLength),
      centerXY_(centerXY),
      ifDistort_(ifDistort),
      ifTwoShot_(ifTwoShot),
      ifPoseMerge_(ifPoseMerge) {
  distortionParams_ = Eigen::Vector4d::Zero(4, 1);
  ceres::Solver::Options solverOptions;
  findPoseAndCamModel(solverOptions);
  transferDots_ =
      reprojectDots(T_camera_target_, distortionParams_, imageDots_);
  getStorageMap();
}

HexGridFitting::HexGridFitting(
    const Eigen::Matrix2Xd& imageDots, const Eigen::Vector2d& centerXY,
    double focalLength, const Eigen::VectorXi& dotLabels, bool ifDistort,
    bool ifTwoShot, bool ifPoseMerge, double goodPoseInlierRatio,
    double spacing, int numNeighboursForPoseEst, int numberSegX, int numberSegY,
    double perPointSearchRadius, int numNeighbourLayer)
    : spacing_(spacing),
      numNeighboursForPoseEst_(numNeighboursForPoseEst),
      numberSegX_(numberSegX),
      numberSegY_(numberSegY),
      imageDots_(imageDots),
      dotLabels_(dotLabels),
      perPointSearchRadius_(perPointSearchRadius),
      numNeighbourLayer_(numNeighbourLayer),
      focalLength_(focalLength),
      centerXY_(centerXY),
      ifDistort_(ifDistort),
      ifTwoShot_(ifTwoShot),
      ifPoseMerge_(ifPoseMerge) {
  distortionParams_ = Eigen::Vector4d::Zero(4, 1);
  ceres::Solver::Options solverOptions;
  if (ifPoseMerge_) {
    Eigen::VectorXi selectedPoseIdx(numberSegX * numberSegY);
    selectedPoseIdx = findGoodPoseIndex(goodPoseInlierRatio, solverOptions);
    std::cout << "selected PoseIdx is: ";
    for (size_t i = 0; i < selectedPoseIdx.size(); i++)
      std::cout << ' ' << selectedPoseIdx[i];
    std::cout << '\n';
    std::vector<Eigen::Matrix2Xd> transferDotsGroup;
    for (int i = 0; i < selectedPoseIdx.size(); ++i) {
      if (selectedPoseIdx(i) == -1) {
        break;
        // selectedPoseIdx is index of selected good poses and unselected pose
        // index will be assigned as -1 so -1 is served as a stop sign
      }
      findPoseAndCamModel(solverOptions, selectedPoseIdx(i));
      Eigen::Matrix2Xd transferDotsNew =
          reprojectDots(T_camera_target_, distortionParams_, imageDots_);
      transferDotsGroup.push_back(transferDotsNew);
    }
    getStorageMapFromPoseSeq(transferDotsGroup);
    transferDots_ = transferDotsGroup.at(0);

  } else {
    findPoseAndCamModel(solverOptions, -1);
    transferDots_ =
        reprojectDots(T_camera_target_, distortionParams_, imageDots_);
    getStorageMap();
  }
}

void HexGridFitting::setParams(const Eigen::Vector2d& centerXY,
                               double focalLength, bool ifDistort,
                               bool ifTwoShot, bool ifPoseMerge, double spacing,
                               int numNeighboursForPoseEst, int numberSegX,
                               int numberSegY, double perPointSearchRadius,
                               int numNeighbourLayer) {
  spacing_ = spacing;
  numNeighboursForPoseEst_ = numNeighboursForPoseEst;
  numberSegX_ = numberSegX;
  numberSegY_ = numberSegY;
  perPointSearchRadius_ = perPointSearchRadius;
  numNeighbourLayer_ = numNeighbourLayer;
  focalLength_ = focalLength;
  centerXY_ = centerXY;
  ifDistort_ = ifDistort;
  ifTwoShot_ = ifTwoShot;
  ifPoseMerge_ = ifPoseMerge;
  distortionParams_ = Eigen::Vector4d::Zero(4, 1);
}

void HexGridFitting::setImageDots(const Eigen::Matrix2Xd& imageDots) {
  imageDots_ = imageDots;
}

void HexGridFitting::setTransferedDots(const Eigen::Matrix2Xd& transferDots) {
  transferDots_ = transferDots;
}

void HexGridFitting::setIntensity(const Eigen::VectorXd& intensity) {
  intensity_ = intensity;
}

void HexGridFitting::setIndexMap(const Eigen::MatrixXi& indexMap) {
  indexMap_ = indexMap;
}

std::vector<Eigen::Matrix2Xd> HexGridFitting::imageNeighbourMatrix(
    int numberNeighours) {
  std::vector<Eigen::Matrix2Xd> result;
  for (int j = 0; j < numberNeighours; ++j) {
    result.push_back(Eigen::Matrix2Xd::Zero(2, imageDots_.cols()));
  }
  for (int idxDot = 0; idxDot < imageDots_.cols(); ++idxDot) {
    std::vector<double> distance;
    std::vector<int> indx;
    for (int j = 0; j < imageDots_.cols(); ++j)
      distance.push_back((imageDots_.col(idxDot) - imageDots_.col(j)).norm());
    getSortIndx(distance, indx);
    for (int j = 0; j < numberNeighours; ++j) {
      result.at(j).col(idxDot) =
          imageDots_.col(indx.at(j + 1));  // the first one is itself
    }
  }  // end for
  return result;
}

template <typename T>
void HexGridFitting::getSortIndx(const T& coords, std::vector<int>& idx) {
  idx.resize(coords.size());
  std::iota(idx.begin(), idx.end(), 0);

  sort(idx.begin(), idx.end(),
       [&coords](size_t i1, size_t i2) { return coords[i1] < coords[i2]; });
  return;
}

std::vector<int> HexGridFitting::findInliers(
    const std::vector<Eigen::Matrix2Xd>& neighbourDots,
    const Sophus::SE3d& T_camera_target,
    const Eigen::Vector4d& distortionParams, const double inlierThreshold) {
  // if each distance of the numberNeighbour closest neighours to the dots in
  // the transferred space is around spacing, the dot is considered as an inlier
  std::vector<int> inliersIndx;
  size_t numberNeighbour = neighbourDots.size();
  std::vector<Eigen::VectorXd> distance;
  for (auto const& neighbourMatrix : neighbourDots) {
    distance.push_back(
        (reprojectDots(T_camera_target, distortionParams, imageDots_) -
         reprojectDots(T_camera_target, distortionParams, neighbourMatrix))
            .colwise()
            .norm() -
        Eigen::MatrixXd::Constant(1, imageDots_.cols(), spacing_));
  }
  for (int i = 0; i < imageDots_.cols(); ++i) {
    size_t numInlierPts = 0;
    for (const auto& neighourDist : distance) {
      if (abs(neighourDist(i)) <= inlierThreshold) {
        numInlierPts++;
      }
    }
    if (numInlierPts == numberNeighbour) inliersIndx.push_back(i);
  }
  return inliersIndx;
}

bool HexGridFitting::findT_camera_target(
    const ceres::Solver::Options& solverOption,
    const std::vector<Eigen::Matrix2Xd>&
        neighbourDots,  // each Matrix2Xd is the neighbours of a detected dot (#
                        // neighbours = numNeighboursForPoseEst_)
    const std::vector<int>& sampleIndx,  // sampleIndx stores the indices of
                                         // detected dots for the patch
    const double inlierThreshold, const Sophus::Plane3d& plane,
    Sophus::SE3d& T_camera_target, std::vector<int>& inliersIndx) {
  ceres::Problem::Options options;
  ceres::Problem problem(options);

  std::vector<double*> params = {T_camera_target.data()};
  LocalParamSe3* localParamSe3 = new LocalParamSe3;
  problem.AddParameterBlock(params[0], Sophus::SE3d::num_parameters,
                            localParamSe3);
  ceres::LossFunction* loss_function = nullptr;
  Eigen::VectorXd cameraModelParams;
  if (ifDistort_) {
    // KB3 camera model
    cameraModelParams.resize(KannalaBrandtK3Projection::kNumParams, 1);
    cameraModelParams << focalLength_, focalLength_, centerXY_(0), centerXY_(1),
        0.0, 0.0, 0.0, 0.0;
  } else {
    // Linear camera model
    cameraModelParams.resize(PinholeProjection::kNumParams, 1);
    cameraModelParams << focalLength_, focalLength_, centerXY_(0), centerXY_(1);
  }
  for (auto& i : sampleIndx) {
    for (auto const& neighbourMatrix : neighbourDots) {
      Ray3d rayInCamera;
      Ray3d rayNeighbourInCamera;
      if (ifDistort_) {
        rayInCamera = KannalaBrandtK3Projection::unproject(imageDots_.col(i),
                                                           cameraModelParams);
        rayNeighbourInCamera = KannalaBrandtK3Projection::unproject(
            neighbourMatrix.col(i), cameraModelParams);
      } else {
        rayInCamera =
            PinholeProjection::unproject(imageDots_.col(i), cameraModelParams);
        rayNeighbourInCamera = PinholeProjection::unproject(
            neighbourMatrix.col(i), cameraModelParams);
      }
      ceres::CostFunction* cost =
          new ceres::AutoDiffCostFunction<isometricPatternPoseCost, 1,
                                          Sophus::SE3d::num_parameters>(
              new isometricPatternPoseCost(rayInCamera, rayNeighbourInCamera,
                                           plane, spacing_));
      problem.AddResidualBlock(cost, loss_function, params[0]);
    }
  }
  ceres::Solver::Summary summary;
  ceres::Solve(solverOption, &problem, &summary);
  if (!summary.IsSolutionUsable()) return false;

  if (T_camera_target.inverse().translation().z() < 0) {
    T_camera_target.translation().z() = -T_camera_target.translation().z();
  }  // make sure the camera is on top of the target (not behind the target)
  if (T_camera_target.translation().z() < 0) {
    T_camera_target *= Sophus::SE3d::rotX(M_PI);
  }  // make sure the camera is facing the target
  inliersIndx = findInliers(neighbourDots, T_camera_target, distortionParams_,
                            inlierThreshold);
  return true;
}

bool HexGridFitting::findKb3DistortionParams(
    const ceres::Solver::Options& solverOption,
    const std::vector<Eigen::Matrix2Xd>& neighbourDots,
    const std::vector<int>& sampleIndx, const double inlierThreshold,
    const Sophus::Plane3d& plane, Sophus::SE3d& T_camera_target,
    std::vector<double>& distortionParams, std::vector<int>& inliersIndx) {
  ceres::LossFunction* loss_function = nullptr;  // new ceres::CauchyLoss(1.0);
  size_t numberNeighbour = neighbourDots.size();
  ceres::Problem::Options options;
  std::vector<double*> params = {T_camera_target.data()};
  std::vector<double*> paramsDistortion = {distortionParams.data()};

  ceres::Problem problemDistortion(options);
  problemDistortion.AddParameterBlock(params[0], Sophus::SE3d::num_parameters,
                                      new LocalParamSe3);

  for (size_t j = 0; j < numberNeighbour; ++j) {
    for (const int& i : sampleIndx) {
      ceres::CostFunction* cost =
          new ceres::AutoDiffCostFunction<isometricPatternDistortionCost, 1,
                                          Sophus::SE3d::num_parameters, 4>(
              new isometricPatternDistortionCost(
                  imageDots_.col(i), neighbourDots.at(j).col(i), centerXY_,
                  focalLength_, plane));
      problemDistortion.AddResidualBlock(cost, loss_function, params[0],
                                         paramsDistortion[0]);
    }
  }
  ceres::Solver::Summary summary;
  ceres::Solve(solverOption, &problemDistortion, &summary);

  if (!summary.IsSolutionUsable()) return false;
  inliersIndx = findInliers(
      neighbourDots, T_camera_target,
      Eigen::Map<Eigen::Vector4d>(distortionParams.data()), inlierThreshold);
  if (T_camera_target.inverse().translation().z() < 0) {
    T_camera_target.translation().z() = -T_camera_target.translation().z();
  }  // make sure the camera is on top of the target (not behind the target)
  if (T_camera_target.translation().z() < 0) {
    T_camera_target *= Sophus::SE3d::rotX(M_PI);
  }  // make sure the camera is facing the target
  return true;
}

Eigen::VectorXi HexGridFitting::findGoodPoseIndex(
    double goodPoseInlierRatio, const ceres::Solver::Options& solverOption,
    const Sophus::SE3d& initT_camera_target) {
  Eigen::VectorXi selectedPoseIdx(numberSegX_ * numberSegY_);
  selectedPoseIdx.fill(-1);
  Sophus::Plane3d plane(Sophus::Vector3d(0.0, 0.0, 1.0), 0.0);
  std::vector<Eigen::Matrix2Xd> neighbourDots =
      imageNeighbourMatrix(numNeighboursForPoseEst_);
  std::vector<Sophus::SE3d> Ts_camera_targetForSubregions;
  std::vector<std::vector<int>> inliersIndx(numberSegX_ * numberSegY_);
  std::vector<int> bestIndxs = calculateSubregionPosesAndBestIndex(
      solverOption, plane, neighbourDots, initT_camera_target,
      Ts_camera_targetForSubregions, inliersIndx);
  // build descend order selectedPoseIdx
  for (int i = 0; i < selectedPoseIdx.size(); ++i) {
    int idx = bestIndxs.size() - i - 1;
    int poseidx = bestIndxs.at(idx);
    // only select pose with enough inliers
    if (inliersIndx[poseidx].size() >=
        imageDots_.cols() * goodPoseInlierRatio) {
      selectedPoseIdx(i) = poseidx;
    }
  }
  return selectedPoseIdx;
}

std::vector<int> HexGridFitting::calculateSubregionPosesAndBestIndex(
    const ceres::Solver::Options& solverOption, const Sophus::Plane3d& plane,
    const std::vector<Eigen::Matrix2Xd>& neighbourDots,
    const Sophus::SE3d& initT_camera_target,
    std::vector<Sophus::SE3d>& Ts_camera_targetForSubregions,
    std::vector<std::vector<int>>& inliersIndx) {
  // blocks
  double maxX = imageDots_.row(0).maxCoeff();
  double minX = imageDots_.row(0).minCoeff();
  double maxY = imageDots_.row(1).maxCoeff();
  double minY = imageDots_.row(1).minCoeff();
  std::vector<double> segX;
  std::vector<double> segY;
  // segment X
  for (int i = 0; i <= numberSegX_; i++) {
    segX.push_back(minX + (maxX - minX) / numberSegX_ * i);
  }
  // segment Y
  for (int i = 0; i <= numberSegY_; i++) {
    segY.push_back(minY + (maxY - minY) / numberSegY_ * i);
  }
  std::vector<std::vector<int>> blockIndx(numberSegX_ * numberSegY_);
  for (size_t i = 0; i < imageDots_.cols(); ++i) {
    for (int x = 0; x < numberSegX_; x++) {
      for (int y = 0; y < numberSegY_; y++) {
        if (imageDots_(0, i) >= segX[x] && imageDots_(0, i) < segX[x + 1] &&
            imageDots_(1, i) >= segY[y] && imageDots_(1, i) < segY[y + 1])
          blockIndx[x * numberSegY_ + y].push_back(i);
      }
    }
  }
  std::vector<int> numberInliers;
  std::vector<int> indx;
  for (int i = 0; i < blockIndx.size(); ++i) {
    Ts_camera_targetForSubregions.push_back(
        initT_camera_target);  // initialization
    if (findT_camera_target(solverOption, neighbourDots, blockIndx[i], 0.2,
                            plane, Ts_camera_targetForSubregions[i],
                            inliersIndx[i])) {
      numberInliers.push_back(inliersIndx[i].size());
    }
  }
  getSortIndx(numberInliers, indx);  // ascend order
  return indx;
}

void HexGridFitting::findPoseAndCamModel(
    const ceres::Solver::Options& solverOption, int selectIndx,
    const Sophus::SE3d& initT_camera_target) {
  Sophus::Plane3d plane(Sophus::Vector3d(0.0, 0.0, 1.0), 0.0);
  std::vector<Eigen::Matrix2Xd> neighbourDots =
      imageNeighbourMatrix(numNeighboursForPoseEst_);
  std::vector<Sophus::SE3d> Ts_camera_targetForSubregions;
  std::vector<std::vector<int>> inliersIndx(numberSegX_ * numberSegY_);
  // per block pose estimation
  std::vector<int> bestIndxs = calculateSubregionPosesAndBestIndex(
      solverOption, plane, neighbourDots, initT_camera_target,
      Ts_camera_targetForSubregions, inliersIndx);
  int bestIndx = bestIndxs.back();
  if (selectIndx != -1) {
    bestIndx = selectIndx;
  }
  // global re-estimate
  T_camera_target_ = Ts_camera_targetForSubregions[bestIndx];
  std::vector<int> inliersPoseIndx;
  bool ifFindTCameraTarget =
      findT_camera_target(solverOption, neighbourDots, inliersIndx[bestIndx],
                          0.3, plane, T_camera_target_, inliersPoseIndx);

  CHECK(ifFindTCameraTarget) << "Cannot find T_camera_target";
  CHECK_GT(inliersPoseIndx.size(), 0) << "No inliers for T_camera_target. You "
                                         "can try with different focal length.";

  inlierPose.resize(2, inliersPoseIndx.size());
  for (size_t i = 0; i < inliersPoseIndx.size(); ++i) {
    inlierPose.col(i) = imageDots_.col(inliersPoseIndx[i]);
  }
  if (ifDistort_) {
    std::vector<int> inliersDistortIndx;
    std::vector<double> distortionParams = {0.0, 0.0, 0.0, 0.0};
    if (findKb3DistortionParams(solverOption, neighbourDots, inliersPoseIndx,
                                0.3, plane, T_camera_target_, distortionParams,
                                inliersDistortIndx)) {
      distortionParams_ = Eigen::Map<Eigen::Vector4d>(distortionParams.data());

      // fmt::print("Distortion parameters: {} \n",
      // distortionParams_.transpose());
      inlierDistortion.resize(2, inliersDistortIndx.size());
      for (size_t i = 0; i < inliersDistortIndx.size(); ++i) {
        inlierDistortion.col(i) = imageDots_.col(inliersDistortIndx[i]);
      }
    } else {
      LOG(INFO) << "Warning: Cannot find distortion parameters. Please try "
                   "with different focal length";
    }
  }
}

Eigen::Matrix2Xd HexGridFitting::reprojectDots(
    const Sophus::SE3d& T_camera_target,
    const Eigen::Vector4d& distortionParams,
    const Eigen::Matrix2Xd& imageDots) {
  Eigen::Matrix2Xd result;
  result.resize(2, imageDots.cols());
  Sophus::Plane3d plane(Sophus::Vector3d(0.0, 0.0, 1.0), 0);
  for (int i = 0; i < imageDots.cols(); ++i) {
    Ray3d rayInCamera;
    if (ifDistort_) {
      Sophus::Vector<double, KannalaBrandtK3Projection::kNumParams> intrinsics;
      intrinsics << focalLength_, focalLength_, centerXY_(0), centerXY_(1),
          distortionParams(0), distortionParams(1), distortionParams(2),
          distortionParams(3);
      rayInCamera =
          KannalaBrandtK3Projection::unproject(imageDots.col(i), intrinsics);
    } else {
      Sophus::Vector<double, PinholeProjection::kNumParams> intrinsics;
      intrinsics << focalLength_, focalLength_, centerXY_(0), centerXY_(1);
      rayInCamera = PinholeProjection::unproject(imageDots.col(i), intrinsics);
    }
    Ray3d rayInTarget = T_camera_target.inverse() * rayInCamera;
    Sophus::Vector3d ptTarget3d = rayInTarget.line().intersectionPoint(plane);
    result.col(i) = ptTarget3d.head<2>();
  }
  return result;
}

void HexGridFitting::getStorageMap() {
  Eigen::Matrix3Xi cubeCoor;
  cubeCoor.resize(3, transferDots_.cols());
  cubeCoor.fill(std::numeric_limits<int>::max());  // not detected =max int
  // get cube coordinate
  int minX = 0;
  int maxX = 0;
  int minZ = 0;
  int maxZ = 0;
  getCubeCoordinate(transferDots_, minX, maxX, minZ, maxZ, cubeCoor,
                    bfsProcessSeq_);
  binaryCode_ = Eigen::VectorXi::Constant(transferDots_.cols(), 1, 2);
  buildBinaryCode(cubeCoor, minX, maxX, minZ, maxZ);
}

int HexGridFitting::getCubeCoordinate(const Eigen::Matrix2Xd& transferDots,
                                      int& minX, int& maxX, int& minZ,
                                      int& maxZ, Eigen::Matrix3Xi& cubeCoor,
                                      std::vector<int>& bfsProcessSeq) {
  cubeCoor.resize(3, transferDots.cols());
  cubeCoor.fill(std::numeric_limits<int>::max());  // not detected =max int
  std::queue<int> bfsQueue;
  Eigen::VectorXi processed;
  processed.setZero(transferDots.cols(), 1);

  Eigen::Matrix3Xi cubedirection;
  cubedirection.resize(3, IsometricGridDot::kNumNeighbours);
  cubedirection << -1, 0, 1, 1, 0, -1, 1, 1, 0, -1, -1, 0, 0, -1, -1, 0, 1, 1;
  // start from the center
  Eigen::Vector2d center;
  center(0) = transferDots.row(0).mean();
  center(1) = transferDots.row(1).mean();
  Eigen::VectorXi centerNeighbour;
  neighboursIdxInArea(transferDots, center, 2 * spacing_,
                      centerNeighbour);  // find a point near the center
  CHECK(centerNeighbour.rows() > 0) << "center neighbor size should be >0";
  int startIdx = centerNeighbour(0);
  // manually set transferDots
  transferDots_ = transferDots;
  searchDirectionsOnPattern_ = getDirections(startIdx);

  // get cube coordinates
  processed(startIdx) = 1;
  cubeCoor.col(startIdx) << 0, 0, 0;
  bfsQueue.push(startIdx);
  while (!bfsQueue.empty()) {
    int centerIndx = bfsQueue.front();
    bfsQueue.pop();
    for (int k = 0; k < IsometricGridDot::kNumNeighbours; k++) {
      Eigen::VectorXi Indx;
      Eigen::Vector2d possLocation =
          transferDots.col(centerIndx) + searchDirectionsOnPattern_.col(k);
      if (neighboursIdxInArea(transferDots, possLocation, perPointSearchRadius_,
                              Indx)) {
        if (processed(Indx(0)) == 0) {
          // check if the point is near the possible location
          bfsQueue.push(Indx(0));
          bfsProcessSeq.push_back(Indx(0));
          processed(Indx(0)) = 1;
          cubeCoor.col(Indx(0)) =
              cubeCoor.col(centerIndx) + cubedirection.col(k);
          if (minX > cubeCoor(0, Indx(0))) minX = cubeCoor(0, Indx(0));
          if (minZ > cubeCoor(2, Indx(0))) minZ = cubeCoor(2, Indx(0));
          if (maxX < cubeCoor(0, Indx(0))) maxX = cubeCoor(0, Indx(0));
          if (maxZ < cubeCoor(2, Indx(0))) maxZ = cubeCoor(2, Indx(0));
        }  // end if
      }    // end if
    }      // end for
  }        // end while

  if (processed.sum() < transferDots.cols())
    LOG(INFO) << fmt::format(
        "{} points are processed in BFS from total {} detected points",
        processed.sum(), transferDots.cols());
  return startIdx;
}

Eigen::Vector3i HexGridFitting::doLeftRotate(const Eigen::Vector3i& coord,
                                             size_t rotIdx) {
  // iteration for rotIdx times of 60 degree left (counter clockwise) rotation
  Eigen::Vector3i rotCubeCoor = coord;
  for (size_t i = 0; i < rotIdx; ++i) {
    rotCubeCoor = IsometricGridDot::rotateLeft60ForDot(rotCubeCoor);
  }
  return rotCubeCoor;
}

Eigen::Vector3i HexGridFitting::doRightRotate(const Eigen::Vector3i& coord,
                                              size_t rotIdx) {
  // iteration for rotIdx times of 60 degree right (clockwise) rotation
  Eigen::Vector3i rotCubeCoor = coord;
  for (size_t i = 0; i < rotIdx; ++i) {
    rotCubeCoor = IsometricGridDot::rotateRight60ForDot(rotCubeCoor);
  }
  return rotCubeCoor;
}

int HexGridFitting::determineRotation(const Eigen::Matrix3Xi& cubeCoor1,
                                      const Eigen::Matrix3Xi& cubeCoor2,
                                      const Eigen::Matrix3Xi& cubeCoorDiff) {
  CHECK(cubeCoor1.cols() == cubeCoor2.cols())
      << "cubeCoor should have same size";
  size_t rotIdx = 0;  // number of left 60 degree
  Eigen::VectorXi inlierNumber(6);
  inlierNumber.fill(0);
  for (int i = 0; i < cubeCoor1.cols(); i++) {
    if ((cubeCoor1(0, i) != std::numeric_limits<int>::max()) &&
        (cubeCoor2(0, i) != std::numeric_limits<int>::max())) {
      // we only use shared dot to determine the rotation
      for (rotIdx = 0; rotIdx < 6; ++rotIdx) {
        if (cubeCoor1.col(i) ==
            doLeftRotate(cubeCoor2.col(i), rotIdx) + cubeCoorDiff) {
          inlierNumber(rotIdx) += 1;
        }
      }
    }
  }
  // determine best rotation index
  int bestNumber = 0;
  for (int i = 0; i < 6; ++i) {
    if (bestNumber < inlierNumber(i)) {
      bestNumber = inlierNumber(i);
      rotIdx = i;
    }
  }
  return rotIdx;
}

Eigen::Matrix3Xi HexGridFitting::mergeCubeCoordinate(
    const Eigen::Matrix3Xi& cubeCoor1, const Eigen::Matrix3Xi& cubeCoor2,
    int startIdx1, int startIdx2, int& minX, int& maxX, int& minZ, int& maxZ,
    int poseIdx) {
  // merge cube coordinate results from two poses
  Eigen::Matrix3Xi cubeCoor;
  cubeCoor.resize(3, cubeCoor1.cols());
  cubeCoor.fill(std::numeric_limits<int>::max());  // not detected =max int
  // calculate coordinate translation
  Eigen::Matrix3Xi cubeCoorDiff;
  cubeCoorDiff.resize(3, 1);
  cubeCoorDiff << 0, 0, 0;
  if (startIdx1 == startIdx2) {
    CHECK(cubeCoor1.col(startIdx2) == cubeCoor2.col(startIdx2))
        << "Both startIdx should have 0,0,0 cube coordinate";
  } else {
    CHECK(cubeCoor1(0, startIdx2) != std::numeric_limits<int>::max())
        << "we assume the center of transferDot2 has valid cubeCoord1";
    cubeCoorDiff = cubeCoor1.col(startIdx2) - cubeCoor2.col(startIdx2);
  }
  LOG(INFO) << fmt::format("translation is {}, {}, {}", cubeCoorDiff(0, 0),
                           cubeCoorDiff(1, 0), cubeCoorDiff(2, 0));
  // calculate coordinate rotation
  int rotIdx = determineRotation(cubeCoor1, cubeCoor2, cubeCoorDiff);
  LOG(INFO) << fmt::format("rotation is left {} degree", rotIdx * 60);

  // Merge coordinate and update minX, maxX, minZ, maxZ
  for (int i = 0; i < cubeCoor.cols(); i++) {
    if (cubeCoor1(0, i) != std::numeric_limits<int>::max()) {
      // Add cube coordinate from valid cubeCoor1
      cubeCoor.col(i) = cubeCoor1.col(i);
      if (poseIdx == 1) {
        // we only push back index i at first time it is met
        bfsProcessSeq_.push_back(i);
      }
    } else if ((cubeCoor2(0, i) != std::numeric_limits<int>::max())) {
      // Add cube coordinate from valid cubeCoor2 while cubeCoor1 is invalid
      cubeCoor.col(i) = doLeftRotate(cubeCoor2.col(i), rotIdx) + cubeCoorDiff;
      if (minX > cubeCoor(0, i)) minX = cubeCoor(0, i);
      if (minZ > cubeCoor(2, i)) minZ = cubeCoor(2, i);
      if (maxX < cubeCoor(0, i)) maxX = cubeCoor(0, i);
      if (maxZ < cubeCoor(2, i)) maxZ = cubeCoor(2, i);
      bfsProcessSeq_.push_back(i);
    }
  }
  LOG(INFO) << fmt::format("total processed points from 2 poses are {}",
                           bfsProcessSeq_.size());
  return cubeCoor;
}
void HexGridFitting::getStorageMapFromPoseSeq(
    const std::vector<Eigen::Matrix2Xd>& transferDotsGroup) {
  int minX = 0;
  int maxX = 0;
  int minZ = 0;
  int maxZ = 0;
  // get cube coordinate from first pose: base pose
  Eigen::Matrix2Xd transferDotsBase = transferDotsGroup.at(0);
  Eigen::Matrix3Xi cubeCoorBase;
  std::vector<int> bfsProcessSeqBase;
  int startIdxBase = getCubeCoordinate(transferDotsBase, minX, maxX, minZ, maxZ,
                                       cubeCoorBase, bfsProcessSeqBase);
  if (transferDotsGroup.size() == 1) {
    bfsProcessSeq_ = bfsProcessSeqBase;
  }

  // loop over transferred dot group: calculate and merge cube coordinate
  for (int poseIdx = 1; poseIdx < transferDotsGroup.size(); ++poseIdx) {
    Eigen::Matrix2Xd transferDotsNew = transferDotsGroup.at(poseIdx);
    CHECK(transferDotsBase.cols() == transferDotsNew.cols())
        << "transferDots should have same size";
    // get cube coordinate from transferred dot of new pose
    Eigen::Matrix3Xi cubeCoorNew;
    std::vector<int> bfsProcessSeqNew;
    int startIdxNew = getCubeCoordinate(transferDotsNew, minX, maxX, minZ, maxZ,
                                        cubeCoorNew, bfsProcessSeqNew);
    cubeCoorBase =
        mergeCubeCoordinate(cubeCoorBase, cubeCoorNew, startIdxBase,
                            startIdxNew, minX, maxX, minZ, maxZ, poseIdx);
  }
  // build binary code
  binaryCode_ = Eigen::VectorXi::Constant(transferDotsBase.cols(), 1, 2);
  buildBinaryCode(cubeCoorBase, minX, maxX, minZ, maxZ);
}

void HexGridFitting::buildBinaryCode(const Eigen::Matrix3Xi& cubeCoor, int minX,
                                     int maxX, int minZ, int maxZ) {
  int storageMapRow =
      maxZ - minZ > maxX - minX ? maxZ - minZ + 1 : maxX - minX + 1;
  detectPattern_ = Eigen::MatrixXi::Constant(
      storageMapRow, storageMapRow, 2);  // not detected pt in Pattern =2
  indexMap_.setConstant(storageMapRow, storageMapRow,
                        -1);  // not detected pt in index map = -1

  for (int i = 0; i < cubeCoor.cols(); ++i) {
    if (cubeCoor(0, i) != std::numeric_limits<int>::max()) {
      indexMap_(cubeCoor(2, i) - minZ, cubeCoor(0, i) - minX) = i;
    }
  }

  for (int i = 0; i < cubeCoor.cols(); ++i) {
    if (cubeCoor(0, i) != std::numeric_limits<int>::max()) {
      Eigen::Vector2i centerRQ =
          Eigen::Vector2i(cubeCoor(2, i), cubeCoor(0, i));
      centerRQ(0) -= minZ;
      centerRQ(1) -= minX;
      if (ifTwoShot_) {
        binaryCode_(i) = dotLabels_(i);
      } else {
        binaryCode_(i) = getBinarycode(centerRQ, numNeighbourLayer_);
      }
      detectPattern_(cubeCoor(2, i) - minZ, cubeCoor(0, i) - minX) =
          binaryCode_(i);
    }
  }
}
bool HexGridFitting::neighboursIdxInArea(const Eigen::Matrix2Xd& dotMatrix,
                                         const Eigen::Vector2d& center,
                                         double searchRadius,
                                         Eigen::VectorXi& result) {
  bool flag = false;
  std::vector<std::pair<double, int>> distanceIndxPair;
  Eigen::VectorXd distance =
      ((dotMatrix.row(0) -
        Eigen::MatrixXd::Constant(1, dotMatrix.cols(), center(0)))
           .cwiseAbs2() +
       (dotMatrix.row(1) -
        Eigen::MatrixXd::Constant(1, dotMatrix.cols(), center(1)))
           .cwiseAbs2())
          .cwiseSqrt();
  for (int i = 0; i < dotMatrix.cols(); ++i) {
    if (distance(i) < searchRadius) {
      distanceIndxPair.push_back(std::make_pair(distance(i), i));
      flag = true;
    }
  }
  if (flag) {
    result.resize(distanceIndxPair.size(), 1);
    sort(distanceIndxPair.begin(), distanceIndxPair.end());
    for (size_t i = 0; i < distanceIndxPair.size(); ++i) {
      result(i) = distanceIndxPair[i].second;
    }
  }
  return flag;
}

int HexGridFitting::getBinarycode(
    const Eigen::Vector2i& centerRQ,
    int layer) {  // the number of layers that the neighours are used to
                  // deternmine the binary code of the center
  std::vector<double> colorNeighbour;
  for (int r = -layer; r <= layer; ++r) {
    for (int q = -layer; q <= layer; ++q) {
      if (r + q >= -layer && r + q <= layer && r + centerRQ.x() >= 0 &&
          r + centerRQ.x() < indexMap_.rows() && q + centerRQ.y() >= 0 &&
          q + centerRQ.y() < indexMap_.cols()) {
        if (indexMap_(r + centerRQ.x(), q + centerRQ.y()) != -1)
          colorNeighbour.push_back(
              intensity_(indexMap_(r + centerRQ.x(), q + centerRQ.y())));
      }
    }                               // end c
  }                                 // end r
  if (colorNeighbour.size() > 2) {  // more than two neighbours
    std::sort(colorNeighbour.begin(), colorNeighbour.end());
    double colorMedian =
        (*colorNeighbour.begin() + colorNeighbour.back()) / 2.0;
    return intensity_(indexMap_(centerRQ.x(), centerRQ.y())) > colorMedian ? 1
                                                                           : 0;
  } else {
    return 2;
  }
}

Eigen::Matrix2Xd HexGridFitting::getDirections(int startIndx) {
  Eigen::VectorXi neighbourIndx;
  neighboursIdxInArea(transferDots_, transferDots_.col(startIndx),
                      spacing_ + perPointSearchRadius_, neighbourIndx);
  Eigen::Matrix2Xd result;
  result.resize(2, IsometricGridDot::kNumNeighbours);
  result.col(0) =
      transferDots_.col(neighbourIndx(1)) -
      transferDots_.col(startIndx);  // neighbourIndx(0) is the point itself
  const double cos60 = 0.50;
  const double sin60 = sqrt(3.0) / 2.0;
  for (int i = 0; i < 5; ++i) {
    result(0, i + 1) = result(0, i) * cos60 + result(1, i) * sin60;
    result(1, i + 1) = result(1, i) * cos60 - result(0, i) * sin60;
  }
  return result;
}

int HexGridFitting::findOffset(
    const std::array<Eigen::MatrixXi, 6>& patternReference,
    const Eigen::MatrixXi& pattern, Eigen::Vector2i& bestOffset,
    int& bestIndx) const {
  const int R = pattern.rows();
  const int C = pattern.cols();
  const int referenceR = patternReference[0].rows();
  const int referenceC = patternReference[0].cols();
  int bestMatch = 0;

  for (int patternIndx = 0; patternIndx < 6; ++patternIndx) {
    // For all offsets
    for (int offsetR = -R; offsetR < R; ++offsetR) {
      for (int offsetC = -C; offsetC < C; ++offsetC) {
        int numberMatch = 0;
        for (int r = 0; r < R; ++r) {
          for (int c = 0; c < C; ++c) {
            if (r + offsetR >= 0 && r + offsetR < referenceR &&
                c + offsetC >= 0 && c + offsetC < referenceC) {
              if (patternReference[patternIndx](r + offsetR, c + offsetC) ==
                      pattern(r, c) &&
                  pattern(r, c) != 2)
                numberMatch += 1;
            }  // end if
          }    // end c
        }      // end r
        if (numberMatch > bestMatch) {
          bestMatch = numberMatch;
          bestOffset << offsetR, offsetC;
          bestIndx = patternIndx;
        }
      }  // end offsetC
    }    // end offsetR
  }
  if (bfsProcessSeq_.size() == 0 ||
      (double(bestMatch) / double(bfsProcessSeq_.size())) < 0.7) {
    LOG(INFO) << fmt::format(
        "Pattern matching failed, only {} points can be matched from {} "
        "processed points.",
        bestMatch, bfsProcessSeq_.size());
  }

  return bestMatch;
}

}  // namespace surreal_opensource
