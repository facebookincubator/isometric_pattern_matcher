// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <IsometricPatternMatcher/CameraModels.h>
#include <IsometricPatternMatcher/HexGridCostFunction.h>
#include <IsometricPatternMatcher/HexGridFitting.h>
#include <IsometricPatternMatcher/IsometricPattern.h>
#include <IsometricPatternMatcher/LocalParamSe3.h>
#include <fmt/format.h>
#include <limits>
#include <numeric>
#include <queue>

namespace surreal_opensource {

HexGridFitting::HexGridFitting(
    const Eigen::Matrix2Xd& imageDots,
    const Eigen::Vector2d& centerXY,
    double focalLength,
    const Eigen::VectorXd& intensity,
    bool ifDistort,
    double spacing,
    int numNeighboursForPoseEst,
    int numberBlock,
    double perPointSearchRadius,
    int numNeighbourLayer)
    : spacing_(spacing),
      numNeighboursForPoseEst_(numNeighboursForPoseEst),
      numberBlock_(numberBlock),
      imageDots_(imageDots),
      intensity_(intensity),
      perPointSearchRadius_(perPointSearchRadius),
      numNeighbourLayer_(numNeighbourLayer),
      focalLength_(focalLength),
      centerXY_(centerXY),
      ifDistort_(ifDistort) {
  distortionParams_ = Eigen::Vector4d::Zero(4, 1);
  ceres::Solver::Options solverOptions;
  findPoseAndCamModel(solverOptions);
  transferDots_ = reprojectDots(T_camera_target_, distortionParams_, imageDots_);
  getStorageMap();
}

void HexGridFitting::setParams(
    const Eigen::Vector2d& centerXY,
    double focalLength,
    bool ifDistort,
    double spacing,
    int numNeighboursForPoseEst,
    int numberBlock,
    double perPointSearchRadius,
    int numNeighbourLayer) {
  spacing_ = spacing;
  numNeighboursForPoseEst_ = numNeighboursForPoseEst;
  numberBlock_ = numberBlock;
  perPointSearchRadius_ = perPointSearchRadius;
  numNeighbourLayer_ = numNeighbourLayer;
  focalLength_ = focalLength;
  centerXY_ = centerXY;
  ifDistort_ = ifDistort;
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

std::vector<Eigen::Matrix2Xd> HexGridFitting::imageNeighbourMatrix(int numberNeighours) {
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
      result.at(j).col(idxDot) = imageDots_.col(indx.at(j + 1)); // the first one is itself
    }
  } // end for
  return result;
}

template <typename T>
void HexGridFitting::getSortIndx(const T& coords, std::vector<int>& idx) {
  idx.resize(coords.size());
  std::iota(idx.begin(), idx.end(), 0);

  sort(idx.begin(), idx.end(), [&coords](size_t i1, size_t i2) { return coords[i1] < coords[i2]; });
  return;
}

std::vector<int> HexGridFitting::findInliers(
    const std::vector<Eigen::Matrix2Xd>& neighbourDots,
    const Sophus::SE3d& T_camera_target,
    const Eigen::Vector4d& distortionParams,
    const double inlierThreshold) {
  // if each distance of the numberNeighbour closest neighours to the dots in the transferred space
  // is around spacing, the dot is considered as an inlier
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
    if (numInlierPts == numberNeighbour)
      inliersIndx.push_back(i);
  }
  return inliersIndx;
}

bool HexGridFitting::findT_camera_target(
    const ceres::Solver::Options& solverOption,
    const std::vector<Eigen::Matrix2Xd>&
        neighbourDots, // each Matrix2Xd is the neighbours of a detected dot (# neighbours =
                       // numNeighboursForPoseEst_)
    const std::vector<int>&
        sampleIndx, // sampleIndx stores the indices of detected dots for the patch
    const double inlierThreshold,
    const Sophus::Plane3d& plane,
    Sophus::SE3d& T_camera_target,
    std::vector<int>& inliersIndx) {
  ceres::Problem::Options options;
  ceres::Problem problem(options);

  std::vector<double*> params = {T_camera_target.data()};
  LocalParamSe3* localParamSe3 = new LocalParamSe3;
  problem.AddParameterBlock(params[0], Sophus::SE3d::num_parameters, localParamSe3);
  ceres::LossFunction* loss_function = nullptr;
  Eigen::VectorXd cameraModelParams;
  if (ifDistort_) {
    // KB3 camera model
    cameraModelParams.resize(KannalaBrandtK3Projection::kNumParams, 1);
    cameraModelParams << focalLength_, focalLength_, centerXY_(0), centerXY_(1), 0.0, 0.0, 0.0, 0.0;
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
        rayInCamera = KannalaBrandtK3Projection::unproject(imageDots_.col(i), cameraModelParams);
        rayNeighbourInCamera =
            KannalaBrandtK3Projection::unproject(neighbourMatrix.col(i), cameraModelParams);
      } else {
        rayInCamera = PinholeProjection::unproject(imageDots_.col(i), cameraModelParams);
        rayNeighbourInCamera =
            PinholeProjection::unproject(neighbourMatrix.col(i), cameraModelParams);
      }
      ceres::CostFunction* cost = new ceres::
          AutoDiffCostFunction<isometricPatternPoseCost, 1, Sophus::SE3d::num_parameters>(
              new isometricPatternPoseCost(rayInCamera, rayNeighbourInCamera, plane, spacing_));
      problem.AddResidualBlock(cost, loss_function, params[0]);
    }
  }
  ceres::Solver::Summary summary;
  ceres::Solve(solverOption, &problem, &summary);
  if (!summary.IsSolutionUsable())
    return false;

  if (T_camera_target.inverse().translation().z() < 0) {
    T_camera_target.translation().z() = -T_camera_target.translation().z();
  } // make sure the camera is on top of the target (not behind the target)
  if (T_camera_target.translation().z() < 0) {
    T_camera_target *= Sophus::SE3d::rotX(M_PI);
  } // make sure the camera is facing the target
  inliersIndx = findInliers(neighbourDots, T_camera_target, distortionParams_, inlierThreshold);
  return true;
}

bool HexGridFitting::findKb3DistortionParams(
    const ceres::Solver::Options& solverOption,
    const std::vector<Eigen::Matrix2Xd>& neighbourDots,
    const std::vector<int>& sampleIndx,
    const double inlierThreshold,
    const Sophus::Plane3d& plane,
    Sophus::SE3d& T_camera_target,
    std::vector<double>& distortionParams,
    std::vector<int>& inliersIndx) {
  ceres::LossFunction* loss_function = nullptr; // new ceres::CauchyLoss(1.0);
  size_t numberNeighbour = neighbourDots.size();
  ceres::Problem::Options options;
  std::vector<double*> params = {T_camera_target.data()};
  std::vector<double*> paramsDistortion = {distortionParams.data()};

  ceres::Problem problemDistortion(options);
  problemDistortion.AddParameterBlock(params[0], Sophus::SE3d::num_parameters, new LocalParamSe3);

  for (size_t j = 0; j < numberNeighbour; ++j) {
    for (const int& i : sampleIndx) {
      ceres::CostFunction* cost = new ceres::
          AutoDiffCostFunction<isometricPatternDistortionCost, 1, Sophus::SE3d::num_parameters, 4>(
              new isometricPatternDistortionCost(
                  imageDots_.col(i), neighbourDots.at(j).col(i), centerXY_, focalLength_, plane));
      problemDistortion.AddResidualBlock(cost, loss_function, params[0], paramsDistortion[0]);
    }
  }
  ceres::Solver::Summary summary;
  ceres::Solve(solverOption, &problemDistortion, &summary);

  if (!summary.IsSolutionUsable())
    return false;
  inliersIndx = findInliers(
      neighbourDots,
      T_camera_target,
      Eigen::Map<Eigen::Vector4d>(distortionParams.data()),
      inlierThreshold);
  if (T_camera_target.inverse().translation().z() < 0) {
    T_camera_target.translation().z() = -T_camera_target.translation().z();
  } // make sure the camera is on top of the target (not behind the target)
  if (T_camera_target.translation().z() < 0) {
    T_camera_target *= Sophus::SE3d::rotX(M_PI);
  } // make sure the camera is facing the target
  return true;
}

void HexGridFitting::findPoseAndCamModel(
    const ceres::Solver::Options& solverOption,
    const Sophus::SE3d& initT_camera_target) {
  Sophus::Plane3d plane(Sophus::Vector3d(0.0, 0.0, 1.0), 0.0);
  std::vector<Eigen::Matrix2Xd> neighbourDots = imageNeighbourMatrix(numNeighboursForPoseEst_);
  std::vector<Sophus::SE3d> Ts_camera_targetForSubregions;

  // blocks
  double maxX = imageDots_.row(0).maxCoeff();
  double minX = imageDots_.row(0).minCoeff();
  double maxY = imageDots_.row(1).maxCoeff();
  double minY = imageDots_.row(1).minCoeff();
  std::vector<double> segX;
  std::vector<double> segY;
  for (int i = 0; i <= numberBlock_; i++) {
    segX.push_back(minX + (maxX - minX) / numberBlock_ * i);
    segY.push_back(minY + (maxY - minY) / numberBlock_ * i);
  }
  std::vector<std::vector<int>> blockIndx(numberBlock_ * numberBlock_);
  for (size_t i = 0; i < imageDots_.cols(); ++i) {
    for (int x = 0; x < numberBlock_; x++) {
      for (int y = 0; y < numberBlock_; y++) {
        if (imageDots_(0, i) >= segX[x] && imageDots_(0, i) < segX[x + 1] &&
            imageDots_(1, i) >= segY[y] && imageDots_(1, i) < segY[y + 1])
          blockIndx[x * numberBlock_ + y].push_back(i);
      }
    }
  }

  std::vector<std::vector<int>> inliersIndx(blockIndx.size());
  int maxInlier = 0;
  int bestIndx;
  for (int i = 0; i < blockIndx.size(); ++i) {
    Ts_camera_targetForSubregions.push_back(initT_camera_target); // initialization
    if (findT_camera_target(
            solverOption,
            neighbourDots,
            blockIndx[i],
            0.2,
            plane,
            Ts_camera_targetForSubregions[i],
            inliersIndx[i])) {
      if (inliersIndx[i].size() > maxInlier) {
        maxInlier = inliersIndx[i].size();
        bestIndx = i;
      }
    }
  }

  // global re-estimate
  T_camera_target_ = Ts_camera_targetForSubregions[bestIndx];
  std::vector<int> inliersPoseIndx;
  bool ifFindTCameraTarget = findT_camera_target(
      solverOption,
      neighbourDots,
      inliersIndx[bestIndx],
      0.3,
      plane,
      T_camera_target_,
      inliersPoseIndx);

  CHECK(ifFindTCameraTarget) << "Cannot find T_camera_target";
  CHECK_GT(inliersPoseIndx.size(), 0)
      << "No inliers for T_camera_target. You can try with different focal length.";

  inlierPose.resize(2, inliersPoseIndx.size());
  for (size_t i = 0; i < inliersPoseIndx.size(); ++i) {
    inlierPose.col(i) = imageDots_.col(inliersPoseIndx[i]);
  }
  if (ifDistort_) {
    std::vector<int> inliersDistortIndx;
    std::vector<double> distortionParams = {0.0, 0.0, 0.0, 0.0};
    if (findKb3DistortionParams(
            solverOption,
            neighbourDots,
            inliersPoseIndx,
            0.3,
            plane,
            T_camera_target_,
            distortionParams,
            inliersDistortIndx)) {
      distortionParams_ = Eigen::Map<Eigen::Vector4d>(distortionParams.data());

      // fmt::print("Distortion parameters: {} \n", distortionParams_.transpose());
      inlierDistortion.resize(2, inliersDistortIndx.size());
      for (size_t i = 0; i < inliersDistortIndx.size(); ++i) {
        inlierDistortion.col(i) = imageDots_.col(inliersDistortIndx[i]);
      }
    } else {
      LOG(INFO)
          << "Warning: Cannot find distortion parameters. Please try with different focal length";
    }
  }
  // LOG(INFO) << fmt::format("T_camera_target:\n{} \n", T_camera_target_.matrix());
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
      intrinsics << focalLength_, focalLength_, centerXY_(0), centerXY_(1), distortionParams(0),
          distortionParams(1), distortionParams(2), distortionParams(3);
      rayInCamera = KannalaBrandtK3Projection::unproject(imageDots.col(i), intrinsics);
    } else {
      Sophus::Vector<double, PinholeProjection::kNumParams> intrinsics;
      intrinsics << focalLength_, focalLength_, centerXY_(0), centerXY_(1);
      rayInCamera = PinholeProjection::unproject(imageDots.col(i), intrinsics);
    }
    Ray3d rayInTarget = T_camera_target.inverse() * rayInCamera;
    Sophus::Vector3d ptTarget3d = rayInTarget.line().intersectionPoint(plane);
    result.col(i) = ptTarget3d({0, 1});
  }
  return result;
}

void HexGridFitting::getStorageMap() {
  Eigen::Matrix3Xi cubeCoor;
  cubeCoor.resize(3, transferDots_.cols());
  cubeCoor.fill(std::numeric_limits<int>::max()); // not detected =max int
  std::queue<int> bfsQueue;
  Eigen::VectorXi processed;
  processed.setZero(transferDots_.cols(), 1);

  Eigen::Matrix3Xi cubedirection;
  cubedirection.resize(3, IsometricGridDot::kNumNeighbours);
  cubedirection << -1, 0, 1, 1, 0, -1, 1, 1, 0, -1, -1, 0, 0, -1, -1, 0, 1, 1;

  // start from the center
  Eigen::Vector2d center;
  center(0) = transferDots_.row(0).mean();
  center(1) = transferDots_.row(1).mean();
  Eigen::VectorXi centerNeighbour;
  neighboursIdxInArea(
      transferDots_, center, 2 * spacing_, centerNeighbour); // find a point near the center
  int startIdx = centerNeighbour(0);
  searchDirectionsOnPattern_ = getDirections(startIdx);

  // get cube coordinates and binary code
  binaryCode_ = Eigen::VectorXi::Constant(transferDots_.cols(), 1, 2);
  processed(startIdx) = 1;
  cubeCoor.col(startIdx) << 0, 0, 0;
  bfsQueue.push(startIdx);
  int minX = 0;
  int maxX = 0;
  int minZ = 0;
  int maxZ = 0;
  while (!bfsQueue.empty()) {
    int centerIndx = bfsQueue.front();
    bfsQueue.pop();
    for (int k = 0; k < IsometricGridDot::kNumNeighbours; k++) {
      Eigen::VectorXi Indx;
      Eigen::Vector2d possLocation =
          transferDots_.col(centerIndx) + searchDirectionsOnPattern_.col(k);
      if (neighboursIdxInArea(transferDots_, possLocation, perPointSearchRadius_, Indx)) {
        if (processed(Indx(0)) == 0) {
          // check if the point is near the possible location
          bfsQueue.push(Indx(0));
          bfsProcessSeq_.push_back(Indx(0));
          processed(Indx(0)) = 1;
          cubeCoor.col(Indx(0)) = cubeCoor.col(centerIndx) + cubedirection.col(k);
          if (minX > cubeCoor(0, Indx(0)))
            minX = cubeCoor(0, Indx(0));
          if (minZ > cubeCoor(2, Indx(0)))
            minZ = cubeCoor(2, Indx(0));
          if (maxX < cubeCoor(0, Indx(0)))
            maxX = cubeCoor(0, Indx(0));
          if (maxZ < cubeCoor(2, Indx(0)))
            maxZ = cubeCoor(2, Indx(0));
        } // end if
      } // end if
    } // end for
  } // end while

  if (processed.sum() < transferDots_.cols())
    LOG(INFO) << fmt::format(
        "{} points are processed in BFS from total {} detected points.",
        processed.sum(),
        transferDots_.cols());

  int storageMapRow = maxZ - minZ > maxX - minX ? maxZ - minZ + 1 : maxX - minX + 1;
  detectPattern_ =
      Eigen::MatrixXi::Constant(storageMapRow, storageMapRow, 2); // not detected pt in Pattern =2
  indexMap_.setConstant(storageMapRow, storageMapRow, -1); // not detected pt in index map = -1

  for (int i = 0; i < cubeCoor.cols(); ++i) {
    if (cubeCoor(0, i) != std::numeric_limits<int>::max()) {
      indexMap_(cubeCoor(2, i) - minZ, cubeCoor(0, i) - minX) = i;
    }
  }

  for (int i = 0; i < cubeCoor.cols(); ++i) {
    if (cubeCoor(0, i) != std::numeric_limits<int>::max()) {
      Eigen::Vector2i centerRQ = cubeCoor({2, 0}, i);
      centerRQ(0) -= minZ;
      centerRQ(1) -= minX;
      binaryCode_(i) = getBinarycode(centerRQ, numNeighbourLayer_);
      detectPattern_(cubeCoor(2, i) - minZ, cubeCoor(0, i) - minX) = binaryCode_(i);
    }
  }
}

bool HexGridFitting::neighboursIdxInArea(
    const Eigen::Matrix2Xd& dotMatrix,
    const Eigen::Vector2d& center,
    double searchRadius,
    Eigen::VectorXi& result) {
  bool flag = false;
  std::vector<std::pair<double, int>> distanceIndxPair;
  Eigen::VectorXd distance =
      ((dotMatrix.row(0) - Eigen::MatrixXd::Constant(1, dotMatrix.cols(), center(0))).cwiseAbs2() +
       (dotMatrix.row(1) - Eigen::MatrixXd::Constant(1, dotMatrix.cols(), center(1))).cwiseAbs2())
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
    int layer) { // the number of layers that the neighours are used to deternmine the binary
                 // code of the center
  std::vector<double> colorNeighbour;
  for (int r = -layer; r <= layer; ++r) {
    for (int q = -layer; q <= layer; ++q) {
      if (r + q >= -layer && r + q <= layer && r + centerRQ.x() >= 0 &&
          r + centerRQ.x() < indexMap_.rows() && q + centerRQ.y() >= 0 &&
          q + centerRQ.y() < indexMap_.cols()) {
        if (indexMap_(r + centerRQ.x(), q + centerRQ.y()) != -1)
          colorNeighbour.push_back(intensity_(indexMap_(r + centerRQ.x(), q + centerRQ.y())));
      }
    } // end c
  } // end r
  if (colorNeighbour.size() > 2) { // more than two neighbours
    std::sort(colorNeighbour.begin(), colorNeighbour.end());
    double colorMedian = (*colorNeighbour.begin() + colorNeighbour.back()) / 2.0;
    return intensity_(indexMap_(centerRQ.x(), centerRQ.y())) > colorMedian ? 1 : 0;
  } else {
    return 2;
  }
}

Eigen::Matrix2Xd HexGridFitting::getDirections(int startIndx) {
  Eigen::VectorXi neighbourIndx;
  neighboursIdxInArea(
      transferDots_, transferDots_.col(startIndx), spacing_ + perPointSearchRadius_, neighbourIndx);
  Eigen::Matrix2Xd result;
  result.resize(2, IsometricGridDot::kNumNeighbours);
  result.col(0) = transferDots_.col(neighbourIndx(1)) -
      transferDots_.col(startIndx); // neighbourIndx(0) is the point itself
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
    const Eigen::MatrixXi& pattern,
    Eigen::Vector2i& bestOffset,
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
            if (r + offsetR >= 0 && r + offsetR < referenceR && c + offsetC >= 0 &&
                c + offsetC < referenceC) {
              if (patternReference[patternIndx](r + offsetR, c + offsetC) == pattern(r, c) &&
                  pattern(r, c) != 2)
                numberMatch += 1;
            } // end if
          } // end c
        } // end r
        if (numberMatch > bestMatch) {
          bestMatch = numberMatch;
          bestOffset << offsetR, offsetC;
          bestIndx = patternIndx;
        }
      } // end offsetC
    } // end offsetR
  }
  if ((double(bestMatch) / double(bfsProcessSeq_.size())) < 0.7) {
    LOG(INFO) << fmt::format(
        "Pattern matching failed, only {} points can be matched from {} processed points.",
        bestMatch,
        bfsProcessSeq_.size());
  }

  return bestMatch;
}

} // namespace surreal_opensource
