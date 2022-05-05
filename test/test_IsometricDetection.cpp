/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <IsometricPatternMatcher/CameraModels.h>
#include <IsometricPatternMatcher/IsometricPattern.h>
#include <IsometricPatternMatcher/PatternMatcherIsometric.h>
#include <gtest/gtest.h>

namespace surreal_opensource {
void generatePattern(int numberLayer, Eigen::Matrix3Xd& gridOnPattern,
                     Eigen::MatrixXi& indexMap, Eigen::MatrixXi& storageMap) {
  double horizontalSpacing = 0.1;
  double verticalSpacing = 0.09;
  gridOnPattern.resize(3, 3 * numberLayer * numberLayer + 3 * numberLayer + 1);
  int storageMapRows = 2 * numberLayer + 1;
  storageMap = Eigen::MatrixXi::Constant(storageMapRows, storageMapRows, 2);
  indexMap = Eigen::MatrixXi::Constant(storageMapRows, storageMapRows, -1);
  int indx = -1;
  for (int r = 0; r < storageMapRows; ++r) {
    double y = (r - numberLayer) * verticalSpacing;
    for (int c = 0; c < storageMapRows; ++c) {
      if (r + c >= numberLayer && r + c <= 3 * numberLayer) {
        double x = (c + (r - numberLayer) / 2.0) * horizontalSpacing -
                   numberLayer * horizontalSpacing;
        indx++;
        gridOnPattern.col(indx) << x, y, 1;
        indexMap(c, r) = indx;
        storageMap(c, r) = rand() % 2;
      }
    }
  }
}

TEST(IsometricGridDetection, findPoseAndCamModel) {
  int numberLayer = 6;
  int storageMapRows = 2 * numberLayer + 1;
  int numberDot = 3 * numberLayer * numberLayer + 3 * numberLayer + 1;
  Eigen::Matrix3Xd gridOnPattern;
  Eigen::MatrixXi indexMap;
  Eigen::MatrixXi storageMap;
  generatePattern(numberLayer, gridOnPattern, indexMap, storageMap);
  // get neighbours for testing their distance in transferred space
  Eigen::MatrixXi neighbourIndx = Eigen::MatrixXi::Constant(6, numberDot, -1);
  Eigen::Matrix2Xi neighbourDirection;
  neighbourDirection.resize(2, 6);
  neighbourDirection << 0, -1, -1, 0, 1, 1, -1, 0, 1, 1, 0, -1;
  for (int r = 0; r < storageMapRows; ++r) {
    for (int c = 0; c < storageMapRows; ++c) {
      if (indexMap(r, c) != -1) {
        for (int i = 0; i < 6; ++i) {
          int neighbourR = r + neighbourDirection(0, i);
          int neighbourC = c + neighbourDirection(1, i);
          if (neighbourR >= 0 && neighbourC >= 0 &&
              neighbourR < storageMapRows && neighbourC < storageMapRows &&
              neighbourR + neighbourC >= numberLayer &&
              neighbourR + neighbourC <= 3 * numberLayer) {
            neighbourIndx(i, indexMap(r, c)) = indexMap(neighbourR, neighbourC);
          }  // end if
        }    // end for
      }      // end if
    }
  }

  // project the points according to a given camera model
  KannalaBrandtK3Projection kb3Model;
  Sophus::Vector<double, KannalaBrandtK3Projection::kNumParams> intrinsics;
  intrinsics << 220, 220, 320, 240, 0.021880085183574073, 0.10384753249118664,
      -0.34375688996267995, 0.24576059052046981;
  Sophus::SE3d T_camera_target = Sophus::SE3d::trans(0.15, 0.08, 0.5) *
                                 Sophus::SE3d::rotX(10 * M_PI / 180);
  Eigen::Matrix2Xd imageDots;
  imageDots.resize(2, numberDot);
  for (int i = 0; i < gridOnPattern.cols(); ++i) {
    imageDots.col(i) =
        kb3Model.project(T_camera_target * gridOnPattern.col(i), intrinsics);
  }

  bool ifDistort = true;
  HexGridFitting grid;
  Eigen::Vector2d centerXY;
  centerXY << 320, 240;
  double spacing = 1.0;
  int numNeighboursForPoseEst = 2;
  int numberSegX = 2;
  int numberSegY = 2;
  grid.setParams(centerXY, 220.0, ifDistort, false, false, spacing,
                 numNeighboursForPoseEst, numberSegX,
                 numberSegY);  // we don't need to test on two-shot case
  grid.setImageDots(imageDots);
  ceres::Solver::Options solverOptions;
  grid.findPoseAndCamModel(solverOptions);

  Eigen::Matrix2Xd TargetSpaceDots = grid.reprojectDots(
      grid.T_camera_target(), grid.distortionParams(), imageDots);

  for (int i = 0; i < TargetSpaceDots.cols(); ++i) {
    for (int j = 0; j < 6; ++j) {
      if (neighbourIndx(j, i) != -1) {
        EXPECT_NEAR(
            (TargetSpaceDots.col(i) - TargetSpaceDots.col(neighbourIndx(j, i)))
                .norm(),
            1.0, 0.05);
      }
    }
  }
}

TEST(IsometricGridDetection, getStorageMap) {
  Eigen::Matrix3Xd gridOnPattern;
  Eigen::MatrixXi indexMap;
  Eigen::MatrixXi storageMap;
  generatePattern(2, gridOnPattern, indexMap, storageMap);
  HexGridFitting grid;
  grid.setTransferedDots(gridOnPattern.topRows(2));
  grid.setIntensity(Eigen::VectorXd::Zero(
      gridOnPattern.cols()));  // the value is not used in this test function
  grid.setParams(Eigen::VectorXd::Zero(2), 220.0,
                 false);  // the value is not used in this test function

  grid.getStorageMap();
  Eigen::MatrixXi estIndexMap = grid.indexMap();

  IsometricGridDot rotateHex;
  std::array<Eigen::MatrixXi, 6> rotatedIndexMap =
      rotateHex.makeIsometricPatternGroup(estIndexMap);
  bool ifMatch = false;
  for (int i = 0; i < 6; ++i) {
    if (rotatedIndexMap[i] == indexMap) {
      ifMatch = true;
    }
  }
  EXPECT_TRUE(ifMatch) << "The reconstructed indexMap is not correct.\n";
}

TEST(IsometricGridDetection, findOffset) {
  int numberLayer = 3;
  int rotateIndx = 3;
  int shiftX = 1;
  int shiftY = 2;
  Eigen::MatrixXi storageMap;
  int storageMapRows = 2 * numberLayer + 1;
  Eigen::Matrix3Xd gridOnPattern;
  Eigen::MatrixXi indexMap;
  generatePattern(numberLayer, gridOnPattern, indexMap, storageMap);
  IsometricGridDot grid;
  std::array<Eigen::MatrixXi, 6> rotatedMap =
      grid.makeIsometricPatternGroup(storageMap);
  Eigen::MatrixXi pattern;
  pattern = rotatedMap.at(rotateIndx)
                .block(shiftX, shiftY, storageMapRows - shiftX,
                       storageMapRows - shiftY);

  HexGridFitting gridFitting;
  Eigen::Vector2i estOffset;
  int estRotateIndx;
  gridFitting.findOffset(rotatedMap, pattern, estOffset, estRotateIndx);
  EXPECT_TRUE(estOffset[0] == shiftX)
      << "The estiamted offsetX is not correct.\n"
      << estOffset[0];
  EXPECT_TRUE(estOffset[1] == shiftY)
      << "The estiamted offsetY is not correct.\n"
      << estOffset[1];
  EXPECT_TRUE(estRotateIndx == rotateIndx)
      << "The estiamted rotation is not correct.\n"
      << estRotateIndx;
}

TEST(IsometricGridDetection, getBinarycode) {
  int numberLayer = 2;
  Eigen::Matrix3Xd gridOnPattern;
  Eigen::MatrixXi indexMap;
  Eigen::MatrixXi storageMap;
  int storageMapRows = 2 * numberLayer + 1;
  generatePattern(numberLayer, gridOnPattern, indexMap, storageMap);
  Eigen::VectorXd intensity;
  intensity.resize(gridOnPattern.cols(), 1);
  for (int r = 0; r < storageMapRows; ++r) {
    for (int c = 0; c < storageMapRows; ++c) {
      if (indexMap(r, c) != -1) {
        intensity(indexMap(r, c)) = storageMap(r, c);
      }
    }
  }

  HexGridFitting grid;
  grid.setIntensity(intensity);
  grid.setIndexMap(indexMap);
  Eigen::MatrixXi detectedPattern =
      Eigen::MatrixXi::Constant(storageMapRows, storageMapRows, 2);
  for (int r = 0; r < storageMapRows; ++r) {
    for (int c = 0; c < storageMapRows; ++c) {
      if (indexMap(r, c) != -1) {
        Eigen::Vector2i centerRQ;
        centerRQ << r, c;
        detectedPattern(r, c) = grid.getBinarycode(centerRQ, 2);
      }
    }
  }
  EXPECT_TRUE(storageMap == detectedPattern)
      << "The binary code determination is not correct.\n storgaeMap:\n"
      << storageMap << "\n binaryCode:\n"
      << detectedPattern << std::endl;
}

}  // namespace surreal_opensource
