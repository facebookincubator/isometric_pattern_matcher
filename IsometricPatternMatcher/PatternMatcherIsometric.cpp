// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <IsometricPatternMatcher/DotExtractor.h>
#include <IsometricPatternMatcher/HexGridCostFunction.h>
#include <IsometricPatternMatcher/PatternMatcherIsometric.h>
#include <glog/logging.h>

namespace surreal_opensource {
Eigen::Matrix2Xd PatternMatcherIsometric::DotDetection(
    const Image<uint8_t>& image) const {
  DotExtractor32 extractor;
  ManagedImage<DotTypeFloat> dots;
  uint32_t numDots;
  Eigen::AlignedBox2i roi(
      Eigen::Vector2i(10, 10),
      Eigen::Vector2i(image.Width() - 10, image.Height() - 10));
  extractor.setHessThresh(opts_.hessThresh);
  extractor.setBlurKernelRadius(opts_.blurKernelRadius);
  extractor.loadIrImage(image);
  extractor.extractDots(roi);
  extractor.copyDetectedDots(dots, numDots);
  Eigen::Matrix2Xd detectedDots;
  detectedDots.resize(2, numDots);
  for (int i = 0; i < numDots; ++i) {
    detectedDots(0, i) = dots[i](0);
    detectedDots(1, i) = dots[i](1);
  }
  return detectedDots;
}

PatternMatcherIsometric::PatternMatcherIsometric(
    std::vector<std::shared_ptr<const IsometricGridDot>>& targets,
    const IsometricOpts& opts)
    : isometricGrids_(targets), opts_(opts) {
  // TODO: HACK: currently only support one pattern
  CHECK_EQ(isometricGrids_.size(), 1) << "Currently only supporting 1 target!";
}

PatternMatcherIsometric::PatternMatcherIsometric(
    const std::string& patternFiles, const IsometricOpts& opts)
    : opts_(opts) {
  CHECK_GT(patternFiles.size(), 0) << fmt::format(
      "Matcher needs at least one target, {} specified", patternFiles.size());

  isometricGrids_.push_back(std::make_shared<IsometricGridDot>(patternFiles));
}

std::vector<std::shared_ptr<const IsometricGridDot>>
PatternMatcherIsometric::GetPatterns() const {
  return isometricGrids_;
}

Eigen::VectorXd PatternMatcherIsometric::GetIntensity(
    const Eigen::Matrix2Xd& detectedDots, const Image<uint8_t>& imageU8) const {
  Eigen::VectorXd intensity;
  intensity.resize(detectedDots.cols());
  for (int i = 0; i < detectedDots.cols(); ++i) {
    double max = 0;
    for (int r = -opts_.intensityWindowSize / 2;
         r <= opts_.intensityWindowSize / 2; ++r) {
      for (int c = -opts_.intensityWindowSize / 2;
           c <= opts_.intensityWindowSize / 2; ++c) {
        const auto* r0 = imageU8.RowPtr((int)detectedDots(1, i) + r) +
                         (int)detectedDots(0, i) + c;
        if (*r0 > max) {
          max = *r0;
        }
      }
    }
    intensity(i) = max;
  }
  return intensity;
}

void PatternMatcherIsometric::StoreIntoMap(const HexGridFitting& grid,
                                           const Eigen::Matrix2Xd& detectedDots,
                                           PatternMatcherIsometric::Result& res,
                                           int& rotationIndx,
                                           Eigen::Vector2i& offset) const {
  Eigen::VectorXi binaryCode = grid.binaryCode();
  res.debug.detected_labels.reserve(binaryCode.size());
  for (int i = 0; i < binaryCode.size(); ++i) {
    res.debug.detected_labels.push_back(binaryCode(i));
  }
  int numberMatch = grid.findOffset(isometricGrids_[0]->GetBinaryPatternGroup(),
                                    grid.detectPattern(), offset, rotationIndx);
  fmt::print("{} points can be matched from total {} detected points \n",
             numberMatch, detectedDots.cols());
}

void PatternMatcherIsometric::generateResult(
    const HexGridFitting& grid, const Eigen::Matrix2Xd& detectedDots,
    int rotationIndx, const Eigen::Vector2i& offset,
    PatternMatcherIsometric::Result& res) const {
  Eigen::MatrixXi referenceIndxMap;
  size_t storageMapRows = isometricGrids_[0]->storageMapRows();
  referenceIndxMap.resize(storageMapRows, storageMapRows);
  for (int r = 0; r < storageMapRows; ++r) {
    for (int c = 0; c < storageMapRows; ++c)
      referenceIndxMap(r, c) = r * storageMapRows + c;
  }
  fmt::print("rotationIndx: {}, offset: ({}, {}).\n", rotationIndx, offset.x(),
             offset.y());
  for (int i = 0; i < rotationIndx; ++i) {
    referenceIndxMap = isometricGrids_[0]->Rotate60Right(referenceIndxMap);
  }
  Eigen::Matrix<double, 2, Eigen::Dynamic> correspondences;
  correspondences.resize(2, isometricGrids_[0]->GetPattern().cols());
  correspondences.setConstant(std::numeric_limits<double>::quiet_NaN());
  Eigen::MatrixXi detectedIndxMap = grid.indexMap();
  int numVisualizedDot = 0;
  for (int r = 0; r < detectedIndxMap.rows(); ++r) {
    for (int c = 0; c < detectedIndxMap.cols(); ++c) {
      if (r + offset.x() >= 0 && r + offset.x() < storageMapRows &&
          c + offset.y() >= 0 && c + offset.y() < storageMapRows) {
        const int id = referenceIndxMap(r + offset.x(), c + offset.y());
        if (detectedIndxMap(r, c) >= 0) {
          CHECK(id < correspondences.cols());
          if (isometricGrids_[0]->GetBinaryPatternGroup()[rotationIndx](
                  r + offset.x(), c + offset.y()) ==
                  grid.detectPattern()(r, c) &&
              grid.detectPattern()(r, c) != 2) {
            correspondences.col(id) = detectedDots.col(detectedIndxMap(r, c));
            numVisualizedDot += 1;
          }
        }
      }  // end if
    }    // end c
  }      // end r
  res.detections.emplace_back(0, correspondences);
  std::cout << "Visualize matched point number is: " << numVisualizedDot
            << std::endl;
}

PatternMatcherIsometric::Result PatternMatcherIsometric::Match(
    const Image<uint8_t>& imageU8) const {
  PatternMatcherIsometric::Result res;
  // detect dots
  Eigen::Matrix2Xd detectedDots = DotDetection(imageU8);

  for (int i = 0; i < detectedDots.cols(); ++i) {
    res.debug.feature_pts.push_back(detectedDots.col(i));
  }

  // get intensity of the extracted dots
  Eigen::VectorXd intensity = GetIntensity(detectedDots, imageU8);

  // detect pattern from the extracted dot
  Eigen::Vector2d centerXY;
  centerXY << imageU8.w / 2, imageU8.h / 2;
  double spacing = 1.0;
  int numNeighboursForPoseEst = 3;
  int numberBlock = 3;  // devide the dots into numberBlock*numberBlock patches
  HexGridFitting grid(detectedDots, centerXY, opts_.focalLength, intensity,
                      opts_.ifDistort, false, false, spacing,
                      numNeighboursForPoseEst, numberBlock);

  // store detected pattern into a storagemap
  Eigen::Vector2i offset;
  int rotationIndx;
  StoreIntoMap(grid, detectedDots, res, rotationIndx, offset);

  // Generate result
  generateResult(grid, detectedDots, rotationIndx, offset, res);

  return res;
}

PatternMatcherIsometric::Result PatternMatcherIsometric::MatchImagePairs(
    const Image<uint8_t>& imageCode1U8,
    const Image<uint8_t>& imageCode0U8) const {
  CHECK(((imageCode1U8.w == imageCode0U8.w) &&
         (imageCode1U8.h == imageCode0U8.h)))
      << "imageCode1 and imageCode0 should have same size";
  // detect dots
  PatternMatcherIsometric::Result res;
  Eigen::Matrix2Xd detectedCode1Dots = DotDetection(imageCode1U8);
  Eigen::Matrix2Xd detectedCode0Dots = DotDetection(imageCode0U8);
  Eigen::Matrix2Xd detectedDots(
      detectedCode1Dots.rows(),
      detectedCode1Dots.cols() + detectedCode0Dots.cols());
  detectedDots << detectedCode1Dots, detectedCode0Dots;
  for (int i = 0; i < detectedCode1Dots.cols(); ++i) {
    res.debug.feature_pts.push_back(detectedCode1Dots.col(i));
  }
  for (int i = 0; i < detectedCode0Dots.cols(); ++i) {
    res.debug.feature_pts.push_back(detectedCode0Dots.col(i));
  }

  // get intensity of the extracted dots
  Eigen::VectorXd intensityCode1 =
      GetIntensity(detectedCode1Dots, imageCode1U8);
  Eigen::VectorXd intensityCode0 =
      GetIntensity(detectedCode0Dots, imageCode0U8);
  CHECK(intensityCode1.rows() == detectedCode1Dots.cols())
      << "intensity and dot position length not consistent";
  CHECK(intensityCode1.rows() == detectedCode1Dots.cols())
      << "intensity and dot position length not consistent";

  // build dot code labels
  Eigen::VectorXi dotLabels(intensityCode1.rows() + intensityCode0.rows());
  Eigen::VectorXi labelOne(intensityCode1.rows());
  labelOne.setOnes();
  Eigen::VectorXi labelZero(intensityCode0.rows());
  labelZero.setZero();
  dotLabels << labelOne, labelZero;

  // detect pattern from the extracted dots
  Eigen::Vector2d centerXY;
  centerXY << imageCode1U8.w / 2, imageCode1U8.h / 2;
  double spacing = 1.0;
  int numNeighboursForPoseEst = 3;
  HexGridFitting grid(detectedDots, centerXY, opts_.focalLength, dotLabels,
                      opts_.ifDistort, true, opts_.ifPoseMerge,
                      opts_.goodPoseInlierRatio, spacing,
                      numNeighboursForPoseEst, opts_.numberBlock);

  // store detected pattern into a storagemap
  Eigen::Vector2i offset;
  int rotationIndx;
  StoreIntoMap(grid, detectedDots, res, rotationIndx, offset);

  // Generate result
  generateResult(grid, detectedDots, rotationIndx, offset, res);

  return res;
}

PatternMatcherIsometric::Result
PatternMatcherIsometric::MatchImagePairsWithConics(
    const Image<uint8_t>& imageCode1U8, const Image<uint8_t>& imageCode0U8,
    const Eigen::Matrix2Xd& detectedCode1Dots,
    const Eigen::Matrix2Xd& detectedCode0Dots) const {
  CHECK(((imageCode1U8.w == imageCode0U8.w) &&
         (imageCode1U8.h == imageCode0U8.h)))
      << "imageCode1 and imageCode0 should have same size";
  PatternMatcherIsometric::Result res;
  // process detected dots
  Eigen::Matrix2Xd detectedDots(
      detectedCode1Dots.rows(),
      detectedCode1Dots.cols() + detectedCode0Dots.cols());
  detectedDots << detectedCode1Dots, detectedCode0Dots;
  for (int i = 0; i < detectedCode1Dots.cols(); ++i) {
    res.debug.feature_pts.push_back(detectedCode1Dots.col(i));
  }
  for (int i = 0; i < detectedCode0Dots.cols(); ++i) {
    res.debug.feature_pts.push_back(detectedCode0Dots.col(i));
  }

  // get intensity of the extracted dots
  Eigen::VectorXd intensityCode1 =
      GetIntensity(detectedCode1Dots, imageCode1U8);
  Eigen::VectorXd intensityCode0 =
      GetIntensity(detectedCode0Dots, imageCode0U8);
  CHECK(intensityCode1.rows() == detectedCode1Dots.cols())
      << "intensity and dot position length not consistent";
  CHECK(intensityCode1.rows() == detectedCode1Dots.cols())
      << "intensity and dot position length not consistent";

  // build dot code labels
  Eigen::VectorXi dotLabels(intensityCode1.rows() + intensityCode0.rows());
  Eigen::VectorXi labelOne(intensityCode1.rows());
  labelOne.setOnes();
  Eigen::VectorXi labelZero(intensityCode0.rows());
  labelZero.setZero();
  dotLabels << labelOne, labelZero;
  // detect pattern from the extracted dots
  Eigen::Vector2d centerXY;
  centerXY << imageCode1U8.w / 2, imageCode1U8.h / 2;
  double spacing = 1.0;
  int numNeighboursForPoseEst = 3;
  HexGridFitting grid(detectedDots, centerXY, opts_.focalLength, dotLabels,
                      opts_.ifDistort, true, opts_.ifPoseMerge,
                      opts_.goodPoseInlierRatio, spacing,
                      numNeighboursForPoseEst, opts_.numberBlock);

  // store detected pattern into a storagemap
  Eigen::Vector2i offset;
  int rotationIndx;
  StoreIntoMap(grid, detectedDots, res, rotationIndx, offset);

  // Generate result
  generateResult(grid, detectedDots, rotationIndx, offset, res);

  return res;
}
}  // namespace surreal_opensource
