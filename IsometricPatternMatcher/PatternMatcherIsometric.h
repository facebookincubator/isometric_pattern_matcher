// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <IsometricPatternMatcher/HexGridFitting.h>
#include <IsometricPatternMatcher/Image.h>
#include <IsometricPatternMatcher/IsometricPattern.h>

namespace surreal_opensource {
class PatternMatcherIsometric {
 public:
  // config options
  struct IsometricOpts {
    IsometricOpts()
        : focalLength(200.0),
          hessThresh(10.0),
          blurKernelRadius(2),
          intensityWindowSize(3),
          ifDistort(0) {}
    double focalLength; // in pixels
    float hessThresh;
    size_t blurKernelRadius;
    int intensityWindowSize;
    bool ifDistort;
  };
  struct Detection {
    Detection();
    Detection(size_t pattern_index, const Eigen::Matrix<double, 2, Eigen::Dynamic>& correspondences)
        : pattern_index(pattern_index), correspondences(correspondences) {}
    // Index into GetPatterns()
    size_t pattern_index;
    // Each column c is the location in image space of the c'th pattern feature.
    // NaN's mean the correspondence is not found.
    Eigen::Matrix<double, 2, Eigen::Dynamic> correspondences;
  };

  struct Debug {
    // Each column c is the location in image space some pattern
    // feature that may or may not have been detected as a valid
    // pattern
    std::vector<Eigen::Matrix<double, 2, 1>> feature_pts;
    std::vector<int> detected_labels;
    ManagedImage<uint8_t> binary_thresholded_image;
  };

  struct Result {
    std::vector<Detection> detections;
    Debug debug;
  };

  PatternMatcherIsometric(
      std::vector<std::shared_ptr<const IsometricGridDot>>& targets,
      const IsometricOpts& opts = IsometricOpts());

  PatternMatcherIsometric(
      const std::string& pattern_files, // only support one pattern
      const IsometricOpts& opts = IsometricOpts());

  std::vector<std::shared_ptr<const IsometricGridDot>> GetPatterns() const;

  Result Match(const Image<uint8_t>& image) const;

  const double focalLength() const {
    return opts_.focalLength;
  }
  Eigen::Matrix2Xd DotDetection(const Image<uint8_t>& image) const;

 private:
  std::vector<std::shared_ptr<const IsometricGridDot>> isometricGrids_;
  IsometricOpts opts_;
};
} // namespace surreal_opensource