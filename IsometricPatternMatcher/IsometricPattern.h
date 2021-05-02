// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <fmt/core.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <vector>

#define ISOMETRIC_PATTERN_TRAILING_STRING "<!-- SURREAL GRID"

// more about the isometric calibration pattern can be found:
// https://docs.google.com/document/d/1lZc0-caCUlBOB10AWK4wunqBcsZU7zduUHcn8L5dBaw/edit
namespace surreal_opensource {
class IsometricGridDot {
 public:
  static const size_t kNumNeighbours = 6;

  IsometricGridDot() {}
  IsometricGridDot(double verticalSpacing, double horizontalSpacing,
                   double dotRadius, size_t numberLayer,
                   const Eigen::Vector2i& gridRowsCols, uint32_t seed);

  IsometricGridDot(const std::string& gridFile);

  Eigen::MatrixXi makeIsometricPattern(uint32_t seed);

  Eigen::MatrixXi Rotate60Right(Eigen::MatrixXi& inputPatternGrid) const;
  std::array<Eigen::MatrixXi, 6> makeIsometricPatternGroup(
      Eigen::MatrixXi pattern0);

  void LoadFromSVG(const std::string& SVGFile);

  bool IsXmlParamsString(std::string s);
  bool ParseXmlParamsString(const std::string& s, std::string& binaryPattern,
                            size_t& numberLayer, Eigen::Vector2i& gridRowsCols,
                            double& horizontalSpacing, double& verticalSpacing,
                            double& dotRadius);

  template <typename T>
  bool ParseOption(const std::string& s, const std::string& key, T& val);

  void SaveSVG(std::string filename, const std::string& color0 = "white",
               const std::string& color1 = "gray",
               const std::string& bgcolor = "black", int PGindex = 0) const;

  void Clear();

  // bool Match(std::map<Eigen::Vector2i, Vertex*>& obs) const;
  const Eigen::AlignedBoxXd GetPatternBounds() const {
    throw std::runtime_error("Not Implemented");
  }

  /**
   * Gets the set of 3D points for this pattern.
   *
   * @return The location of the calibu points as a [3 x (col * row)] matrix.
   * All points are projected to the z=0 plane.
   */
  const Eigen::MatrixXd& GetPattern() const { return patternPts_; }

  /**
   * Gets the binary code.
   *
   * @return Vector of binary codes (1 or 0), entries are aligned with the
   * points returned in the GetPattern() function.
   */
  const std::vector<int64_t>& GetPatternCodes() const {
    return patternPtsCodes_;
  }

  inline size_t NumberLayer() const { return numberLayer_; }

  inline size_t gridRows() const { return gridRowsCols_[0]; }

  inline size_t gridCols() const { return gridRowsCols_[1]; }

  inline double verticalSpacing() const { return verticalSpacing_; }

  inline double horizontalSpacing() const { return horizontalSpacing_; }

  inline double dotRadius() const { return dotRadius_; }

  inline size_t storageMapRows() const { return storageMapRows_; }

  const Eigen::MatrixXi& GetBinaryPattern(unsigned int idx = 0) const {
    return patternGroup_[idx];
  }
  const std::array<Eigen::MatrixXi, 6>& GetBinaryPatternGroup(void) const {
    return patternGroup_;
  }

 protected:
  void Init();

  Eigen::MatrixXd patternPts_;  // 3XN
  std::vector<int64_t> patternPtsCodes_;
  double verticalSpacing_;
  double horizontalSpacing_;
  std::array<Eigen::MatrixXi, 6> patternGroup_;  // 2X2 storage map
  double dotRadius_;
  size_t numberLayer_;
  Eigen::Vector2i gridRowsCols_;
  size_t storageMapRows_;
};

}  // namespace surreal_opensource
