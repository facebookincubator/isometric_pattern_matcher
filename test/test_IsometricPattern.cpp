/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <IsometricPatternMatcher/IsometricPattern.h>
#include <gtest/gtest.h>
#include <random>

namespace surreal_opensource {

TEST(IsometricGridDot, TestSeed) {
  std::random_device rd;
  size_t seed = static_cast<uint32_t>(rd());
  int numberLayer = 5;
  Eigen::Vector2i gridRowsCols;
  gridRowsCols[0] = numberLayer * 2 + 1;
  gridRowsCols[1] = numberLayer * 2 + 1;

  IsometricGridDot grid1 = IsometricGridDot(
      9.0 / 1000,  // millimeters to meters
      10.0 / 1000, 1.0 / 1000, numberLayer, gridRowsCols, seed);

  IsometricGridDot grid2 = IsometricGridDot(
      9.0 / 1000,  // millimeters to meters
      10.0 / 1000, 1.0 / 1000, numberLayer, gridRowsCols, seed);

  EXPECT_TRUE(grid1.GetBinaryPattern() == grid2.GetBinaryPattern())
      << "Different patterns generated from the same seed.";
}

}  // namespace surreal_opensource
