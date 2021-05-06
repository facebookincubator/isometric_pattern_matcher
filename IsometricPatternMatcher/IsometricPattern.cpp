// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <IsometricPatternMatcher/IsometricPattern.h>
#include <glog/logging.h>
#include <pangolin/pangolin.h>
#include <fstream>
#include <random>

namespace surreal_opensource {
Eigen::MatrixXi IsometricGridDot::makeIsometricPattern(uint32_t seed) {
  Eigen::MatrixXi M(storageMapRows_, storageMapRows_);
  std::mt19937 rng(seed);

  std::uniform_int_distribution<uint32_t> uintDist1(0, 1);

  for (int r = 0; r < M.rows(); ++r) {
    int row = r - numberLayer_;
    for (int q = 0; q < M.cols(); ++q) {
      int col =
          q - numberLayer_ + (r - numberLayer_ - ((r - numberLayer_) & 1)) / 2;

      M(r, q) =
          (r + q < numberLayer_ ||
           r + q >
               3 * numberLayer_ ||  // upper and lower triangles of the matrix
           abs(row) > gridRowsCols_[0] / 2 ||
           abs(col) > gridRowsCols_[1] /
                          2)  // outside the specified rows and cols of the grid
              ? 2
              : uintDist1(rng);
    }  // upper and lower triangles of the matrix should be Null, incidated by 2
  }
  return M;
}

std::array<Eigen::MatrixXi, 6> IsometricGridDot::makeIsometricPatternGroup(
    Eigen::MatrixXi pattern0) {
  std::array<Eigen::MatrixXi, 6> outputPatternGroup;
  outputPatternGroup[0] = pattern0;
  for (int i = 1; i < 6; ++i) {
    outputPatternGroup[i] = Rotate60Right(outputPatternGroup[i - 1]);
  }
  return outputPatternGroup;
}

Eigen::MatrixXi IsometricGridDot::Rotate60Right(
    Eigen::MatrixXi& inputPatternGrid) const {
  Eigen::MatrixXi outputPatternGrid = inputPatternGrid;
  const size_t numLayer = (inputPatternGrid.rows() - 1) / 2;
  for (int r = 0; r < inputPatternGrid.rows(); ++r) {
    for (int q = 0; q < inputPatternGrid.cols(); ++q) {
      if (r + q >= numLayer && r + q <= 3 * numLayer)
        outputPatternGrid(r, q) =
            inputPatternGrid(2 * numLayer - q, r + q - numLayer);
    }  // upper and lower triangles of the matrix should be Null, incidated by 2
  }
  return outputPatternGrid;
}

void IsometricGridDot::Init() {
  // get patternPtsCodes and patternPts according to patternGroup_[0]
  patternPts_.resize(3, storageMapRows_ * storageMapRows_);
  patternPtsCodes_.resize(storageMapRows_ * storageMapRows_);
  double centerX = (gridRowsCols_[1] - 1) / 2 * horizontalSpacing_;  // in meter
  double centerY = (gridRowsCols_[0] - 1) / 2 * verticalSpacing_;
  int centerIndx = numberLayer_;
  for (int r = 0; r < patternGroup_[0].rows(); ++r) {
    double y = centerY + (r - centerIndx) * verticalSpacing_;
    for (int c = 0; c < patternGroup_[0].cols(); ++c) {
      patternPtsCodes_[r * storageMapRows_ + c] = patternGroup_[0](r, c);
      double x = centerX + ((c - centerIndx) + (r - centerIndx) / 2.0) *
                               horizontalSpacing_;
      patternPts_.col(r * storageMapRows_ + c) = Eigen::Vector3d(x, y, 0);
    }
  }
}

IsometricGridDot::IsometricGridDot(double verticalSpacing,
                                   double horizontalSpacing, double dotRadius,
                                   size_t numberLayer,
                                   const Eigen::Vector2i& gridRowsCols,
                                   uint32_t seed)
    : verticalSpacing_(verticalSpacing),
      horizontalSpacing_(horizontalSpacing),
      dotRadius_(dotRadius),
      numberLayer_(numberLayer),
      gridRowsCols_(gridRowsCols),
      storageMapRows_(numberLayer * 2 + 1) {
  // Create binary pattern (and rotated pattern) from seed
  patternGroup_ = makeIsometricPatternGroup(makeIsometricPattern(seed));
  Init();
}

IsometricGridDot::IsometricGridDot(const std::string& gridFile) {
  std::string ext = pangolin::FileLowercaseExtention(gridFile);
  if (ext == ".svg")
    LoadFromSVG(gridFile);
  else
    CHECK(false) << fmt::format(
        "Unsupported Grid FileType. Needs to be svg (grid file: {})", gridFile);
}

void IsometricGridDot::LoadFromSVG(const std::string& gridFile) {
  std::fstream svgFile;
  svgFile.open(gridFile, std::fstream::in);
  CHECK(svgFile.is_open()) << fmt::format(
      "Unable to open Target SVG file, '{}'", gridFile);
  std::string binaryPattern;

  // look for surreal parameters line
  std::string line;
  while (!svgFile.eof()) {
    getline(svgFile, line);

    if (IsXmlParamsString(line) == false) continue;

    bool success =
        ParseXmlParamsString(line, binaryPattern, numberLayer_, gridRowsCols_,
                             horizontalSpacing_, verticalSpacing_, dotRadius_);

    CHECK(success) << fmt::format("Unable to parse grid file {}",
                                  gridFile.c_str());
    break;
  }

  CHECK(!svgFile.eof()) << "Grid file {} does not contain target parameters.",
      gridFile.c_str();

  storageMapRows_ = numberLayer_ * 2 + 1;
  Eigen::MatrixXi pattern0(1, storageMapRows_ * storageMapRows_);
  for (int i = 0; i < binaryPattern.length(); i++) {
    pattern0(0, i) = binaryPattern[i] - '0';
  }
  pattern0.resize(storageMapRows_, storageMapRows_);

  patternGroup_ = makeIsometricPatternGroup(pattern0.adjoint());
  Init();
}

bool IsometricGridDot::ParseXmlParamsString(
    const std::string& s, std::string& binaryPattern, size_t& numberLayer,
    Eigen::Vector2i& gridRowsCols, double& horizontalSpacing,
    double& verticalSpacing, double& dotRadius) {
  if (!ParseOption(s, "-vertical-spacing ", verticalSpacing)) {
    LOG(ERROR) << "Xml comment in svg pattern file does not contain "
                  "-vertical-spacing field.";
    return false;
  }

  if (!ParseOption(s, "-horizontal-spacing ", horizontalSpacing)) {
    LOG(ERROR) << "Xml comment in svg pattern file does not contain "
                  "-horizontal-spacing field.";
    return false;
  }

  if (!ParseOption(s, "-layer-number ", numberLayer)) {
    LOG(ERROR) << "Xml comment in svg pattern file does not contain "
                  "-layer-number field.";
    return false;
  }

  if (!ParseOption(s, "-rows ", gridRowsCols[0])) {
    LOG(ERROR)
        << "Xml comment in svg pattern file does not contain -rows field.";
    return false;
  }

  if (!ParseOption(s, "-cols ", gridRowsCols[1])) {
    LOG(ERROR)
        << "Xml comment in svg pattern file does not contain -cols field.";
    return false;
  }

  if (!ParseOption(s, "-grid-dotRadius ", dotRadius)) {
    LOG(ERROR) << "Xml comment in svg pattern file does not contain "
                  "-grid-dotRadius field.";
    return false;
  }

  if (!ParseOption(s, "-grid-pattern ", binaryPattern)) {
    LOG(ERROR) << "Xml comment in svg pattern file does not contain "
                  "-grid-pattern field.";
    return false;
  }

  if (std::ceil((numberLayer * 2 + 1) * (numberLayer * 2 + 1)) !=
      binaryPattern.length()) {
    LOG(ERROR)
        << "Pattern string length does not agree with the specified grid size.";
    return false;
  }

  return true;
}

bool IsometricGridDot::IsXmlParamsString(std::string s) {
  return (s.find(ISOMETRIC_PATTERN_TRAILING_STRING) != std::string::npos);
}

template <typename T>
bool IsometricGridDot::ParseOption(const std::string& s, const std::string& key,
                                   T& val) {
  size_t pos = s.find(key);
  if (pos == std::string::npos) {
    return false;
  } else {
    std::stringstream ss(s.substr(pos + key.length()));
    ss >> val;
    return true;
  }
}

void IsometricGridDot::SaveSVG(
    std::string filename, const std::string& color0, const std::string& color1,
    const std::string& bgcolor,
    int patternGroupIndex)  // patternGroupIndex only for testing
    const {
  std::ofstream f(filename.c_str());
  const Eigen::MatrixXi& M = patternGroup_[patternGroupIndex];
  const double offsetInMm = horizontalSpacing_ * 1000;  // meters to millimeters
  f << "<?xml version=\"1.0\" standalone=\"no\"?>" << std::endl
    << "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" "
       "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">"
    << std::endl;

  f << ISOMETRIC_PATTERN_TRAILING_STRING << " -vertical-spacing "
    << verticalSpacing_ << "  -horizontal-spacing " << horizontalSpacing_
    << "  -layer-number " << numberLayer_ << "  -rows " << gridRowsCols_[0]
    << "  -cols " << gridRowsCols_[1] << "  -grid-dotRadius " << dotRadius_;
  f << "  -grid-pattern ";
  for (int r = 0; r < M.rows(); ++r) {
    for (int q = 0; q < M.cols(); ++q) f << M(r, q);
  }
  f << std::endl << " -matrix " << std::endl << M;
  f << " -->" << std::endl;
  // units in mm
  const double canvasWidth =
      ((gridRowsCols_[1] - 1) * horizontalSpacing_ * 1000) + offsetInMm * 2.;
  const double canvasHeight =
      ((gridRowsCols_[0] - 1) * verticalSpacing_ * 1000) + offsetInMm * 2.;

  f << fmt::format(R"#(<svg width="{}mm" height="{}mm">)#", canvasWidth,
                   canvasHeight);
  f << std::endl;
  f << fmt::format(
      R"#(<rect width="{}mm" height="{}mm" style="fill:{}" z-index="-1"/>)#",
      canvasWidth, canvasHeight, bgcolor);
  f << std::endl;
  for (int r = 0; r < M.rows(); ++r) {
    for (int q = 0; q < M.cols(); ++q) {
      if (M(r, q) == 0 || M(r, q) == 1) {
        std::string color = (M(r, q) == 1) ? color1 : color0;
        f << fmt::format(
            R"#(<circle cx="{}mm" cy="{}mm" r="{}mm" fill="{}" stroke-width="0"/>)#",
            patternPts_(0, r * M.cols() + q) * 1000 + offsetInMm,
            patternPts_(1, r * M.cols() + q) * 1000 + offsetInMm,
            dotRadius_ * 1000,  // meters to millimeters
            color);
        f << std::endl;
      }
    }
  }
  f << "</svg>" << std::endl;
}

}  // namespace surreal_opensource
