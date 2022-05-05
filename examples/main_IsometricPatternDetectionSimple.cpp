/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <IsometricPatternMatcher/Image.h>
#include <IsometricPatternMatcher/PatternMatcherIsometric.h>
#include <CLI/CLI.hpp>
#define DEFAULT_LOG_CHANNEL "IsometricGrid"

namespace surreal_opensource {

struct InputArgs {
  std::string inputFile;
  std::string patternSVG;
  std::string outputPath;
  double focalLength = 200;
  bool ifDistorted = false;
};

struct singleFrameResult {
  const Eigen::MatrixXd pattern;
  const Eigen::Matrix2Xd detectCorrespondence;
  singleFrameResult(const Eigen::MatrixXd& patternPoints,
                    const Eigen::Matrix2Xd& detectPoints)
      : pattern(patternPoints), detectCorrespondence(detectPoints) {}
};

void run(const InputArgs& args) {
  PatternMatcherIsometric::IsometricOpts isometricOpts;
  isometricOpts.focalLength = args.focalLength;
  isometricOpts.ifDistort = args.ifDistorted;

  ManagedImage<uint8_t> image = LoadImage<uint8_t>(args.inputFile);

  std::unique_ptr<PatternMatcherIsometric> matcher =
      std::make_unique<PatternMatcherIsometric>(args.patternSVG, isometricOpts);

  PatternMatcherIsometric::Result res = matcher->Match(image);
  std::vector<std::shared_ptr<const IsometricGridDot>> pat =
      matcher->GetPatterns();
  singleFrameResult detectedResult(pat.at(0).get()->GetPattern(),
                                   res.detections[0].correspondences);

  if (!args.outputPath.empty()) {
    std::ofstream ofs(args.outputPath.c_str());
    for (int i = 0; i < detectedResult.pattern.cols(); ++i) {
      if (!detectedResult.detectCorrespondence.col(i).hasNaN()) {
        ofs << "coor on pattern: (" << detectedResult.pattern(0, i) << ","
            << detectedResult.pattern(1, i) << "), ";
        ofs << "coor in camera space: ("
            << detectedResult.detectCorrespondence(0, i) << ","
            << detectedResult.detectCorrespondence(1, i) << ");" << std::endl;
      }
    }
  }
}  // end run
}  // namespace surreal_opensource

int main(int argc, char** argv) {
  surreal_opensource::InputArgs args;
  CLI::App app{"IsometricGridDetection"};

  app.add_option("-i,--input-image", args.inputFile, "Input image.")
      ->required();
  app.add_option("--pattern-file", args.patternSVG, "Pattern svg file.")
      ->required();
  app.add_option("--focal_length_in_pixels", args.focalLength,
                 "Focal length in pixels of the camera (optional).");
  app.add_option("--if_distorted_images", args.ifDistorted,
                 "If the input images are distored (default = false).");

  app.add_option("--output_dir", args.outputPath,
                 "directory for outputting the pattern and corresponding "
                 "detected points.")
      ->required();

  CLI11_PARSE(app, argc, argv);
  surreal_opensource::run(args);
  return 0;
}
