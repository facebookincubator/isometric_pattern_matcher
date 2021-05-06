// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <IsometricPatternMatcher/IsometricPattern.h>
#include <fmt/core.h>
#include <glog/logging.h>
#include <CLI/CLI.hpp>
#include <random>

namespace surreal_opensource {
struct InputArgs {
  size_t numberLayer;
  double dotRadMm;
  double horizontalSpacing;
  double verticalSpacing;
  std::string fgColor0;
  std::string fgColor1;
  std::string bgColor;
  std::string outputFileName;
  Eigen::Vector2i gridRowsCols;

  uint32_t seed;
};

void run(const InputArgs& args) {
  // Binary code

  IsometricGridDot grid =
      IsometricGridDot(args.verticalSpacing / 1000,  // millimeters to meters
                       args.horizontalSpacing / 1000, args.dotRadMm / 1000,
                       args.numberLayer, args.gridRowsCols, args.seed);

  std::string outputFileName;
  if (args.outputFileName.empty()) {
    outputFileName = "isometric_NLayer" + std::to_string(grid.NumberLayer()) +
                     '_' + std::to_string(args.seed) + ".svg";
  } else {
    outputFileName = args.outputFileName;
  }

  grid.SaveSVG(outputFileName, args.fgColor0, args.fgColor1, args.bgColor);
  LOG(INFO) << fmt::format("Writing to: {}", outputFileName);

}  // end run
}  // namespace surreal_opensource

int main(int argc, char** argv) {
  surreal_opensource::InputArgs args;
  CLI::App app{"IsometricGridGen"};

  std::random_device rd;
  size_t seed = static_cast<uint32_t>(rd());
  app.add_option("--seed", args.seed,
                 "Random seed to use to generate binary pattern. Defaults to "
                 "random seed")
      ->default_val(seed);

  app.add_option("--dot-radius", args.dotRadMm, "Radius of dots (mm).")
      ->default_val("0.2");

  app.add_option("--horizontal-spacing", args.horizontalSpacing,
                 "Horizontal distance between adjacent dots (mm)")
      ->default_val("10.0");
  app.add_option("--vertical-spacing", args.verticalSpacing,
                 "Vertical distance between adjacent dots (mm)")
      ->default_val("9.0");

  app.add_option("--number-layer", args.numberLayer, "Number of layers.")
      ->default_val(0);
  app.add_option("--rows", args.gridRowsCols[0],
                 "Number of rows in pattern (odd)")
      ->excludes("--number-layer");
  app.add_option("--cols", args.gridRowsCols[1],
                 "Number of columns in pattern (odd)")
      ->excludes("--number-layer");

  app.add_option("--color0", args.fgColor0, "Color of '0' dots.")
      ->default_val("gray");
  app.add_option("--color1", args.fgColor1, "Color of '1' dots.")
      ->default_val("white");
  app.add_option("--bg-color", args.bgColor, "Background color")
      ->default_val("black");

  app.add_option("-o,--output-svgfile", args.outputFileName,
                 "Output filename to save SVG printable pattern to "
                 "isometric_NLayer${NUMBERLAYER}_${SEED}.svg");

  CLI11_PARSE(app, argc, argv);
  if (args.numberLayer > 0) {
    args.gridRowsCols[0] = args.numberLayer * 2 + 1;
    args.gridRowsCols[1] = args.numberLayer * 2 + 1;
  } else {
    CHECK(args.gridRowsCols[0] % 2 != 0 && args.gridRowsCols[1] % 2 != 0)
        << "rows and cols should be odd numbers";
    int r = -args.gridRowsCols[0] / 2;
    int x = args.gridRowsCols[1] / 2 - (r - (r & 1)) / 2;
    args.numberLayer =
        abs(x) > abs(r) ? abs(x)
                        : abs(r);  // make sure the storage map is large enough
  }

  surreal_opensource::run(args);
  return 0;
}
