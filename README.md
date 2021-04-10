# IsometricPatternMatcher

IsometricPatternMatcher is a C++ library. It proposes a hex-grid, intensity-modulated calibration pattern for camera intrinsics and extrinsics calibration, and have corresponding code for generating and matching the proposed pattern. Its usage is similar to that of [calibu](https://github.com/arpg/Calibu).

## Examples

1. For generating an isometric pattern (see `examples/main_IsometricPatternGeneration.cpp` for complete example):

```
    IsometricGridDot grid(
      args.verticalSpacing / 1000, // vertical spacing, should be approximate sqrt(3)/2 * horizontal spacing
      args.horizontalSpacing / 1000,    // horizontal spacing
      args.dotRadMm / 1000,     // radius of dot, should be as small as possible
      args.numberLayer,     // 1 layer has 1+6=7 dots, 2 layers have 1+6+12=19 dots, ...
      args.gridRowsCols,    // you can pick only the center square region of the hex pattern
      args.seed);   // random seed for binary encoding.
```

2. For pattern matching (see `examples/main_IsometricPatternDetectionSimple.cpp` for complete example):

```
  PatternMatcherIsometric::IsometricOpts isometricOpts;
  isometricOpts.focalLength = args.focalLength;
  isometricOpts.ifDistort = args.ifDistorted;

  std::unique_ptr<PatternMatcherIsometric> matcher =
      std::make_unique<PatternMatcherIsometric>(pattern, isometricOpts);

  PatternMatcherIsometric::Result res = matcher->Match(image);
```

## Requirements

- BUCK build tested with Linux and Mac OSX systems.
- Depends on [Pangolin](https://github.com/stevenlovegrove/Pangolin), [fmt](https://github.com/fmtlib/fmt), [glog](https://github.com/google/glog), and [ceres-solver](https://github.com/ceres-solver/ceres-solver).
- `examples` depends on [CLI11](https://github.com/CLIUtils/CLI11).
- `test` depends on [googletest](https://github.com/google/googletest).

## Building IsometricPatternMatcher

Will fill in after tested with CMake.

## Installing IsometricPatternMatcher

Will fill in after tested with CMake

## Join the IsometricPatternMatcher community

Send emails to ilyesse\${zero-eight} AT gmail for questions.
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

IsometricPatternMatcher is MIT licensed, as found in the LICENSE file.
