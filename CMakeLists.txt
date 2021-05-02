cmake_minimum_required(VERSION 3.10)

project(IsometricPatterMatcher VERSION 1.0
                               DESCRIPTION "Hex-grid, intensity-modulated calibration pattern."
                               LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3.0 REQUIRED)
find_package(Ceres 2.0.0 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(fmt REQUIRED)

add_library(IsometricPatterMatcher IsometricPatternMatcher/HexGridFitting.cpp
                                   IsometricPatternMatcher/IsometricPattern.cpp
                                   IsometricPatternMatcher/LocalParamSe3.cpp
                                   IsometricPatternMatcher/PatternMatcherIsometric.cpp
                                   ${ISO_COMMON_HEADER_FILES}
                                   ${ISO_DOT_EXTRACTOR})
target_link_libraries(IsometricPatterMatcher Ceres::ceres)

add_executable(IsometricPatternGeneration examples/main_IsometricPatternGeneration.cpp)
target_link_libraries(IsometricPatternGeneration IsometricPatterMatcher ${Pangolin_LIBRARIES} fmt::fmt)

add_executable(IsometricPatternDetectionSimple examples/main_IsometricPatternDetectionSimple.cpp)
target_link_libraries(IsometricPatternDetectionSimple IsometricPatterMatcher ${Pangolin_LIBRARIES} fmt::fmt)

include_directories(. ${EIGEN3_INCLUDE_DIR})


# Tests to run
set(TEST_SOURCES test_IsometricDetection test_IsometricPattern)

find_package(GTest REQUIRED)
enable_testing()

foreach(test_src ${TEST_SOURCES})
  add_executable(${test_src} test/${test_src}.cpp)
  target_link_libraries( ${test_src} IsometricPatterMatcher fmt::fmt ${Pangolin_LIBRARIES} GTest::GTest GTest::Main pthread)
  add_test(${test_src} ${test_src} )
endforeach(test_src)
