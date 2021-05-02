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

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context)

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
