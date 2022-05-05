#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -x # echo on
set -e # exit on error
brew update
brew install ceres-solver glew googletest
git clone https://github.com/fmtlib/fmt.git
cd fmt
git checkout 6.2.1
mkdir build
cd build
cmake .. -DFMT_TEST=OFF
make -j2
sudo make install
cd ../..
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ../..
ls -l
git clone https://github.com/strasdat/Sophus.git
cd Sophus
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ../..
git clone https://github.com/CLIUtils/CLI11.git
cd CLI11
git checkout v1.9.0
mkdir build
cd build
cmake .. -DCLI11_BUILD_TESTS=OFF
make -j2
sudo make install
cd ../..
