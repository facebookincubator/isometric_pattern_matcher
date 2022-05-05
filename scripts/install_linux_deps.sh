#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -x # echo on
set -e # exit on error

cmake --version

sudo apt-get -qq update
sudo apt-get install gfortran libc++-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev libceres-dev libglew-dev
git clone https://github.com/google/googletest
cd googletest
mkdir build
cd build
cmake ..
make
sudo make install
cd ../..
git clone https://github.com/fmtlib/fmt.git
cd fmt
git checkout 6.2.1
mkdir build
cd build
cmake .. -DFMT_TEST=OFF
make -j2
sudo make install
cd ../..
ls -l
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.4/eigen-3.3.4.tar.bz2
tar xvf eigen-3.3.4.tar.bz2
cd eigen-3.3.4
mkdir build-eigen
cd build-eigen
cmake ..
sudo make install
cd ../..
ls -l
git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
cd ceres-solver
git reset --hard 399cda773035d99eaf1f4a129a666b3c4df9d1b1
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
make -j8
sudo make install
cd ../..
ls -l
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
make -j8
sudo make install
cd ../..
ls -l
git clone https://github.com/strasdat/Sophus.git
cd Sophus
mkdir build
cd build
cmake -DBUILD_TESTS=OFF ..
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
