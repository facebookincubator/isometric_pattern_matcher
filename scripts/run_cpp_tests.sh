#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -x # echo on
set -e # exit on error

mkdir build
cd build
pwd
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j8
make CTEST_OUTPUT_ON_FAILURE=1 test
