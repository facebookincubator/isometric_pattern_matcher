
#!/bin/bash

set -x # echo on
set -e # exit on error
brew update
brew install ceres-solver glew
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
git clone https://github.com/stevenlovegrove/Sophus.git
cd Sophus
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ../..
