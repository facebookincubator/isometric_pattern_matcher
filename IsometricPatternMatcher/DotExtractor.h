// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <IsometricPatternMatcher/Image.h>
#include <glog/logging.h>
#include <Eigen/Core>
#include <atomic>
#include <memory>
#include <vector>

namespace surreal_opensource {
typedef Eigen::Matrix<float, 2, 1> DotTypeFloat;
typedef Eigen::Matrix<int, 2, 1> DotTypeInt;

const size_t MAX_RAD_BLUR_SMEM = 8;

enum RejectedDotStatus { DETERMINANT, HESSIAN, MAXDELTA, MAXNUM };

template <typename INDEX_TYPE>
class DotExtractorWithIndexType {
 public:
  typedef std::shared_ptr<DotExtractorWithIndexType<INDEX_TYPE>> Ptr;

  DotExtractorWithIndexType(
      INDEX_TYPE numDots = std::numeric_limits<INDEX_TYPE>::max())
      : rejectedDots_(numDots),
        rejectedStatus_(numDots),
        numRejectedDots_(0),
        maxNumDots_(numDots),
        name_("DotExtractor"),
        blurSigma_(1.161),
        maxDelta_(0.807f),
        clampDelta_(0.f),
        numDots_(0),
        hessThresh_(0.01),
        width_(0),
        height_(0),
        dots_(maxNumDots_),
        kernel_(MAX_RAD_BLUR_SMEM, 1) {
    blurKernel(blurKernelRadius_, blurSigma_);
  }

  ~DotExtractorWithIndexType() {}

  DotExtractorWithIndexType<INDEX_TYPE>& operator=(
      const DotExtractorWithIndexType<INDEX_TYPE>& c) {
    dots_.CopyFrom(c.dots_);
    blurredImage_.CopyFrom(c.blurredImage_);
    rejectedDots_ = c.rejectedDots_;
    rejectedStatus_.CopyFrom(c.rejectedStatus_);
    numRejectedDots_ = c.numRejectedDots_;
    numDots_ = c.numDots_;
    maxNumDots_ = c.maxNumDots_;
    maxDelta_ = c.maxDelta_;
    return *this;
  }

  template <typename Tout, typename Tin>
  void blur(surreal_opensource::ManagedImage<Tout>& out,
            const surreal_opensource::ManagedImage<Tin>& in) {
    CHECK(out.w == in.w && out.h == in.h);

    surreal_opensource::ManagedImage<Tout> med(in.w, in.h);
    for (int y = 0; y < in.h; y++) {
      for (int x = 0; x < in.w; ++x) {
        float blurSum = kernel_[0] * in(x, y);
        for (int r = 1; r < this->blurKernelRadius_; ++r) {
          float w = kernel_[r];
          if (x - r >= 0) {
            blurSum += w * in(x - r, y);
          }
          if (x + r < in.w) {
            blurSum += w * in(x + r, y);
          }
        }
        med(x, y) = blurSum;
      }
    }
    for (int x = 0; x < in.w; x++) {
      for (int y = 0; y < in.h; ++y) {
        float blurSum = kernel_[0] * med(x, y);
        for (int r = 1; r < blurKernelRadius_; ++r) {
          const float w = kernel_[r];
          if (y - r >= 0) {
            blurSum += w * med(x, y - r);
          }
          if (y + r < in.h) {
            blurSum += w * med(x, y + r);
          }
        }
        out(x, y) = blurSum;
      }
    }
  }

  void blurKernel(size_t& radius, const float sigma) {
    radius = std::min((size_t)ceil(3 * sigma), MAX_RAD_BLUR_SMEM - 1);

    float sum = 0.0f;

    const float a = 1.0f / (sigma * std::sqrt(2.0f * M_PI));

    for (size_t i = 0; i < MAX_RAD_BLUR_SMEM; i++) {
      const float b = (float)i * (float)i;

      const float c = -b / (2.0f * (sigma * sigma));

      kernel_[i] = a * exp(c);

      // the kernel is symmetric, so we only store one side of it
      // as a result, coeffs count double...
      sum += (i == 0 ? 1.0 : 2.0) * kernel_[i];
    }

    for (size_t i = 0; i < MAX_RAD_BLUR_SMEM; i++) {
      kernel_[i] /= sum;
    }
  }

  void loadIrImage(const surreal_opensource::Image<uint8_t>& input_image) {
    const Eigen::Vector2i imgDim = input_image.Dim();

    if (width_ != (size_t)imgDim(0) || height_ != (size_t)imgDim(1)) {
      width_ = imgDim(0);
      height_ = imgDim(1);
      blurredImage_.Reinitialise(width_, height_);
      rawImage8_.Reinitialise(width_, height_);
    }
    // apply blur
    rawImage8_.CopyFrom(input_image);
    blur(blurredImage_, rawImage8_);
  }

  void detectDots(const Eigen::AlignedBox2i roi) {
    std::atomic<INDEX_TYPE> atomicIndex(0);
    std::atomic<INDEX_TYPE> rejectedAtomicIndex(0);
    for (int y = roi.min().y(); y < roi.max().y(); ++y) {
      for (int x = roi.min().x(); x < roi.max().x(); ++x) {
        const auto* rm = blurredImage_.RowPtr(y - 1) + x;
        const auto* r0 = blurredImage_.RowPtr(y) + x;
        const auto* rp = blurredImage_.RowPtr(y + 1) + x;

        const float ixy = *r0;

        /* Ignore non-maximal pixels.

          Calculate and(ixy >= X) where X is all 8 other pixels surrounding the
         center pixel.
        */

        bool isMaximalPixel = ixy > rm[-1] && ixy > rm[0] && ixy > rm[1] &&
                              ixy > r0[-1] && ixy > r0[1] && ixy > rp[-1] &&
                              ixy > rp[0] && ixy > rp[1];

        if (isMaximalPixel) {
          const float dyy = (*rp + *rm - 2.0f * ixy);
          const float dxx = (r0[1] + r0[-1] - 2.0f * ixy);
          const float dxy = (rm[-1] - rm[1] - rp[-1] + rp[1]) / 4.0f;

          const float det = (dxx * dyy - dxy * dxy);

          if (det != 0.0f) {
            const float dx = (r0[1] - r0[-1]) / 2.0f;
            const float dy = (*rp - *rm) / 2.0f;
            const float invdet = 1.0f / det;

            // x,y update (newton step)
            Eigen::Vector2f fp((dxy * dy - dyy * dx) * invdet,
                               (dxy * dx - dxx * dy) * invdet);

            if (fp.lpNorm<Eigen::Infinity>() < maxDelta_) {
              if (dxx * dxx + dyy * dyy > hessThresh_) {
                INDEX_TYPE memoryIndex = atomicIndex++;
                if (memoryIndex < maxNumDots_) {
                  if (clampDelta_ > 0.f) {
                    fp[0] = std::clamp(fp[0], -clampDelta_, clampDelta_);
                    fp[1] = std::clamp(fp[1], -clampDelta_, clampDelta_);
                  }

                  dots_[memoryIndex] =
                      Eigen::Matrix<float, 2, 1>(x + fp[0], y + fp[1]);

                } else {  // maxdots
                  INDEX_TYPE idx = rejectedAtomicIndex++;
                  rejectedDots_[idx] = Eigen::Vector2f(x + fp[0], y + fp[1]);
                  rejectedStatus_[idx] = RejectedDotStatus::MAXNUM;
                }
              } else {
                INDEX_TYPE idx = rejectedAtomicIndex++;
                rejectedDots_[idx] = Eigen::Vector2f(x + fp[0], y + fp[1]);
                rejectedStatus_[idx] = RejectedDotStatus::HESSIAN;
              }
            } else {
              INDEX_TYPE idx = rejectedAtomicIndex++;
              rejectedDots_[idx] = Eigen::Vector2f(x + fp[0], y + fp[1]);
              rejectedStatus_[idx] = RejectedDotStatus::MAXDELTA;
            }
          } else {
            INDEX_TYPE idx = rejectedAtomicIndex++;
            rejectedDots_[idx] = Eigen::Vector2f(x, y);
            rejectedStatus_[idx] = RejectedDotStatus::DETERMINANT;
          }
        }
      }
    }

    INDEX_TYPE ai = atomicIndex == 0
                        ? 0
                        : atomicIndex - 1;  // to compensate for the last ++
    numDots_ = std::clamp(ai, static_cast<INDEX_TYPE>(0), maxNumDots_);

    INDEX_TYPE rai =
        rejectedAtomicIndex == 0
            ? 0
            : rejectedAtomicIndex - 1;  // to compensate for the last ++
    numRejectedDots_ = std::clamp(rai, static_cast<INDEX_TYPE>(0), maxNumDots_);
  }

  void extractDots(const Eigen::AlignedBox2i roi) { detectDots(roi); }

  void extractDots() {
    Eigen::AlignedBox2i roi(Eigen::Vector2i(0, 0),
                            Eigen::Vector2i(this->width_, this->height_));
    detectDots(roi);
  }

  void copyDetectedDots(surreal_opensource::ManagedImage<DotTypeFloat>& dots,
                        INDEX_TYPE& num_dots) {
    dots.CopyFrom(dots_);
    num_dots = numDots_;
  }

  surreal_opensource::ManagedImage<DotTypeFloat>& getDetectedDotsGPU() {
    return dots_;
  }

  INDEX_TYPE getNumDots() const { return numDots_; }

  DotExtractorWithIndexType<INDEX_TYPE>& setBlurSigma(float sigma) {
    if (sigma != blurSigma_) {
      blurKernel(blurKernelRadius_, sigma);
    }

    blurSigma_ = sigma;
    return *this;
  }

  float getBlurSigma() const { return blurSigma_; }

  DotExtractorWithIndexType<INDEX_TYPE>& setBlurKernelRadius(size_t radius) {
    if (radius != blurKernelRadius_) {
      blurKernel(radius, blurSigma_);
    }

    blurKernelRadius_ = radius;
    return *this;
  }

  size_t getBlurKernelRadius(size_t radius) { return blurKernelRadius_; }

  DotExtractorWithIndexType<INDEX_TYPE>& setHessThresh(float thresh) {
    hessThresh_ = thresh;
    return *this;
  }

  float getHessThresh() const { return hessThresh_; }

  DotExtractorWithIndexType<INDEX_TYPE>& setMaxDelta(float d) {
    maxDelta_ = d;
    return *this;
  }

  float getMaxDelta() const { return maxDelta_; }

  DotExtractorWithIndexType<INDEX_TYPE>& setClampDelta(float d) {
    clampDelta_ = d;
    return *this;
  }

  float getClampDelta() const { return clampDelta_; }

  surreal_opensource::ManagedImage<float> blurredImage_;

  std::vector<Eigen::Vector2f> rejectedDots_;
  surreal_opensource::ManagedImage<RejectedDotStatus> rejectedStatus_;
  INDEX_TYPE numRejectedDots_;

 private:
  INDEX_TYPE maxNumDots_;
  std::string name_;
  size_t blurKernelRadius_;
  float blurSigma_;
  // subpixel refinements larger than maxDelta_ are rejected
  float maxDelta_;
  // subpixel refinements larger than clampDelta_ are clamped to clampDelta_
  float clampDelta_;

  INDEX_TYPE numDots_;
  float hessThresh_;
  INDEX_TYPE width_, height_;

  surreal_opensource::ManagedImage<unsigned char> rawImage8_;
  surreal_opensource::ManagedImage<DotTypeFloat> dots_;
  surreal_opensource::ManagedImage<int> globalNextFree_;
  surreal_opensource::ManagedImage<float> kernel_;
};

using DotExtractor = DotExtractorWithIndexType<uint16_t>;
using DotExtractor32 = DotExtractorWithIndexType<uint32_t>;
using DotExtractor64 = DotExtractorWithIndexType<uint64_t>;
}  // namespace surreal_opensource
