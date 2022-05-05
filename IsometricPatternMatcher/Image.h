/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>

#include <glog/logging.h>
#include <pangolin/image/image_io.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace surreal_opensource {
template <class T>
using DefaultImageAllocator = std::allocator<T>;

template <class T, class Enable = void>
struct DefaultImageValTraits {
  static constexpr int max_value = 0;
};
template <class T, class Enable>
constexpr int DefaultImageValTraits<T, Enable>::max_value;

template <class T>
struct DefaultImageValTraits<
    T, typename std::enable_if<std::is_integral<T>::value>::type> {
  static constexpr int max_value =
      (sizeof(T) >= sizeof(int))
          ? std::numeric_limits<int>::max()
          : static_cast<int>(std::numeric_limits<T>::max());
};

template <class T>
struct DefaultImageValTraits<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
  static constexpr int max_value = 1;
};

template <class T, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct DefaultImageValTraits<
    Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>> {
  static constexpr int max_value = DefaultImageValTraits<T>::max_value;
};

struct ImageDimensions {
  size_t width;
  size_t height;
};

namespace details {
// Scalar case
template <class T>
struct Zero {
  static T val() { return T(0); }
};

// Specialization for Eigen types
template <class T, int M, int N, int Opts>
struct Zero<Eigen::Matrix<T, M, N, Opts>> {
  static Eigen::Matrix<T, M, N, Opts> val() {
    Eigen::Matrix<T, M, N, Opts> o;
    o.setZero();
    return o;
  }
};

// underlying memcopy function
inline void PitchedCopy(char* dst, size_t dst_pitch_bytes, const char* src,
                        size_t src_pitch_bytes, size_t width_bytes,
                        size_t height) {
  if (dst_pitch_bytes == width_bytes && src_pitch_bytes == width_bytes) {
    std::memcpy(dst, src, height * width_bytes);
  } else {
    for (size_t row = 0; row < height; ++row) {
      std::memcpy(dst, src, width_bytes);
      dst += dst_pitch_bytes;
      src += src_pitch_bytes;
    }
  }
}

}  // namespace details

// Image class that stores a weak ptr to it's memory.
//
// MaxValue is a compile-time hint, which indicates the maximum value, if this
// structure represents an intensity image. This class is CUDA aware: it does
// not require CUDA, but supports being compiled in CUDA. If HAVE_CUDA is
// defined, then some operations will use CUDA functions.
template <typename T, int MaxValue = DefaultImageValTraits<T>::max_value>
struct Image {
  static constexpr int max_value = MaxValue;
  using BaseType = T;
  inline Image() noexcept : pitch(0), ptr(nullptr), w(0), h(0) {}
  inline Image(T* ptr) noexcept : pitch(0), ptr(ptr), w(0), h(0) {}
  inline Image(T* ptr, size_t w) noexcept
      : pitch(sizeof(T) * w), ptr(ptr), w(w), h(1) {}
  inline Image(T* ptr, size_t w, size_t h) noexcept
      : pitch(sizeof(T) * w), ptr(ptr), w(w), h(h) {}
  inline Image(T* ptr, size_t w, size_t h, size_t pitch) noexcept
      : pitch(pitch), ptr(ptr), w(w), h(h) {}
  inline Image(const Image<std::remove_cv_t<T>, MaxValue>& other) noexcept
      : pitch(other.pitch), ptr(other.ptr), w(other.w), h(other.h) {}

  //////////////////////////////////////////////////////
  // Query dimensions
  //////////////////////////////////////////////////////

  inline size_t Width() const { return w; }
  inline size_t Height() const { return h; }
  inline size_t Area() const { return w * h; }
  inline size_t SizeBytes() const { return pitch * h; }
  inline Eigen::Vector2i Dim() const { return Eigen::Vector2i(w, h); }
  //////////////////////////////////////////////////////
  // Iterators
  //////////////////////////////////////////////////////

  inline T* begin() { return ptr; }
  inline T* end() { return (T*)((unsigned char*)(ptr) + h * pitch); }
  inline const T* begin() const { return ptr; }
  inline const T* end() const {
    return (T*)((unsigned char*)(ptr) + h * pitch);
  }
  inline size_t size() const { return w * h; }
  //////////////////////////////////////////////////////
  // Image set / copy
  //////////////////////////////////////////////////////

  inline void Fill(const T& val) {
    CHECK(IsValid() && IsContiguous());
    const T* end_element = end();
    for (T* it = begin(); it != end_element; ++it) {
      *it = val;
    }
  }

  template <typename TOther, int OtherMaxValue,
            typename std::enable_if<
                std::is_same<T, typename std::remove_cv<TOther>::type>::value,
                int>::type = 0>
  inline void CopyFrom(const Image<TOther, OtherMaxValue>& img) {
    if (IsValid() && img.IsValid()) {
      CHECK(w >= img.w && h >= img.h);
      details::PitchedCopy((char*)ptr, pitch, (const char*)img.ptr, img.pitch,
                           std::min(img.w, w) * sizeof(T), std::min(img.h, h));
    } else if (img.IsValid() != IsValid()) {
      CHECK(false && "Cannot copy from / to an unasigned image.");
    }
  }

  // Converting one of the Eigen image types is not implemented
  template <
      typename R,
      typename std::enable_if<std::is_integral<BaseType>::value,
                              typename R::Base::BaseType>::type = 0,
      typename std::enable_if<
          std::is_integral<typename R::Base::BaseType>::value, int>::type = 0>
  inline R ConvertTo() const {
    // Ensure for now that they are a powers of 2 (true for all current
    // use cases). This simplifies the arithmetic.
    static_assert(((R::Base::max_value + 1) & R::Base::max_value) == 0,
                  "Must be power of 2");
    static_assert(((max_value + 1) & max_value) == 0, "Must be power of 2");
    // Only support converting from higher bpp to lower bpp.
    // since we're doing integer division.
    static_assert(R::Base::max_value < max_value, "Can only reduce");

    // Sanity check
    static_assert((max_value + 1) % (R::Base::max_value + 1) == 0,
                  "Should be divisble");
    // It's not correct to divide by 4 for say going from 10bpp to 8bpp.
    // To be exact, we'd need to multiply by 255 / 1023
    // However, this should be sufficiently accurate.
    static constexpr int CONVERSION_FACTOR =
        (max_value + 1) / (R::Base::max_value + 1);

    R dest_image{this->w, this->h};
    for (size_t y = 0; y < this->h; ++y) {
      const BaseType* src_pixel_ptr = this->RowPtr(y);
      typename R::Base::BaseType* dest_pixel_ptr = dest_image.RowPtr(y);
      for (size_t x = 0; x < this->w; ++x) {
        *dest_pixel_ptr = *src_pixel_ptr / CONVERSION_FACTOR;
        ++src_pixel_ptr;
        ++dest_pixel_ptr;
      }
    }
    return dest_image;
  }

  //////////////////////////////////////////////////////
  // Direct Pixel Access
  //////////////////////////////////////////////////////

  inline bool IsValid() const { return ptr != 0; }
  inline bool IsContiguous() const { return w * sizeof(T) == pitch; }

  inline T* RowPtr(size_t y) {
    CHECK(YInBounds(y));
    return (T*)((unsigned char*)(ptr) + y * pitch);
  }
  inline const T* RowPtr(size_t y) const {
    CHECK(YInBounds(y));
    return (T*)((unsigned char*)(ptr) + y * pitch);
  }

  inline T& operator()(size_t x, size_t y) {
    CHECK(InBounds(x, y));
    return RowPtr(y)[x];
  }

  inline const T& operator()(size_t x, size_t y) const {
    CHECK(InBounds(x, y));
    return RowPtr(y)[x];
  }

  inline T& operator()(const Eigen::Vector2i& p) {
    CHECK(InBounds(p[0], p[1]));
    return RowPtr(p[1])[p[0]];
  }

  inline const T& operator()(const Eigen::Vector2i& p) const {
    CHECK(InBounds(p[0], p[1]));
    return RowPtr(p[1])[p[0]];
  }

  inline T& operator[](size_t ix) {
    CHECK(InImage(ptr + ix));
    return ptr[ix];
  }

  inline const T& operator[](size_t ix) const {
    CHECK(InImage(ptr + ix));
    return ptr[ix];
  }

  inline T& Get(int x, int y) {
    CHECK(InBounds(x, y));
    return RowPtr(y)[x];
  }

  inline const T& Get(int x, int y) const {
    CHECK(InBounds(x, y));
    return RowPtr(y)[x];
  }

  //////////////////////////////////////////////////////
  // Bounds Checking
  //////////////////////////////////////////////////////

  bool InImage(const T* ptest) const { return ptr <= ptest && ptest < end(); }

  inline bool YInBounds(int y) const { return 0 <= y && y < (int)h; }

  inline bool InBounds(int x, int y) const {
    return 0 <= x && x < (int)w && 0 <= y && y < (int)h;
  }

  inline bool InBounds(float x, float y, float border) const {
    return border <= x && x < (w - border) && border <= y && y < (h - border);
  }

  template <typename Derived>
  inline bool InBounds(
      const Eigen::MatrixBase<Derived>& p,
      const typename Eigen::MatrixBase<Derived>::Scalar border) const {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 2,
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return border <= p[0] && p[0] < ((int)w - border) && border <= p[1] &&
           p[1] < ((int)h - border);
  }

  template <typename Derived>
  inline bool InBounds(const Eigen::MatrixBase<Derived>& p) const {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 2,
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return 0 <= p[0] && p[0] < (int)w && 0 <= p[1] && p[1] < (int)h;
  }

  //////////////////////////////////////////////////////
  // Obtain slices / subimages
  //////////////////////////////////////////////////////

  inline const Image<const T, MaxValue> SubImage(size_t x, size_t y,
                                                 size_t width,
                                                 size_t height) const {
    CHECK((x + width) <= w && (y + height) <= h);
    return Image<const T, MaxValue>(RowPtr(y) + x, width, height, pitch);
  }

  inline Image<T, MaxValue> SubImage(size_t x, size_t y, size_t width,
                                     size_t height) {
    CHECK((x + width) <= w && (y + height) <= h);
    return Image<T, MaxValue>(RowPtr(y) + x, width, height, pitch);
  }

  inline const Image<const T, MaxValue> SubImage(
      const Eigen::AlignedBox2i& b) const {
    return SubImage(b.min()(0), b.min()(1), b.sizes()(0), b.sizes()(1));
  }

  inline Image<T> SubImage(const Eigen::AlignedBox2i& b) {
    return SubImage(b.min()(0), b.min()(1), b.sizes()(0), b.sizes()(1));
  }

  inline Image<const T, MaxValue> Row(int y) const {
    return SubImage(0, y, w, 1);
  }
  inline Image<const T, MaxValue> Col(int x) const {
    return SubImage(x, 0, 1, h);
  }
  inline Image<T, MaxValue> Row(int y) { return SubImage(0, y, w, 1); }
  inline Image<T, MaxValue> Col(int x) { return SubImage(x, 0, 1, h); }
  inline Image<T, MaxValue> SubImage(int width, int height) {
    CHECK(width <= (int)w && height <= (int)h);
    return Image<T, MaxValue>(ptr, width, height, pitch);
  }

  inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
  GetEigenMap() {
    // to enable operations on images using eigen: e.g. im_map = im_map +
    // im_map*2;
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> im_map(ptr, h,
                                                                        w);
    return im_map;
  }

  inline Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
  GetEigenMap() const {
    // to enable operations on images using eigen: e.g. im_map = im_map +
    // im_map*2;
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> im_map(
        ptr, h, w);
    return im_map;
  }

  inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0,
                    Eigen::Stride<Eigen::Dynamic, 1>>
  GetEigenMapPitched() {
    // to enable operations on images using eigen: e.g. im_map = im_map +
    // im_map*2; use Eigen::Stride to work on subimages as well.
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0,
               Eigen::Stride<Eigen::Dynamic, 1>>
        im_map(ptr, h, w,
               Eigen::Stride<Eigen::Dynamic, 1>(pitch / sizeof(T), 1));
    return im_map;
  }

  inline Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0,
                    Eigen::Stride<Eigen::Dynamic, 1>>
  GetEigenMapPitched() const {
    // to enable operations on images using eigen: e.g. im_map = im_map +
    // im_map*2; use Eigen::Stride to work on subimages as well.
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0,
               Eigen::Stride<Eigen::Dynamic, 1>>
        im_map(ptr, h, w,
               Eigen::Stride<Eigen::Dynamic, 1>(pitch / sizeof(T), 1));
    return im_map;
  }

  template <typename Tout = T>
  Tout Sum() const {
    Tout sum = details::Zero<Tout>::val();
    for (size_t r = 0; r < h; ++r) {
      const T* row_ptr = RowPtr(r);
      const T* end = row_ptr + w;
      while (row_ptr != end) {
        sum += Tout(*row_ptr);
        ++row_ptr;
      }
    }
    return sum;
  }

  template <typename Tout = T>
  Tout Mean() const {
    return Sum<Tout>() / Area();
  }

  size_t pitch;
  T* ptr;
  size_t w;
  size_t h;
};

#if defined __AVX2__ && !defined __NVCC__
// TODO: break this into vector utility
inline float hsum(__m256 v) {
  __m128 v2 = _mm256_extractf128_ps(v, 1);
  __m128 v1 = _mm256_castps256_ps128(v);
  v1 = _mm_add_ps(v1, v2);
  v2 = _mm_movehdup_ps(v1);  // broadcast elements 3,1 to 2,0
  v1 = _mm_add_ps(v1, v2);
  v2 = _mm_movehl_ps(v2, v1);  // high half -> low half
  v1 = _mm_add_ss(v1, v2);
  return _mm_cvtss_f32(v1);
}

template <>
template <>
inline float Image<float>::Sum<float>() const {
  const size_t w32loops = w / 32;
  const size_t w8loops = (w - w32loops * 32) / 8;
  const size_t w1loops = w & 7;

  __m256 sum = _mm256_setzero_ps();
  alignas(32) float arr[8];
  const float* rowStart = RowPtr(0);
  const size_t rowElts = pitch / sizeof(float);

  _mm256_store_ps(arr, sum);

  for (size_t r = 0; r < h; ++r, rowStart += rowElts) {
    const float* rowOffset = rowStart;
    for (size_t i = 0; i < w32loops; ++i, rowOffset += 32) {
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(rowOffset));
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(rowOffset + 8));
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(rowOffset + 16));
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(rowOffset + 24));
    }
    for (size_t i = 0; i < w8loops; ++i, rowOffset += 8) {
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(rowOffset));
    }
    std::copy_n(rowOffset, w1loops, arr);
    sum = _mm256_add_ps(sum, _mm256_load_ps(arr));
  }

  return hsum(sum);
}
#endif  //__AVX2__

template <typename T, int MaxValue>
constexpr int Image<T, MaxValue>::max_value;

static_assert(Image<uint8_t>::max_value == 255, "Compile time sanity check");
static_assert(Image<float>::max_value == 1, "Compile time sanity check");
static_assert(Image<uint16_t, 1023>::max_value == 1023,
              "Compile time sanity check");
static_assert(Image<float, 0>::max_value == 0, "Compile time sanity check");
static_assert(Image<Eigen::Matrix<uint8_t, 3, 3>>::max_value == 255,
              "Compile time sanity check");
static_assert(Image<Eigen::Matrix<float, 1, 2>>::max_value == 1,
              "Compile time sanity check");
static_assert(Image<Eigen::Matrix<uint16_t, 1, 3>, 1023>::max_value == 1023,
              "Compile time sanity check");
static_assert(Image<Eigen::Vector3d, -1>::max_value == -1,
              "Compile time sanity check");

// Image that manages it's own memory, storing a strong pointer to it's memory
template <typename T, class Allocator_ = DefaultImageAllocator<T>,
          int MaxValue = DefaultImageValTraits<T>::max_value>
class ManagedImage : public Image<T, MaxValue> {
 public:
  using Base = Image<T, MaxValue>;
  using Allocator = Allocator_;

  // Destructor
  inline ~ManagedImage() { Deallocate(); }
  // Null image
  inline ManagedImage() {}
  // Row image
  //
  // Precondition: w must not be 0.
  inline ManagedImage(size_t w)
      : Base(Allocator().allocate(w), w, 1, w * sizeof(T)) {
    CHECK(w != 0);
  }

  // Precondition: Neither w nor h should be 0.
  inline ManagedImage(size_t w, size_t h)
      : Base(Allocator().allocate(w * h), w, h, w * sizeof(T)) {
    CHECK(w != 0 && h != 0);
  }

  // Precondition: Neither dim.width nor dim.height should be 0.
  inline ManagedImage(ImageDimensions dim)
      : ManagedImage(dim.width, dim.height) {}

  // Precondition: Neither dim.x() nor dim.y() should be 0.
  inline ManagedImage(Eigen::Vector2i dim) : ManagedImage(dim.x(), dim.y()) {}

  // Not copy constructable
  inline ManagedImage(const ManagedImage& other) = delete;

  // Move constructor
  inline ManagedImage(ManagedImage&& img) noexcept
      : Base(img.ptr, img.w, img.h, img.pitch) {
    img.ptr = nullptr;
  }

  // Move asignment
  inline void operator=(ManagedImage&& img) {
    Deallocate();
    Base::pitch = img.pitch;
    Base::ptr = img.ptr;
    Base::w = img.w;
    Base::h = img.h;
    img.ptr = nullptr;
  }

  inline void Swap(ManagedImage& img) {
    std::swap(img.pitch, Image<T>::pitch);
    std::swap(img.ptr, Image<T>::ptr);
    std::swap(img.w, Image<T>::w);
    std::swap(img.h, Image<T>::h);
  }

  template <typename TOther, int OtherMaxValue>
  inline void CopyFrom(const Image<TOther, OtherMaxValue>& img) {
    if (!Base::IsValid() || Base::w != img.w || Base::h != img.h) {
      Reinitialise(img.w, img.h);
    }
    Base::CopyFrom(img);
  }

  inline void Reinitialise(size_t width, size_t height) {
    if (width == 0 || height == 0) {
      *this = ManagedImage();
    } else if (!Base::ptr || Base::Width() != width ||
               Base::Height() != height) {
      *this = ManagedImage(width, height);
    }
  }

  inline void Reinitialise(const Eigen::Vector2i& dim) {
    Reinitialise(dim(0), dim(1));
  }

 protected:
  inline void Deallocate() {
    if (Base::ptr) {
      Allocator().deallocate(Base::ptr, Base::Area());
      Base::ptr = nullptr;
    }
  }
};

template <typename T, template <typename...> class Alloc>
using ManagedImageAlloc = ManagedImage<T, Alloc<T>>;

// Image Processing / IO functions
template <typename Tout, typename Tin>
ManagedImage<Tout> Own(pangolin::Image<Tin>& im) {
  return std::move(reinterpret_cast<ManagedImage<Tout>&>(im));
}

template <typename T>
ManagedImage<T> LoadImage(const std::string& filename) {
  pangolin::TypedImage img = pangolin::LoadImage(filename);
  CHECK(img.fmt.bpp == sizeof(T) * 8);
  return Own<T, uint8_t>(img);
}
}  // namespace surreal_opensource
