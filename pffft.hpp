/* cpp wrapper around Julien Pommier's PFFFT and my port to double precision
floating point with AVX instructions.
Author: Dario Mambro @ https://github.com/unevens/pffft */

/*
PFFFT and my port are redistributed under the original PFFFT license,
see the LICENSE file.
This wrapper is of public domain.

This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.
In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
For more information, please refer to <http://unlicense.org/>
*/

#pragma once

#include "pffft-double.h"
#include "pffft.h"
#include <memory>

namespace pffft {

enum class TransformType
{
  Real = 0,
  Complex
};

namespace {
template<typename T>
class FftCommon
{
protected:
  T* work;
  int length;
  T invLength;
  TransformType type;

  virtual void Setup() = 0;

  FftCommon(int length, TransformType type)
    : length(length)
    , invLength(1.0 / length)
    , type(type)
    , work(nullptr)
  {}

public:
  void SetLength(int length);
  int GetLength() { return length; }
  void SetType(TransformType type);
  virtual ~FftCommon() {}
};
} // namespace

template<typename T>
class Fft final : public FftCommon<T>
{};

template<>
class Fft<float> final : public FftCommon<float>
{
  struct Deleter
  {
    void operator()(PFFFT_Setup* ptr) { pffft_destroy_setup(ptr); }
  };

  std::unique_ptr<PFFFT_Setup, Deleter> self;

  void Setup() override;

public:
  Fft(int length, TransformType type = TransformType::Real);
  ~Fft() { pffft_aligned_free(work); }
  void Forward(float* input, float* output);
  void Inverse(float* input, float* output);
};

template<>
class Fft<double> final : public FftCommon<double>
{
  struct Deleter
  {
    void operator()(PFFFTD_Setup* ptr) { pffftd_destroy_setup(ptr); }
  };

  std::unique_ptr<PFFFTD_Setup, Deleter> self;
  void Setup() override;

public:
  Fft(int length, TransformType type = TransformType::Real);
  ~Fft() { pffftd_aligned_free(work); }
  void Forward(double* input, double* output);
  void Inverse(double* input, double* output);
};

// implementation

template<typename T>
void
FftCommon<T>::SetLength(int length_)
{
  length = length_;
  invLength = 1.0 / length;
  Setup();
}

template<typename T>
inline void
FftCommon<T>::SetType(TransformType type_)
{
  type = type_;
  Setup();
}

inline Fft<float>::Fft(int length, TransformType type)
  : FftCommon<float>(length, type)
{
  Setup();
}

inline void
Fft<float>::Setup()
{
  self = std::unique_ptr<PFFFT_Setup, Deleter>(
    pffft_new_setup(length, static_cast<pffft_transform_t>(type)));
  pffft_aligned_free(work);
  if (length <= 16384) {
    work = nullptr;
    return;
  }
  int buffer_length = length * sizeof(float);
  if (type == TransformType::Complex) {
    buffer_length *= 2;
  }
  work = (float*)pffft_aligned_malloc(buffer_length);
}

inline void
Fft<float>::Forward(float* input, float* output)
{
  pffft_transform_ordered(self.get(), input, output, work, PFFFT_FORWARD);
}

inline void
Fft<float>::Inverse(float* input, float* output)
{
  pffft_transform_ordered(self.get(), input, output, work, PFFFT_BACKWARD);
  for (int i = 0; i < length; ++i) {
    output[i] *= invLength;
  }
}

inline Fft<double>::Fft(int length, TransformType type)
  : FftCommon<double>(length, type)
{
  Setup();
}

inline void
Fft<double>::Setup()
{
  self = std::unique_ptr<PFFFTD_Setup, Deleter>(
    pffftd_new_setup(length, static_cast<pffftd_transform_t>(type)));
  pffftd_aligned_free(work);
  if (length <= 8192) {
    work = nullptr;
    return;
  }
  int buffer_length = length * sizeof(double);
  if (type == TransformType::Complex) {
    buffer_length *= 2;
  }
  work = (double*)pffftd_aligned_malloc(buffer_length);
}

inline void
Fft<double>::Forward(double* input, double* output)
{
  pffftd_transform_ordered(self.get(), input, output, work, PFFFTD_FORWARD);
}

inline void
Fft<double>::Inverse(double* input, double* output)
{
  pffftd_transform_ordered(self.get(), input, output, work, PFFFTD_BACKWARD);
  for (int i = 0; i < length; ++i) {
    output[i] *= invLength;
  }
}

} // namespace pffft
