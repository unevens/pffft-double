#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include "../pffft.hpp"
#include "fft4g.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace pffft;
using namespace std;

#ifdef __AVX__
constexpr bool hasAvx = true;
#else
constexpr bool hasAvx = false;
#endif

void
test(int fftLength, int profilingIterations, int numOfIterationsToSkip)
{
  assert(profilingIterations > 0);
  assert(numOfIterationsToSkip > 0);
  assert(profilingIterations > numOfIterationsToSkip);

  cout << "testing with fft length = " << fftLength << "\n";

  double* dInput = (double*)pffftd_aligned_malloc(fftLength * sizeof(double));
  double* dSpectrum = (double*)pffftd_aligned_malloc(fftLength * sizeof(double));
  double* dOutput = (double*)pffftd_aligned_malloc(fftLength * sizeof(double));
  double* oInOut = (double*)pffftd_aligned_malloc(fftLength * sizeof(double));

  float* fInput = (float*)pffft_aligned_malloc(fftLength * sizeof(float));
  float* fSpectrum = (float*)pffft_aligned_malloc(fftLength * sizeof(float));
  float* fOutput = (float*)pffft_aligned_malloc(fftLength * sizeof(float));

  for (int i = 0; i < fftLength; ++i) {
    dInput[i] =
      sin(2.0 * M_PI * 0.45 * (double)i * (1.0 - ((double)i) / (2.0 * (double)fftLength)));
    fInput[i] = (float)dInput[i];
    oInOut[i] = dInput[i];
  }

int oWiSize = (int)ceil(2.0 + sqrt((double)(fftLength >> 1)));
int oWdSize = fftLength >> 1;

int* oWi = (int*)pffftd_aligned_malloc(oWiSize * sizeof(int));
double* oWd = (double*)pffftd_aligned_malloc(oWdSize * sizeof(double));
oWi[0] = 0;

Fft<double> fftDouble(fftLength);
Fft<float> fftFloat(fftLength);

int realProfilingIterations = profilingIterations - numOfIterationsToSkip;
std::vector<long long int> dTimes(realProfilingIterations);
std::vector<long long int> fTimes(realProfilingIterations);
std::vector<long long int> oTimes(realProfilingIterations);

double oCoef = 2.0 / (double)fftLength;

for (int i = 0; i < profilingIterations; ++i) {

  auto oStart = chrono::high_resolution_clock::now();
  r8b::ooura_fft::rdft(fftLength, 1, oInOut, oWi, oWd);
  r8b::ooura_fft::rdft(fftLength, -1, oInOut, oWi, oWd);
  for (int i = 0; i < fftLength; ++i) {
    oInOut[i] *= oCoef;
  }
  auto oEnd = chrono::high_resolution_clock::now();
  if (i > numOfIterationsToSkip) {
    oTimes[i - numOfIterationsToSkip] =
      chrono::duration_cast<chrono::nanoseconds>(oEnd - oStart).count();
  }
}

for (int i = 0; i < profilingIterations; ++i) {

  auto dStart = chrono::high_resolution_clock::now();

  fftDouble.Forward(dInput, dSpectrum);
  fftDouble.Inverse(dSpectrum, dOutput);

  auto dEnd = chrono::high_resolution_clock::now();
  if (i > numOfIterationsToSkip) {
    dTimes[i - numOfIterationsToSkip] =
      chrono::duration_cast<chrono::nanoseconds>(dEnd - dStart).count();
  }
}

for (int i = 0; i < profilingIterations; ++i) {

  auto fStart = chrono::high_resolution_clock::now();

  fftFloat.Forward(fInput, fSpectrum);
  fftFloat.Inverse(fSpectrum, fOutput);

  auto fEnd = chrono::high_resolution_clock::now();
  if (i > numOfIterationsToSkip) {
    fTimes[i - numOfIterationsToSkip] =
      chrono::duration_cast<chrono::nanoseconds>(fEnd - fStart).count();
  }
}

double dTimeAverage = 0.0;
double dTimeDeviation = 0.0;
double fTimeAverage = 0.0;
double fTimeDeviation = 0.0;
double oTimeAverage = 0.0;
double oTimeDeviation = 0.0;

for (int i = 0; i < realProfilingIterations; ++i) {
  dTimeAverage += ((double)dTimes[i]) / (double)realProfilingIterations;
  fTimeAverage += ((double)fTimes[i]) / (double)realProfilingIterations;
  oTimeAverage += ((double)oTimes[i]) / (double)realProfilingIterations;
}
for (int i = 0; i < realProfilingIterations; ++i) {
  double dDiff = dTimeAverage - (double)dTimes[i];
  dTimeDeviation += dDiff * dDiff / (double)realProfilingIterations;
  double fDiff = fTimeAverage - (double)fTimes[i];
  fTimeDeviation += fDiff * fDiff / (double)realProfilingIterations;
  double oDiff = oTimeAverage - (double)oTimes[i];
  oTimeDeviation += oDiff * oDiff / (double)realProfilingIterations;
}
dTimeDeviation = 100.0 * sqrt(dTimeDeviation) / dTimeAverage;
fTimeDeviation = 100.0 * sqrt(fTimeDeviation) / fTimeAverage;
oTimeDeviation = 100.0 * sqrt(oTimeDeviation) / oTimeAverage;

double dNoisePower = 0.0;
double dInputPower = 0.0;
double fNoisePower = 0.0;
double fInputPower = 0.0;
double oNoisePower = 0.0;

for (int i = 0; i < fftLength; ++i) {
  dInputPower += dInput[i] * dInput[i];
  fInputPower += fInput[i] * fInput[i];

  double dDiff = dInput[i] - dOutput[i];
  dNoisePower += dDiff * dDiff;

  double fDiff = fInput[i] - fOutput[i];
  fNoisePower += fDiff * fDiff;

  double oDiff = dInput[i] - oInOut[i];
  oNoisePower += oDiff * oDiff;
}

dNoisePower /= (double)fftLength;
fNoisePower /= (double)fftLength;
oNoisePower /= (double)fftLength;

cout << "pfft double roundtrip SNR = " << 10.0 * log10(dInputPower / dNoisePower) << " dB\n";
cout << "pfft float roundtrip SNR = " << 10.0 * log10(fInputPower / fNoisePower) << " dB\n";
cout << "ooura (double) roundtrip SNR = " << 10.0 * log10(dInputPower / oNoisePower) << " dB\n";

#ifdef NDEBUG
cout << "AVX instructions enabled: " << (hasAvx ? "True" : "False") << "\n";
cout << "pffft double took " << dTimeAverage
     << " nanoseconds time to preform a forward and a backward transform. \n";
cout << "Average computed with " << realProfilingIterations << " iterations, after skipping "
     << numOfIterationsToSkip << " iterations. Deviation = " << dTimeDeviation << " %.\n";
cout << "pffft float took " << fTimeAverage
     << " nanoseconds time to preform a forward and a backward transform. \n";
cout << "Average computed with " << realProfilingIterations << " iterations, after skipping "
     << numOfIterationsToSkip << " iterations. Deviation = " << fTimeDeviation << " %.\n";
cout << "ooura took " << oTimeAverage
     << " nanoseconds time to preform a forward and a backward transform. \n";
cout << "Average computed with " << realProfilingIterations << " iterations, after skipping "
     << numOfIterationsToSkip << " iterations. Deviation = " << oTimeDeviation << " %.\n";

#endif

cout << "\n\n";

// cleanup
pffftd_aligned_free(dInput);
pffftd_aligned_free(dSpectrum);
pffftd_aligned_free(dOutput);
pffft_aligned_free(fInput);
pffft_aligned_free(fSpectrum);
pffft_aligned_free(fOutput);
}

int
main()
{
  cout << "AVX instructions enabled: " << (hasAvx ? "True" : "False") << "\n";

#ifdef NDEBUG
  int profilingIterations = 2048;
  int iterationsToSkip = 256;
#else
  int profilingIterations = 1;
  int iterationsToSkip = 0;
#endif

  test(256, profilingIterations, iterationsToSkip);
  test(512, profilingIterations, iterationsToSkip);
  test(1024, profilingIterations, iterationsToSkip);
  test(2048, profilingIterations, iterationsToSkip);
  test(4096, profilingIterations, iterationsToSkip);
  test(8192, profilingIterations, iterationsToSkip);
  test(16384, profilingIterations, iterationsToSkip);
  test(32768, profilingIterations, iterationsToSkip);

  return 0;
}
