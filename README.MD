# Deterministic MFCC Frontend for ESP32-Class Edge AI
A resource-constrained audio feature-extraction pipeline designed for
ESP32-class microcontrollers.
The project implements a deterministic MFCC frontend in C++ that converts
16 kHz, 16-bit mono PCM audio into the exact INT8 tensor expected by the
deployed machine-learning model.
The embedded implementation is designed to maintain algorithmic parity
with the corresponding Python training/reference pipeline.

## Features
- 16 kHz mono PCM audio processing
- 1-second audio windows
- 1024-point radix-2 FFT
- 640-sample Hann window
- 320-sample hop length
- 128 Slaney-normalized Mel filters
- 80 dB dynamic-range clipping
- Orthonormal DCT-II
- 13 MFCC coefficients
- 51 × 13 MFCC representation
- Float32 DSP processing
- Training-set normalization
- TFLite-compatible affine INT8 quantization
- No dynamic memory allocation
- Precomputed DSP constants
- No large Mel filter-bank matrix stored in RAM
- Python/C++ frontend parity
- Golden-vector validation support
- Designed for ESP32-class resource constraints

## Pipeline
16-bit PCM Audio
	|
	V
16 kHz / 1 second
	|
	V
PCM normalization
	|
	V
640-sample Hann window
	|
	V
1024-point radix-2 FFT
	|
	V
Power spectrum
	|
	V
128 Slaney-normalized Mel filters
	|
	V
Log-power spectrum
	|
	V
80 dB dynamic-range clipping
	|
	V
Orthonormal DCT-II
	|
	V
13 MFCC coefficients
	|
	V
51 × 13 feature matrix
	|
	V
Training-set normalization
	|
	V
TFLite affine INT8 quantization
	|
	V
663-byte INT8 model tensor
	|
	V
Edge ML inference
The implementation specifically produces 663 INT8 values (51 × 13)

## Resource-Conscious Design
The frontend is designed for microcontroller deployment rather than
desktop-oriented audio processing.
Key design decisions include:
- Static memory allocation
- No dynamic allocation during feature extraction
- Precomputed FFT twiddle factors
- Precomputed Hann window
- Precomputed DCT coefficients
- Precomputed Mel-filter geometry
- On-the-fly Mel filter weights
- Reuse of FFT working memory
- No 128 × 513 Mel filter-bank matrix stored in RAM

## Python/C++ Parity
The embedded frontend is intended to reproduce the reference Python
frontend algorithmically.
Parity is maintained across:
- FFT implementation
- Float32 arithmetic
- Window generation
- Mel-frequency calculations
- Mel filter geometry
- Log-power conversion
- Dynamic-range clipping
- DCT-II calculation
- Accumulation order
- Feature normalization
- INT8 quantization
The embedded implementation should be validated against Python using
golden input vectors and expected MFCC/INT8 outputs.

## INT8 Quantization
After MFCC extraction, each coefficient is normalized using training-set
statistics:
	normalized = (MFCC - TRAIN_MEAN) / TRAIN_STD
The normalized value is then converted using the TFLite affine
quantization parameters:
	q = round(real / scale) + zero_point
The resulting value is clamped to the signed INT8 range:
	[-128, 127]
Training statistics and model quantization parameters must be obtained
from the corresponding training and TFLite artifacts rather than
estimated on the ESP32.
