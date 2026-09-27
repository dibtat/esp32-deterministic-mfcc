
# yash_mfcc

A small, NumPy-free Python MFCC frontend designed around the ESP32 C++ MFCC
algorithm.

## Why no NumPy?

The training pipeline should not depend on a second FFT/Mel/MFCC
implementation. This package owns the frontend. The model-training notebook
only needs:

```python
from yash_mfcc import MFCC

mfcc = MFCC()
X = mfcc.extract(audio_int16)
```

`librosa.feature.mfcc`, `librosa.filters.mel`, and `numpy.fft` are not used.

## Important: what "bit-level parity" actually requires

This package reproduces the C++ operation order and explicitly rounds the
intermediate arithmetic to IEEE-754 float32. However, `sin`, `cos`, and
`log10` are transcendental functions supplied by the host math library.

Therefore:

**Same algorithm + float32 rounding does not mathematically guarantee the
same bits on Python/Windows/Linux and ESP32.**

For genuine bit-for-bit parity, both implementations must use the same
deterministic scalar math implementation, or both must consume the exact
same precomputed float32 tables.

The current package is therefore the clean architectural layer and an
operation-order parity reference. The next step for strict bit identity is
to replace the platform transcendental calls with shared deterministic
functions/constants in both Python and C++.

## Public API

```python
mfcc = MFCC()
features = mfcc.extract(audio_int16)       # 51 x 13, float32-valued Python floats
flat = mfcc.extract_flat(audio_int16)      # 663 values

normalized = MFCC.normalize(features, mean, std)
q = MFCC.quantize(normalized, scale, zero_point)

model_input = mfcc.extract_int8(audio_int16, mean, std, scale, zero_point)
```

The package intentionally keeps normalization and quantization outside the
MFCC algorithm itself.

## Architecture

Training notebook
    -> yash_mfcc
    -> MFCC 51x13
    -> normalization
    -> model

ESP32
    -> mfcc_esp32_parity.h
    -> MFCC 51x13
    -> normalization
    -> int8
    -> TFLite Micro

Both sides therefore have one named frontend contract.
