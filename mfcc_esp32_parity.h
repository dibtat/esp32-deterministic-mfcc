#pragma once
/*
 * ESP32 MFCC frontend
 *
 * PURPOSE
 * -------
 * Convert a 1-second 16 kHz mono PCM window into the INT8 tensor expected by
 * the current model:
 *
 *     audio:          16000 samples
 *     MFCC:           51 x 13
 *     normalized:     (MFCC - TRAIN_MEAN) / TRAIN_STD
 *     model input:    int8[51 * 13]
 *
 * This frontend is the deployment reference for the matching Python training
 * implementation. Python must reproduce these operations rather than calling
 * librosa.feature.mfcc() directly.
 *
 * Frontend:
 *   int16 PCM / 32768
 *   -> centered 640-sample Hann window
 *   -> 1024-point radix-2 FFT
 *   -> power spectrum
 *   -> 128 Slaney-normalized Mel filters
 *   -> 10*log10(power)
 *   -> 80 dB dynamic-range clipping
 *   -> orthonormal DCT-II
 *   -> 13 MFCC
 *
 * Frame geometry:
 *   sample rate = 16000 Hz
 *   FFT = 1024
 *   window = 640
 *   hop = 320
 *   center = true
 *   zero padding = N_FFT/2
 *   output = 51 x 13
 *
 * Normalization statistics and model input quantization parameters are supplied
 * separately from the training artifacts.
 *
 * Design goals:
 *   - no dynamic allocation
 *   - no large Mel filter-bank table in RAM
 *   - precompute DSP constants once
 *   - caller supplies the 16000-sample audio window
 *   - output is exactly 51*13 = 663 int8 values
 *
 * PARITY:
 *   The matching Python implementation must use the same radix-2 FFT, float32
 *   arithmetic, table generation, Mel calculations, log clipping and DCT
 *   accumulation order. This makes the C++ and Python frontends algorithmically
 *   identical; a golden-vector test should still be used to verify the build.
 */

#ifndef PI
#define PI 3.14159265358979323846f
#endif
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace mfcc {

static constexpr int SAMPLE_RATE   = 16000;
static constexpr int AUDIO_SAMPLES = 16000;

static constexpr int N_FFT      = 1024;
static constexpr int WIN_LENGTH = 640;
static constexpr int HOP_LENGTH = 320;

static constexpr int N_MELS = 128;
static constexpr int N_MFCC = 13;

/*
 * librosa with center=True:
 * n_frames = 1 + floor((16000 + 2*(1024/2) - 1024) / 320) = 51
 */
static constexpr int N_FRAMES = 51;
static constexpr int FEATURE_COUNT = N_FRAMES * N_MFCC;

/*
 * REPLACE THESE FOUR VALUES FROM YOUR TRAINING/TFLITE ARTIFACTS.
 *
 * TRAIN_MEAN / TRAIN_STD:
 *   statistics computed from the training set
 *
 * MODEL_INPUT_SCALE / MODEL_INPUT_ZERO_POINT:
 *   read directly from the INT8 input tensor of the TFLite model.
 *
 * Do NOT estimate these values on the ESP32.
 */
#ifndef MFCC_TRAIN_MEAN
#define MFCC_TRAIN_MEAN 0.0f
#endif

#ifndef MFCC_TRAIN_STD
#define MFCC_TRAIN_STD 1.0f
#endif

#ifndef MFCC_INPUT_SCALE
#define MFCC_INPUT_SCALE 1.0f
#endif

#ifndef MFCC_INPUT_ZERO_POINT
#define MFCC_INPUT_ZERO_POINT 0
#endif

static constexpr float TRAIN_MEAN = MFCC_TRAIN_MEAN;
static constexpr float TRAIN_STD  = MFCC_TRAIN_STD;

static constexpr float INPUT_SCALE =
    MFCC_INPUT_SCALE;
static constexpr int INPUT_ZERO_POINT =
    MFCC_INPUT_ZERO_POINT;

/* librosa/Slaney mel range: 0 Hz .. Nyquist */
static constexpr float MEL_FMIN = 0.0f;
static constexpr float MEL_FMAX = 8000.0f;

/*
 * Internal working memory.
 *
 * FFT buffer:
 *   complex float represented as interleaved real/imag -> 1024*2*4 = 8 KB
 *
 * Window/DCT/twiddles:
 *   roughly 13 KB total.
 *
 * No 128x513 Mel matrix is stored.
 */
static float fft_buf[2 * N_FFT];
static float hann_window[WIN_LENGTH];
static float dct_table[N_MFCC][N_MELS];
static float fft_cos[N_FFT / 2];
static float fft_sin[N_FFT / 2];

/*
 * Precomputed Mel geometry.
 * These are generated once using the same float32 operations as Python.
 * Storing geometry avoids recalculating log/pow for every frame.
 */
static float mel_left_hz[N_MELS];
static float mel_center_hz[N_MELS];
static float mel_right_hz[N_MELS];
static float mel_enorm[N_MELS];
static int mel_left_bin[N_MELS];
static int mel_right_bin[N_MELS];

static bool initialized = false;

static inline float hz_to_mel(float hz) {
    /*
     * librosa's default Slaney formula:
     *   below 1 kHz: linear
     *   above 1 kHz: logarithmic
     */
    if (hz < 1000.0f)
        return 3.0f * hz / 200.0f;

    return 15.0f +
           27.0f * (log10f(hz / 1000.0f) / log10f(6.4f));
}

static inline float mel_to_hz(float mel) {
    if (mel < 15.0f)
        return 200.0f * mel / 3.0f;

    return 1000.0f *
           powf(10.0f, (mel - 15.0f) * log10f(6.4f) / 27.0f);
}

static void init() {
    if (initialized) return;

    /*
     * Periodic Hann window.
     * This is the scipy/librosa "hann" window convention.
     */
    for (int n = 0; n < WIN_LENGTH; ++n) {
        hann_window[n] =
            0.5f - 0.5f * cosf(2.0f * PI * n / WIN_LENGTH);
    }

    /*
     * DCT-II, orthonormal.
     */
    const float inv_sqrt_mels = 1.0f / sqrtf((float)N_MELS);
    const float sqrt_2_over_m = sqrtf(2.0f / (float)N_MELS);

    for (int k = 0; k < N_MFCC; ++k) {
        for (int n = 0; n < N_MELS; ++n) {
            const float scale = (k == 0) ? inv_sqrt_mels
                                         : sqrt_2_over_m;
            dct_table[k][n] =
                scale * cosf(PI * ((float)n + 0.5f) *
                            (float)k / (float)N_MELS);
        }
    }

    /*
     * Twiddle factors for the same radix-2 FFT used by Python.
     */
    for (int k = 0; k < N_FFT / 2; ++k) {
        const float a = -2.0f * PI * (float)k / (float)N_FFT;
        fft_cos[k] = cosf(a);
        fft_sin[k] = sinf(a);
    }

    /*
     * Precompute the exact Mel filter geometry once.
     *
     * This is mathematically the same calculation previously done inside
     * mel_filter_energy(), but avoids repeating it 51*128 times.
     */
    const float mel_min = hz_to_mel(MEL_FMIN);
    const float mel_max = hz_to_mel(MEL_FMAX);
    const float mel_step =
        (mel_max - mel_min) / (float)(N_MELS + 1);
    const float bin_hz =
        (float)SAMPLE_RATE / (float)N_FFT;

    for (int m = 0; m < N_MELS; ++m) {
        mel_left_hz[m] =
            mel_to_hz(mel_min + mel_step * (float)m);

        mel_center_hz[m] =
            mel_to_hz(mel_min + mel_step * (float)(m + 1));

        mel_right_hz[m] =
            mel_to_hz(mel_min + mel_step * (float)(m + 2));

        mel_enorm[m] =
            2.0f / (mel_right_hz[m] - mel_left_hz[m]);

        int left_bin =
            (int)floorf(mel_left_hz[m] / bin_hz);

        int right_bin =
            (int)ceilf(mel_right_hz[m] / bin_hz);

        if (left_bin < 0)
            left_bin = 0;

        if (right_bin > N_FFT / 2)
            right_bin = N_FFT / 2;

        mel_left_bin[m] = left_bin;
        mel_right_bin[m] = right_bin;
    }

    initialized = true;
}

/*
 * In-place radix-2 complex FFT.
 *
 * Input/output layout:
 *   fft_buf[2*i+0] = real
 *   fft_buf[2*i+1] = imag
 *
 * We only need bins 0..512 afterwards.
 */
static void fft1024() {
    /* Bit reversal */
    for (int i = 1, j = 0; i < N_FFT; ++i) {
        int bit = N_FFT >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j) {
            float tr = fft_buf[2*i];
            float ti = fft_buf[2*i + 1];
            fft_buf[2*i] = fft_buf[2*j];
            fft_buf[2*i + 1] = fft_buf[2*j + 1];
            fft_buf[2*j] = tr;
            fft_buf[2*j + 1] = ti;
        }
    }

    for (int len = 2; len <= N_FFT; len <<= 1) {
        const int half = len >> 1;
        const int step = N_FFT / len;

        for (int i = 0; i < N_FFT; i += len) {
            for (int j = 0; j < half; ++j) {
                const int a = i + j;
                const int b = a + half;
                const int tw = j * step;

                const float wr = fft_cos[tw];
                const float wi = fft_sin[tw];

                const float br = fft_buf[2*b];
                const float bi = fft_buf[2*b + 1];

                const float tr = br * wr - bi * wi;
                const float ti = br * wi + bi * wr;

                const float ar = fft_buf[2*a];
                const float ai = fft_buf[2*a + 1];

                fft_buf[2*a]     = ar + tr;
                fft_buf[2*a + 1] = ai + ti;
                fft_buf[2*b]     = ar - tr;
                fft_buf[2*b + 1] = ai - ti;
            }
        }
    }
}

/*
 * Calculate one Slaney-normalized Mel filter directly from FFT power bins.
 *
 * We intentionally calculate triangle weights on the fly instead of storing
 * a 128x513 matrix. This saves ~263 KB of RAM.
 */
static float mel_filter_energy(const float *power, int mel_index) {
    const float left_hz = mel_left_hz[mel_index];
    const float center_hz = mel_center_hz[mel_index];
    const float right_hz = mel_right_hz[mel_index];
    const float enorm = mel_enorm[mel_index];

    const int left_bin = mel_left_bin[mel_index];
    const int right_bin = mel_right_bin[mel_index];

    const float bin_hz =
        (float)SAMPLE_RATE / (float)N_FFT;

    float sum = 0.0f;

    for (int k = left_bin; k <= right_bin; ++k) {
        const float hz = bin_hz * (float)k;
        float weight;

        if (hz < center_hz) {
            const float den = center_hz - left_hz;
            weight = (den > 0.0f)
                     ? (hz - left_hz) / den
                     : 0.0f;
        } else {
            const float den = right_hz - center_hz;
            weight = (den > 0.0f)
                     ? (right_hz - hz) / den
                     : 0.0f;
        }

        if (weight > 0.0f)
            sum += power[k] * weight * enorm;
    }

    return sum;
}

/*
 * Extract one 13-element MFCC vector from a 1024-sample centered frame.
 */
static void frame_mfcc(const int16_t *audio,
                       int frame_start,
                       float *out_mfcc) {
    /*
     * librosa's center=True pads the signal by N_FFT/2 on both sides.
     * Therefore frame_start may be negative or extend past AUDIO_SAMPLES.
     *
     * Constant zero padding matches pad_mode="constant".
     *
     * The actual 640-sample Hann window is centered inside the 1024 FFT
     * frame, matching librosa/scipy's stft convention.
     */
    memset(fft_buf, 0, sizeof(fft_buf));

    const int win_offset = (N_FFT - WIN_LENGTH) / 2;  // 192

    for (int n = 0; n < WIN_LENGTH; ++n) {
        const int src = frame_start + n;

        float sample = 0.0f;
        if (src >= 0 && src < AUDIO_SAMPLES)
            sample = (float)audio[src] / 32768.0f;

        fft_buf[2 * (win_offset + n)] =
            sample * hann_window[n];
    }

    fft1024();

    /*
     * We reuse the beginning of fft_buf as a power-spectrum scratch area.
     * power[k] = real^2 + imag^2.
     *
     * 513 floats fit in the same 1024-complex buffer.
     */
    float *power = fft_buf;

    for (int k = 0; k <= N_FFT / 2; ++k) {
        const float re = fft_buf[2*k];
        const float im = fft_buf[2*k + 1];
        power[k] = re * re + im * im;
    }

    float mel_log[N_MELS];

    /*
     * librosa.power_to_db is applied inside librosa.feature.mfcc through
     * the default mel spectrogram -> power_to_db path.
     *
     * For parity, use a fixed tiny floor before log. The exact reference
     * comparison against Python should determine whether this needs to be
     * adjusted for the installed librosa version.
     */
    /*
     * Match librosa.power_to_db(..., ref=1.0, amin=1e-10, top_db=80).
     *
     * The top_db operation is important: it clips very quiet Mel bands to
     * 80 dB below the loudest band in this frame. Omitting it can produce
     * noticeably different MFCCs.
     */
    float max_db = -INFINITY;

    for (int m = 0; m < N_MELS; ++m) {
        float e = mel_filter_energy(power, m);
        if (e < 1.0e-10f)
            e = 1.0e-10f;

        mel_log[m] = 10.0f * log10f(e);

        if (mel_log[m] > max_db)
            max_db = mel_log[m];
    }

    const float min_db = max_db - 80.0f;

    for (int m = 0; m < N_MELS; ++m) {
        if (mel_log[m] < min_db)
            mel_log[m] = min_db;
    }

    for (int k = 0; k < N_MFCC; ++k) {
        float acc = 0.0f;
        for (int m = 0; m < N_MELS; ++m)
            acc += dct_table[k][m] * mel_log[m];

        out_mfcc[k] = acc;
    }
}

/*
 * Convert one 16000-sample int16 PCM window into the model's int8 input.
 *
 * audio:
 *   signed 16-bit PCM, mono, exactly 16 kHz, exactly 16000 samples.
 *
 * output:
 *   663 bytes in row-major [51][13] order.
 *
 * Return value:
 *   true if parameters are valid.
 */
/*
 * Extract raw float32 MFCCs before normalization/quantization.
 *
 * output layout:
 *   output[frame * N_MFCC + coefficient]
 *
 * This function is intended for golden-vector validation against Python.
 */
static bool extract_float(const int16_t *audio, float *output) {
    if (!audio || !output)
        return false;

    init();

    float mfcc[N_MFCC];

    for (int frame = 0; frame < N_FRAMES; ++frame) {
        const int frame_start =
            frame * HOP_LENGTH - (N_FFT / 2);

        frame_mfcc(audio, frame_start, mfcc);

        for (int c = 0; c < N_MFCC; ++c)
            output[frame * N_MFCC + c] = mfcc[c];
    }

    return true;
}

/*
 * Convert raw MFCCs to normalized INT8 model input.
 *
 * This is deliberately separate from extract_float() so the DSP frontend
 * can be validated independently of training normalization and TFLite
 * quantization parameters.
 */
static bool extract_int8(const int16_t *audio, int8_t *output) {
    if (!audio || !output)
        return false;

    if (TRAIN_STD <= 0.0f || INPUT_SCALE <= 0.0f)
        return false;

    init();

    float mfcc[N_MFCC];

    /*
     * With center=True:
     * frame t is centered at t*HOP_LENGTH.
     * The equivalent padded-array frame starts at:
     *
     *     t*320 - 512
     */
    for (int frame = 0; frame < N_FRAMES; ++frame) {
        const int frame_start =
            frame * HOP_LENGTH - (N_FFT / 2);

        frame_mfcc(audio, frame_start, mfcc);

        for (int c = 0; c < N_MFCC; ++c) {
            /*
             * Same scalar normalization as the current notebook.
             */
            const float normalized =
                (mfcc[c] - TRAIN_MEAN) / TRAIN_STD;

            /*
             * TFLite affine quantization:
             *
             *   q = round(real / scale) + zero_point
             *
             * Clamp to signed int8 range.
             */
            int q = (int)lrintf(normalized / INPUT_SCALE)
                    + INPUT_ZERO_POINT;

            if (q > 127) q = 127;
            if (q < -128) q = -128;

            output[frame * N_MFCC + c] = (int8_t)q;
        }
    }

    return true;
}

} // namespace mfcc
