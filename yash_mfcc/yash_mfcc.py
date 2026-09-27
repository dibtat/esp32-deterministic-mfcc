
"""Deterministic MFCC frontend matching the ESP32 implementation.

Public API intentionally has no NumPy dependency:
    from yash_mfcc import MFCC
    mfcc = MFCC()
    X = mfcc.extract(audio_int16)

The implementation uses explicit IEEE-754 float32 rounding after arithmetic.
For true cross-platform bit-for-bit parity, the scalar transcendental
functions (sin/cos/log10) must also come from the same implementation on
both sides; see README.md.
"""
from .mfcc import MFCC, Config

__all__ = ["MFCC", "Config"]
__version__ = "1.0.0"


import math
import struct
from dataclasses import dataclass
from typing import Sequence, List


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def i32_round(x: float) -> int:
    # Matches the intended C/C++ round-to-nearest behavior for finite values.
    # Python's round() is bankers rounding, so do not use it here.
    x = float(x)
    return int(math.floor(x + 0.5)) if x >= 0.0 else int(math.ceil(x - 0.5))


@dataclass(frozen=True)
class Config:
    sample_rate: int = 16000
    audio_samples: int = 16000
    n_fft: int = 1024
    win_length: int = 640
    hop_length: int = 320
    n_mels: int = 128
    n_mfcc: int = 13
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    top_db: float = 80.0


class MFCC:
    """ESP32-parity MFCC frontend, implemented without NumPy/librosa."""

    def __init__(self, config: Config = Config()):
        self.cfg = config
        c = config
        if c.n_fft & (c.n_fft - 1):
            raise ValueError("n_fft must be a power of two")
        if c.win_length > c.n_fft:
            raise ValueError("win_length must not exceed n_fft")

        self.hann = [f32(0.0)] * c.win_length
        self.dct = [[f32(0.0)] * c.n_mels for _ in range(c.n_mfcc)]
        self.fft_cos = [f32(0.0)] * (c.n_fft // 2)
        self.fft_sin = [f32(0.0)] * (c.n_fft // 2)

        self.mel_left_hz = [f32(0.0)] * c.n_mels
        self.mel_center_hz = [f32(0.0)] * c.n_mels
        self.mel_right_hz = [f32(0.0)] * c.n_mels
        self.mel_enorm = [f32(0.0)] * c.n_mels
        self.mel_left_bin = [0] * c.n_mels
        self.mel_right_bin = [0] * c.n_mels

        self._init_tables()

    @staticmethod
    def hz_to_mel(hz: float) -> float:
        hz = f32(hz)
        if hz < f32(1000.0):
            return f32(f32(3.0) * hz / f32(200.0))
        return f32(
            f32(15.0)
            + f32(27.0)
            * f32(
                f32(math.log10(f32(hz / f32(1000.0))))
                / f32(math.log10(f32(6.4)))
            )
        )

    @staticmethod
    def mel_to_hz(mel: float) -> float:
        mel = f32(mel)
        if mel < f32(15.0):
            return f32(f32(200.0) * mel / f32(3.0))
        return f32(
            f32(1000.0)
            * f32(
                10.0 ** f32(
                    f32(mel - f32(15.0))
                    * f32(math.log10(f32(6.4)))
                    / f32(27.0)
                )
            )
        )

    def _init_tables(self):
        c = self.cfg
        pi = f32(math.pi)

        for n in range(c.win_length):
            angle = f32(f32(2.0) * pi * f32(n) / f32(c.win_length))
            self.hann[n] = f32(
                f32(0.5) - f32(f32(0.5) * f32(math.cos(angle)))
            )

        inv_sqrt = f32(1.0 / math.sqrt(f32(c.n_mels)))
        sqrt2 = f32(math.sqrt(f32(2.0) / f32(c.n_mels)))

        for k in range(c.n_mfcc):
            scale = inv_sqrt if k == 0 else sqrt2
            for n in range(c.n_mels):
                angle = f32(
                    pi * f32(f32(n) + f32(0.5))
                    * f32(k) / f32(c.n_mels)
                )
                self.dct[k][n] = f32(scale * f32(math.cos(angle)))

        for k in range(c.n_fft // 2):
            angle = f32(
                -f32(2.0) * pi * f32(k) / f32(c.n_fft)
            )
            self.fft_cos[k] = f32(math.cos(angle))
            self.fft_sin[k] = f32(math.sin(angle))

        mel_min = self.hz_to_mel(c.mel_fmin)
        mel_max = self.hz_to_mel(c.mel_fmax)
        mel_step = f32((mel_max - mel_min) / f32(c.n_mels + 1))
        bin_hz = f32(f32(c.sample_rate) / f32(c.n_fft))

        for m in range(c.n_mels):
            left = self.mel_to_hz(
                f32(mel_min + f32(mel_step * f32(m)))
            )
            center = self.mel_to_hz(
                f32(mel_min + f32(mel_step * f32(m + 1)))
            )
            right = self.mel_to_hz(
                f32(mel_min + f32(mel_step * f32(m + 2)))
            )
            self.mel_left_hz[m] = left
            self.mel_center_hz[m] = center
            self.mel_right_hz[m] = right
            self.mel_enorm[m] = f32(
                f32(2.0) / f32(right - left)
            )

            lb = int(math.floor(f32(left / bin_hz)))
            rb = int(math.ceil(f32(right / bin_hz)))
            self.mel_left_bin[m] = max(0, lb)
            self.mel_right_bin[m] = min(c.n_fft // 2, rb)

    def _fft(self, buf: List[float]):
        n = self.cfg.n_fft
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                tr = f32(buf[2 * i])
                ti = f32(buf[2 * i + 1])
                buf[2 * i] = f32(buf[2 * j])
                buf[2 * i + 1] = f32(buf[2 * j + 1])
                buf[2 * j] = tr
                buf[2 * j + 1] = ti

        length = 2
        while length <= n:
            half = length >> 1
            step = n // length
            i = 0
            while i < n:
                for jj in range(half):
                    a = i + jj
                    b = a + half
                    tw = jj * step
                    wr = self.fft_cos[tw]
                    wi = self.fft_sin[tw]
                    br = f32(buf[2 * b])
                    bi = f32(buf[2 * b + 1])
                    tr = f32(f32(br * wr) - f32(bi * wi))
                    ti = f32(f32(br * wi) + f32(bi * wr))
                    ar = f32(buf[2 * a])
                    ai = f32(buf[2 * a + 1])
                    buf[2 * a] = f32(ar + tr)
                    buf[2 * a + 1] = f32(ai + ti)
                    buf[2 * b] = f32(ar - tr)
                    buf[2 * b + 1] = f32(ai - ti)
                i += length
            length <<= 1

    def _mel_energy(self, power, m: int) -> float:
        c = self.cfg
        left = self.mel_left_hz[m]
        center = self.mel_center_hz[m]
        right = self.mel_right_hz[m]
        enorm = self.mel_enorm[m]
        bin_hz = f32(f32(c.sample_rate) / f32(c.n_fft))
        total = f32(0.0)

        for k in range(self.mel_left_bin[m], self.mel_right_bin[m] + 1):
            hz = f32(bin_hz * f32(k))
            if hz < center:
                den = f32(center - left)
                weight = f32(hz - left) / den if den > f32(0.0) else f32(0.0)
            else:
                den = f32(right - center)
                weight = f32(right - hz) / den if den > f32(0.0) else f32(0.0)

            if weight > f32(0.0):
                total = f32(
                    total + f32(f32(power[k] * weight) * enorm)
                )
        return total

    def _frame(self, audio, frame_start):
        c = self.cfg
        buf = [f32(0.0)] * (2 * c.n_fft)
        off = (c.n_fft - c.win_length) // 2

        for n in range(c.win_length):
            src = frame_start + n
            if 0 <= src < c.audio_samples:
                sample = f32(f32(audio[src]) / f32(32768.0))
            else:
                sample = f32(0.0)
            buf[2 * (off + n)] = f32(sample * self.hann[n])

        self._fft(buf)

        # Reproduce the C++ scratch-buffer overwrite exactly.
        power = buf
        for k in range(c.n_fft // 2 + 1):
            re = f32(buf[2 * k])
            im = f32(buf[2 * k + 1])
            power[k] = f32(f32(re * re) + f32(im * im))

        mel_log = [f32(0.0)] * c.n_mels
        max_db = float("-inf")
        for m in range(c.n_mels):
            e = self._mel_energy(power, m)
            if e < f32(1.0e-10):
                e = f32(1.0e-10)
            db = f32(f32(10.0) * f32(math.log10(e)))
            mel_log[m] = db
            if db > max_db:
                max_db = db

        min_db = f32(max_db - f32(c.top_db))
        for m in range(c.n_mels):
            if mel_log[m] < min_db:
                mel_log[m] = min_db

        out = [f32(0.0)] * c.n_mfcc
        for k in range(c.n_mfcc):
            acc = f32(0.0)
            for m in range(c.n_mels):
                acc = f32(acc + f32(self.dct[k][m] * mel_log[m]))
            out[k] = acc
        return out

    def extract(self, audio: Sequence[int]) -> List[List[float]]:
        c = self.cfg
        if len(audio) < c.audio_samples:
            a = [0] * c.audio_samples
            for i, x in enumerate(audio):
                a[i] = int(x)
        else:
            a = [int(x) for x in audio[:c.audio_samples]]

        features = []
        for frame in range(c.audio_samples // c.hop_length + 1):
            start = frame * c.hop_length - (c.n_fft // 2)
            features.append(self._frame(a, start))
        return features

    def extract_flat(self, audio: Sequence[int]) -> List[float]:
        return [x for row in self.extract(audio) for x in row]

    @staticmethod
    def normalize(features, mean, std):
        mean = f32(mean)
        std = f32(std)
        return [[f32((f32(x) - mean) / std) for x in row] for row in features]

    @staticmethod
    def quantize(features, scale, zero_point):
        scale = f32(scale)
        zp = int(zero_point)
        out = []
        for row in features:
            qrow = []
            for x in row:
                q = i32_round(f32(f32(x) / scale) + f32(zp))
                qrow.append(max(-128, min(127, q)))
            out.append(qrow)
        return out

    def extract_int8(self, audio, mean, std, scale, zero_point):
        return self.quantize(self.normalize(self.extract(audio), mean, std),
                             scale, zero_point)
