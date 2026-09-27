
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
