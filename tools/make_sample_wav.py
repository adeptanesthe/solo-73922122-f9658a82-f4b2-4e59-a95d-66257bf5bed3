"""生成可试剪、跳听的示例“访谈”录音（单声道 PCM16 WAV）。

内容结构按 4096 样点对齐：若干段“讲话”（多正弦 + 噪声，带幅度起伏）之间
夹着整块的“停顿”（近静音），方便直接撤下停顿块再跳听接缝。

用法：
    python tools/make_sample_wav.py sample_audio/interview_demo.wav
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.wavio import build_mono_pcm16_wav  # noqa: E402
from app.engine import CHUNK  # noqa: E406

SR = 16000
N_CHUNKS = 40                      # 40 * 4096 = 163840 样点 ≈ 10.24 s
SILENT_BLOCKS = {3, 4, 9, 16, 17, 25, 32}   # 停顿块：试剪目标
rng = np.random.default_rng(20261124)


def synth_speech(n: int, seed: int) -> np.ndarray:
    """合成一段类似人声的信号：基频 + 谐波 + 噪声，幅度随时间起伏。"""
    r = np.random.default_rng(seed)
    t = np.arange(n) / SR
    f0 = r.uniform(110, 150)
    sig = (
        0.55 * np.sin(2 * np.pi * f0 * t)
        + 0.30 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    sig += 0.05 * r.standard_normal(n)
    # 音节级幅度起伏（3~6 Hz），避免削顶
    env = 0.55 + 0.45 * np.sin(2 * np.pi * r.uniform(3, 6) * t + r.uniform(0, 6))
    sig *= env
    return sig


def main(path: str) -> None:
    total = N_CHUNKS * CHUNK
    pcm = np.empty(total, dtype=np.float64)
    for b in range(N_CHUNKS):
        sl = slice(b * CHUNK, (b + 1) * CHUNK)
        if b in SILENT_BLOCKS:
            # 近静音停顿：底噪量级，不是严格 0，更像真实录音
            pcm[sl] = 0.004 * rng.standard_normal(CHUNK)
        else:
            pcm[sl] = 0.8 * synth_speech(CHUNK, seed=1000 + b)
    pcm16 = np.clip(np.rint(pcm * 32767.0), -32768, 32767).astype("<i2")
    wav = build_mono_pcm16_wav(pcm16.tobytes(), SR)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(wav)
    print(f"wrote {path}: {total} samples, {SR} Hz, {N_CHUNKS} chunks, "
          f"{len(SILENT_BLOCKS)} silent blocks {sorted(SILENT_BLOCKS)}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_audio/interview_demo.wav"
    main(out)
