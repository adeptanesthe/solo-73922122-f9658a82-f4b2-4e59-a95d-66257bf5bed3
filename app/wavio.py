"""PCM16 单声道 WAV 的读取与写出（仅支持本任务约定的格式）。"""
from __future__ import annotations

import io
import wave


class WavFormatError(ValueError):
    """上传文件不是合法的单声道 PCM16 WAV。"""


def read_mono_pcm16_wav(data: bytes) -> tuple[bytes, int, int]:
    """校验并解析上传的 WAV，返回 (PCM16 裸字节, 采样率, 样点数)。"""
    try:
        wf = wave.open(io.BytesIO(data), "rb")
    except (wave.Error, EOFError) as exc:
        raise WavFormatError(f"无法解析 WAV 文件：{exc}") from exc
    with wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)
    if n_channels != 1:
        raise WavFormatError(f"只支持单声道，当前为 {n_channels} 声道")
    if sampwidth != 2:
        raise WavFormatError(f"只支持 PCM16（16 位），当前位宽为 {sampwidth * 8} 位")
    if not (1 <= sample_rate <= 192000):
        raise WavFormatError(f"采样率异常：{sample_rate}")
    if n_frames <= 0:
        raise WavFormatError("录音为空（没有样点）")
    if len(pcm) != n_frames * 2:
        raise WavFormatError("WAV 数据长度与帧计数不一致")
    return pcm, sample_rate, n_frames


def build_mono_pcm16_wav(pcm: bytes, sample_rate: int) -> bytes:
    """把 PCM16 裸字节打包成完整 WAV。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
