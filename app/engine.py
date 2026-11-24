"""分块剪辑引擎。

核心思想
========
录音按 ``CHUNK`` 个样点分块，每块可以撤下（剪辑师剪掉停顿）或恢复。
在任意一组撤下决策之上，需要支持两个亚线性操作：

1. **成片时码定位**：成片第 k 个保留样点，对应哪个原块、块内偏移多少？
2. **接缝跳听**：从成片第 k 个样点到其所在块块末的低通滤波结果。

数据结构
--------
* :class:`CountFenwick` —— 每块保留样点数（0 或 CHUNK，末块可能更短）的
  Fenwick 树，支持单点增减、前缀和、第 k 个保留位置查询，全部 O(log N)。
* :class:`AffineSegTree` —— 一阶低通 ``y[n] = a*y[n-1] + (1-a)*x[n]`` 对一块
  样点的作用是关于输入状态的仿射变换 ``y_out = w*y_in + b``。线段树叶子存
  每块的 (w, b)，撤下的块存恒等变换 (1, 0)——相当于把该块从状态流里删掉，
  内部节点存左右两段变换的复合。前缀查询即得到“进入某块之前”的滤波器状态。

撤下/恢复一块只改两个树的 O(log N) 个摘要节点，不碰原始样点；跳听只读命中
的那一块原始 PCM，其余状态全部来自摘要。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

CHUNK = 4096


class TimecodeOutOfRange(Exception):
    """成片样点号越界。"""


def compose(g: tuple[float, float], h: tuple[float, float]) -> tuple[float, float]:
    """先做 g 再做 h 的复合仿射变换。

    g(y) = wg*y + bg, h(y) = wh*y + bh
    h(g(y)) = (wh*wg)*y + (wh*bg + bh)
    """
    wg, bg = g
    wh, bh = h
    return (wh * wg, wh * bg + bh)


IDENTITY = (1.0, 0.0)


class CountFenwick:
    """块保留样点数的 Fenwick 树（1 基内部存储）。"""

    def __init__(self, counts: list[int]):
        n = len(counts)
        self.n = n
        bit = [0] * (n + 1)
        for i, v in enumerate(counts, start=1):
            bit[i] += v
            j = i + (i & -i)
            if j <= n:
                bit[j] += bit[i]
        self.bit = bit
        self.touches = 0

    def add(self, i: int, delta: int) -> None:
        """0 基位置 i 单点增减。"""
        i += 1
        n = self.n
        bit = self.bit
        while i <= n:
            bit[i] += delta
            self.touches += 1
            i += i & -i

    def prefix(self, r: int) -> int:
        """前 r 个块 [0, r) 的保留样点总数。"""
        s = 0
        bit = self.bit
        i = r
        while i > 0:
            s += bit[i]
            self.touches += 1
            i -= i & -i
        return s

    def kth(self, k: int) -> int:
        """第 k（0 基）个保留样点所在的块号。"""
        bit = self.bit
        n = self.n
        idx = 0
        step = 1 << (n.bit_length() - 1)
        while step:
            nxt = idx + step
            if nxt <= n and bit[nxt] <= k:
                idx = nxt
                k -= bit[nxt]
            self.touches += 1
            step >>= 1
        return idx


class AffineSegTree:
    """块仿射摘要的线段树，叶子在 [size, 2*size)。"""

    def __init__(self, leaves: list[tuple[float, float]]):
        n = len(leaves)
        size = 1
        while size < n:
            size <<= 1
        self.size = size
        self.n = n
        tree = [IDENTITY] * (2 * size)
        for i, leaf in enumerate(leaves):
            tree[size + i] = leaf
        for i in range(size - 1, 0, -1):
            tree[i] = compose(tree[2 * i], tree[2 * i + 1])
        self.tree = tree
        self.touches = 0

    def set_leaf(self, i: int, summary: tuple[float, float]) -> None:
        """把块 i 的摘要换成 summary（撤下 -> 恒等，恢复 -> 原滤波摘要）。"""
        t = self.tree
        p = self.size + i
        t[p] = summary
        self.touches += 1
        p >>= 1
        while p:
            t[p] = compose(t[2 * p], t[2 * p + 1])
            self.touches += 1
            p >>= 1

    def prefix(self, r: int) -> tuple[float, float]:
        """前 r 个块 [0, r) 顺序复合后的仿射变换；r=0 时为恒等。"""
        t = self.tree
        l = self.size
        rr = self.size + r
        left_acc = IDENTITY
        right_acc = IDENTITY
        while l < rr:
            if l & 1:
                left_acc = compose(left_acc, t[l])
                self.touches += 1
                l += 1
            if rr & 1:
                rr -= 1
                right_acc = compose(t[rr], right_acc)
                self.touches += 1
            l >>= 1
            rr >>= 1
        return compose(left_acc, right_acc)


def chunk_summary(x: np.ndarray, a: float) -> tuple[float, float]:
    """一块样点 x 对一阶低通的仿射摘要 (w, b)：y_out = w*y_in + b。

    递推 L 步后 ``y_out = a^L * y_in + (1-a) * sum_n a^(L-1-n) x[n]``，
    全程 float64、中途不取整，与逐样点递推共用同一公式。
    """
    L = x.shape[0]
    if L == 0:
        return IDENTITY
    xf = x.astype(np.float64)
    w = float(a) ** L
    # a^(L-1), a^(L-2), ..., a^0 ；L <= 4096，整数指数在 float64 中精确
    powers = np.float64(a) ** np.arange(L - 1, -1, -1, dtype=np.float64)
    b = float((1.0 - a) * np.dot(powers, xf))
    return (w, b)


def quantize_pcm16(y: np.ndarray) -> np.ndarray:
    """float64 滤波结果量化回 PCM16（四舍五入 + 夹幅）。"""
    return np.clip(np.rint(y), -32768.0, 32767.0).astype("<i2")


@dataclass
class ChunkInfo:
    index: int
    samples: int          # 该块原始样点数
    active: bool
    start_sample: int     # 在原始录音中的起始样点号


class Project:
    """一个上传录音的剪辑工程（线程安全由调用方加锁保证）。"""

    def __init__(self, project_id: str, pcm_path: str, sample_rate: int,
                 total_samples: int, a: float):
        self.id = project_id
        self.pcm_path = pcm_path
        self.sample_rate = sample_rate
        self.total_samples = total_samples
        self.a = float(a)
        self.n = math.ceil(total_samples / CHUNK)
        self.chunk_loads = 0             # 原始样点块被读取的次数（供测试核对）

        counts: list[int] = []
        leaves: list[tuple[float, float]] = []
        for i in range(self.n):
            x = self._read_chunk(i)
            counts.append(x.shape[0])
            leaves.append(chunk_summary(x, self.a))

        self.chunk_counts = counts
        self.active = [True] * self.n
        self.summaries = leaves          # 恢复时直接用，不重读原始样点
        self.fenwick = CountFenwick(counts)
        self.segtree = AffineSegTree(leaves)

    # ---------- 原始样点：只允许按块随机读取 ----------
    def _read_chunk(self, i: int) -> np.ndarray:
        start = i * CHUNK
        count = min(CHUNK, self.total_samples - start)
        with open(self.pcm_path, "rb") as f:
            f.seek(start * 2)  # PCM16 -> 每样点 2 字节
            raw = f.read(count * 2)
        self.chunk_loads += 1
        return np.frombuffer(raw, dtype="<i2")

    # ---------- 工程信息 ----------
    def kept_samples(self) -> int:
        return self.fenwick.prefix(self.n)

    def list_chunks(self) -> list[ChunkInfo]:
        return [
            ChunkInfo(
                index=i,
                samples=self.chunk_counts[i],
                active=self.active[i],
                start_sample=i * CHUNK,
            )
            for i in range(self.n)
        ]

    # ---------- 撤下 / 恢复：只访问摘要 ----------
    def set_chunk_active(self, i: int, active: bool) -> None:
        if not (0 <= i < self.n):
            raise IndexError(f"块号 {i} 越界，共 {self.n} 块")
        if self.active[i] == active:
            return
        self.active[i] = active
        self.fenwick.add(i, self.chunk_counts[i] if active else -self.chunk_counts[i])
        self.segtree.set_leaf(i, self.summaries[i] if active else IDENTITY)

    # ---------- 成片时码定位 ----------
    def locate(self, k: int) -> tuple[int, int]:
        """成片 0 基样点号 k -> (原块号, 块内偏移)。"""
        kept = self.kept_samples()
        if not (0 <= k < kept):
            raise TimecodeOutOfRange(
                f"成片样点号 {k} 越界：当前保留 {kept} 个样点，"
                f"有效范围 0..{kept - 1}"
            )
        chunk = self.fenwick.kth(k)
        before = self.fenwick.prefix(chunk)
        offset = k - before
        return chunk, offset

    # ---------- 跳听：只读命中块的原始样点 ----------
    def audition(self, k: int) -> tuple[int, int, np.ndarray]:
        """返回 (原块号, 块内偏移, 从 k 到块末的 PCM16 滤波结果)。"""
        chunk, offset = self.locate(k)

        # 进入该块之前的滤波器状态：前缀块复合仿射作用于零状态 y[-1]=0
        _, state_in = self.segtree.prefix(chunk)

        x = self._read_chunk(chunk).astype(np.float64)
        aa = self.a
        one_minus_a = 1.0 - aa
        out = np.empty(x.shape[0], dtype=np.float64)
        s = float(state_in)
        # 块内逐样点递推，不取整
        for n in range(x.shape[0]):
            s = aa * s + one_minus_a * x[n]
            out[n] = s
        return chunk, offset, quantize_pcm16(out[offset:])

    def reset_counters(self) -> None:
        self.fenwick.touches = 0
        self.segtree.touches = 0
        self.chunk_loads = 0
