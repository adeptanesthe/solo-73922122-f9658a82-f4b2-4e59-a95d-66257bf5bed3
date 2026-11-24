"""引擎正确性、量化误差与访问复杂度核对（不依赖 Flask）。

运行：PYTHONPATH=. python tests/test_core.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.engine import (  # noqa: E402
    CHUNK, AffineSegTree, CountFenwick, Project, TimecodeOutOfRange,
    chunk_summary, quantize_pcm16,
)

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok  " if cond else " FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def reference_filter(kept: np.ndarray, a: float) -> np.ndarray:
    """先拼接保留样点、从零状态顺序滤波再量化（题目中的对照实现）。"""
    y = np.empty(kept.shape[0], dtype=np.float64)
    s = 0.0
    for n in range(kept.shape[0]):
        s = a * s + (1.0 - a) * kept[n]
        y[n] = s
    return quantize_pcm16(y)


def make_project(samples: np.ndarray, a: float) -> Project:
    d = tempfile.mkdtemp(prefix="cliplab-")
    path = os.path.join(d, "input.pcm")
    samples.astype("<i2").tofile(path)
    return Project("t", path, 16000, samples.shape[0], a)


def test_fenwick() -> None:
    print("[1] Fenwick 计数树")
    rng = np.random.default_rng(1)
    counts = rng.integers(1, 5, size=37).tolist()
    fw = CountFenwick(counts)
    arr = counts[:]
    for r in range(len(counts) + 1):
        check(fw.prefix(r) == sum(arr[:r]), f"prefix({r})")
    pos = np.cumsum(arr)
    k = 0
    for block, c in enumerate(arr):
        for _ in range(c):
            check(fw.kth(k) == block, f"kth({k}) -> {block}")
            k += 1
    fw.add(5, -arr[5]); arr[5] = 0
    fw.add(10, arr[10]); arr[10] *= 2
    for r in range(len(counts) + 1):
        check(fw.prefix(r) == sum(arr[:r]), f"更新后 prefix({r})")
    k = 0
    for block, c in enumerate(arr):
        for _ in range(c):
            check(fw.kth(k) == block, f"更新后 kth({k}) -> {block}")
            k += 1


def test_affine_against_loop() -> None:
    print("[2] 仿射摘要与逐样点递推一致（任意输入状态）")
    rng = np.random.default_rng(2)
    a = 0.73
    for length in (1, 2, 17, 4096):
        x = rng.integers(-30000, 30000, size=length).astype(np.float64)
        w, b = chunk_summary(x, a)
        for y_in in (-12345.0, 0.0, 999.5, 30000.0):
            s = y_in
            for v in x:
                s = a * s + (1.0 - a) * v
            got = w * y_in + b
            check(abs(got - s) <= 1e-8, f"L={length} y_in={y_in}: 仿射 {got:.6f} vs 递推 {s:.6f}")


def test_segment_tree_compose() -> None:
    print("[3] 线段树前缀复合")
    rng = np.random.default_rng(3)
    a = 0.91
    n = 23
    xs = [rng.integers(-1000, 1000, size=rng.integers(1, 9)).astype(np.float64)
          for _ in range(n)]
    leaves = [chunk_summary(x, a) for x in xs]
    tree = AffineSegTree(leaves)
    for r in range(n + 1):
        w, b = tree.prefix(r)
        s = 0.0
        for x in xs[:r]:
            for v in x:
                s = a * s + (1.0 - a) * v
        check(abs(b - s) <= 1e-8 and abs(w - a ** sum(len(x) for x in xs[:r])) <= 1e-12,
              f"prefix({r}) 零状态结果 {b:.6f} vs {s:.6f}")
    # 撤下叶子 -> 恒等
    tree.set_leaf(7, (1.0, 0.0))
    leaves[7] = (1.0, 0.0)
    w1, b1 = tree.prefix(n)
    s = 0.0
    for j, x in enumerate(xs):
        if j == 7:
            continue
        for v in x:
            s = a * s + (1.0 - a) * v
    check(abs(b1 - s) <= 1e-8, "撤下叶子后前缀等于跳过该块")


def test_end_to_end() -> None:
    print("[4] 端到端：随机撤块 + 跳听 vs 先拼接顺序滤波（误差 ≤ 1 量化单位）")
    rng = np.random.default_rng(4)
    for a in (0.0, 0.3, 0.85, 0.99):
        # 含非整块结尾：总样点不是 4096 的整数倍
        total = 6 * CHUNK + 123
        x = rng.integers(-32000, 32000, size=total).astype(np.int16)
        proj = make_project(x, a)
        n = proj.n
        off_mask = set(rng.choice(n, size=max(1, n // 3), replace=False).tolist())
        for i in off_mask:
            proj.set_chunk_active(i, False)

        kept_parts = [x[b * CHUNK:min((b + 1) * CHUNK, total)]
                      for b in range(n) if b not in off_mask]
        ref = reference_filter(np.concatenate(kept_parts), a)
        check(proj.kept_samples() == ref.shape[0],
              f"a={a}: 保留样点数 {proj.kept_samples()} == {ref.shape[0]}")

        # 对每个保留块的块首（成片时码）做跳听，逐样点比对到块末
        cursor = 0
        max_diff = 0
        for b in range(n):
            if b in off_mask:
                continue
            length = min(CHUNK, total - b * CHUNK)
            chunk, offset, out = proj.audition(cursor)
            check(chunk == b and offset == 0, f"a={a}: 块首定位 ({chunk},{offset}) == ({b},0)")
            seg = ref[cursor:cursor + length]
            check(out.shape[0] == seg.shape[0], f"a={a} 块{b}: 试听长度 {out.shape[0]} == {seg.shape[0]}")
            d = int(np.max(np.abs(out.astype(np.int32) - seg.astype(np.int32))))
            max_diff = max(max_diff, d)
            check(np.array_equal(out, seg) or d <= 1, f"a={a} 块{b}: 最大差 {d} ≤ 1")
            cursor += length
        print(f"    → a={a} 全片跳听最大量化差 = {max_diff}")

        # 块内任意偏移
        for k in (0, 1, 777, cursor - 1, cursor // 2):
            chunk, offset, out = proj.audition(k)
            check(out.shape[0] == proj.chunk_counts[chunk] - offset,
                  f"a={a} k={k}: 试听长度到块末")
            d = int(np.max(np.abs(out.astype(np.int32) - ref[k:k + out.shape[0]].astype(np.int32))))
            check(d <= 1, f"a={a} k={k} (块{chunk}+{offset}): 最大差 {d} ≤ 1")


def test_locate_and_range() -> None:
    print("[5] 时码定位与越界提示")
    rng = np.random.default_rng(5)
    total = 4 * CHUNK + 10
    x = rng.integers(-1000, 1000, size=total).astype(np.int16)
    proj = make_project(x, 0.8)
    proj.set_chunk_active(1, False)
    kept_total = proj.kept_samples()
    # 直接枚举所有成片样点号
    expect = []
    for b in range(proj.n):
        if b == 1:
            continue
        for off in range(proj.chunk_counts[b]):
            expect.append((b, off))
    check(len(expect) == kept_total, "枚举定位条目数等于保留样点数")
    for k, (eb, eo) in enumerate(expect):
        b, o = proj.locate(k)
        if (b, o) != (eb, eo):
            check(False, f"locate({k}) -> ({b},{o})，应为 ({eb},{eo})")
    check(True, "全部成片样点号定位正确")
    for bad in (-1, kept_total, kept_total + 100):
        try:
            proj.locate(bad)
            check(False, f"k={bad} 应报越界")
        except TimecodeOutOfRange:
            check(True, f"k={bad} 越界抛出 TimecodeOutOfRange")


def test_access_complexity() -> None:
    print("[6] 访问复杂度：撤/恢复只碰摘要，跳听只读 1 个原始块")
    rng = np.random.default_rng(6)
    sizes = [16, 64, 256]
    prev = None
    for n in sizes:
        x = rng.integers(-1000, 1000, size=n * CHUNK).astype(np.int16)
        proj = make_project(x, 0.8)
        proj.reset_counters()
        proj.set_chunk_active(n // 2, False)
        t_touches = proj.fenwick.touches + proj.segtree.touches
        bound = 8 * math.ceil(math.log2(n + 1)) + 4
        check(proj.chunk_loads == 0, f"N={n}: 撤块不读原始块（loads={proj.chunk_loads}）")
        check(t_touches <= bound, f"N={n}: 撤块摘要访问 {t_touches} ≤ O(log N) 界 {bound}")

        proj.reset_counters()
        proj.set_chunk_active(n // 2, True)
        t_touches = proj.fenwick.touches + proj.segtree.touches
        check(proj.chunk_loads == 0, f"N={n}: 恢复不读原始块（loads={proj.chunk_loads}）")
        check(t_touches <= bound, f"N={n}: 恢复摘要访问 {t_touches} ≤ {bound}")

        proj.set_chunk_active(n // 2, False)
        proj.reset_counters()
        _c, _o, _w = proj.audition(12345)
        a_touches = proj.fenwick.touches + proj.segtree.touches
        check(proj.chunk_loads == 1, f"N={n}: 跳听只读 1 个原始块（loads={proj.chunk_loads}）")
        check(a_touches <= bound, f"N={n}: 跳听摘要访问 {a_touches} ≤ {bound}")
        if prev is not None:
            growth = t_touches / prev
            check(growth < 1.6, f"N 翻 4 倍时摘要访问增长 {growth:.2f} 倍（亚线性/对数级）")
        prev = t_touches


def test_a_zero() -> None:
    print("[7] 边界 a=0：滤波器退化为直通")
    rng = np.random.default_rng(7)
    x = rng.integers(-32768, 32767, size=2 * CHUNK + 5).astype(np.int16)
    proj = make_project(x, 0.0)
    proj.set_chunk_active(0, False)
    _c, offset, out = proj.audition(0)
    check(offset == 0, "a=0: 成片 0 号样点落在保留块块首")
    ref_seg = x[CHUNK:CHUNK + out.shape[0]]
    check(np.array_equal(out.astype(np.int32), ref_seg.astype(np.int32)),
          "a=0 撤下首块后块首跳听等于命中块原始 PCM")


if __name__ == "__main__":
    test_fenwick()
    test_affine_against_loop()
    test_segment_tree_compose()
    test_end_to_end()
    test_locate_and_range()
    test_access_complexity()
    test_a_zero()
    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败")
        sys.exit(1)
    print("全部通过")
