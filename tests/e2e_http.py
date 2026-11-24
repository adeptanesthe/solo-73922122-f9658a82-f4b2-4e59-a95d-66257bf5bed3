"""真实 HTTP 端到端验证（针对运行中的 Flask 服务）。

步骤：
1. 上传示例录音 interview_demo.wav（a=0.9）；
2. 撤下全部停顿块，再恢复其中一块，检查摘要访问量与原始块读取次数；
3. 对若干成片时码（含块首/块中/最后一个样点）跳听，定位结果与 WAV 校验；
4. 独立做“先拼接保留样点、从零状态顺序滤波再量化”，与所有跳听段逐样点比对；
5. 越界时码返回 400 与重选提示。
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
import wave

import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5055"
WAV_PATH = "sample_audio/interview_demo.wav"
A = "0.9"
SILENT = [3, 4, 9, 16, 17, 25, 32]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(_NoRedirect)


def http(method: str, url: str, data=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with OPENER.open(req) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def multipart(fields: dict[str, str], file_field: str, filename: str, content: bytes):
    boundary = "----cliplab-boundary-7q"
    lines = []
    for k, v in fields.items():
        lines.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    lines.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
    )
    lines.append(content)
    lines.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(lines), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def main() -> None:
    failures = []

    def check(cond, msg):
        print(("  ok  " if cond else " FAIL ") + msg)
        if not cond:
            failures.append(msg)

    with open(WAV_PATH, "rb") as f:
        wav_bytes = f.read()
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr = wf.getframerate()
        total = wf.getnframes()
        pcm_all = np.frombuffer(wf.readframes(total), dtype="<i2").copy()

    print("[1] 上传示例录音")
    body, hdrs = multipart({"a": A}, "wav", "interview_demo.wav", wav_bytes)
    status, _b, h = http("POST", f"{BASE}/upload", body, hdrs)
    check(status == 302, f"上传 302（实得 {status}）")
    pid = h["Location"].rstrip("/").split("/")[-1]
    print(f"      工程 id = {pid}")

    status, raw, _ = http("GET", f"{BASE}/api/p/{pid}/state")
    st = json.loads(raw)
    check(st["n_chunks"] == 40 and st["kept_samples"] == total,
          f"40 块、全保留 {total} 样点（实得 {st['n_chunks']}/{st['kept_samples']}）")

    print("[2] 撤下 7 个停顿块 + 恢复 1 块")
    max_touch = 0
    for i in SILENT:
        status, raw, _ = http("POST", f"{BASE}/api/p/{pid}/chunks/{i}/toggle",
                              json.dumps({"active": False}).encode(),
                              {"Content-Type": "application/json"})
        d = json.loads(raw)
        check(d["stats"]["chunk_loads"] == 0, f"撤块 #{i} 原始块读取 0")
        max_touch = max(max_touch, d["stats"]["summary_touches"])
    check(max_touch <= 24, f"单次撤块最多访问 {max_touch} 个摘要（log2(40)≈6，界 24）")
    status, raw, _ = http("POST", f"{BASE}/api/p/{pid}/chunks/16/toggle",
                          json.dumps({"active": True}).encode(),
                          {"Content-Type": "application/json"})
    d = json.loads(raw)
    check(d["stats"]["chunk_loads"] == 0, "恢复块 #16 原始块读取 0")
    off = [i for i in SILENT if i != 16]
    st = d["state"]
    check(st["kept_samples"] == total - 4096 * len(off),
          f"成片样点 = 总量 - {len(off)} 块")

    print("[3] 独立参考实现：先拼接、顺序滤波、再量化")
    kept = np.concatenate([
        pcm_all[b * 4096:(b + 1) * 4096] for b in range(40) if b not in off
    ]).astype(np.float64)
    a = float(A)
    y = np.empty_like(kept)
    s = 0.0
    for n in range(kept.shape[0]):
        s = a * s + (1.0 - a) * kept[n]
        y[n] = s
    ref = np.clip(np.rint(y), -32768, 32767).astype(np.int16)

    print("[4] 跳听：定位、长度、量化误差 ≤ 1、原始块只读 1 个")
    # 覆盖每个保留块块首 + 块中几个点 + 末样点
    ks = [0]
    cursor = 0
    block_starts = []
    for b in range(40):
        if b in off:
            continue
        block_starts.append((b, cursor))
        cursor += 4096
    ks += [v for _, v in block_starts]
    ks += [v + 123 for _, v in block_starts[:8]]
    ks += [ref.shape[0] - 1]
    ks = sorted(set(ks))
    worst = 0
    for k in ks:
        status, raw, _ = http("GET", f"{BASE}/api/p/{pid}/audition?k={k}")
        check(status == 200, f"k={k} 跳听 200")
        meta = json.loads(raw)
        check(meta["chunk_loads"] == 1, f"k={k} 只读 1 个原始块")
        check(meta["summary_touches"] <= 24, f"k={k} 摘要访问 {meta['summary_touches']} 对数级")
        status2, wb, _ = http("GET", f"{BASE}{meta['wav_url']}")
        check(status2 == 200 and wb[:4] == b"RIFF", f"k={k} WAV 下载成功")
        with wave.open(io.BytesIO(wb)) as wf2:
            nfr = wf2.getnframes()
            out = np.frombuffer(wf2.readframes(nfr), dtype="<i2")
        check(nfr == meta["out_samples"] == out.shape[0], f"k={k} WAV 帧数一致")
        seg = ref[k:k + nfr]
        diff = int(np.max(np.abs(out.astype(np.int32) - seg.astype(np.int32))))
        worst = max(worst, diff)
        if diff > 1:
            check(False, f"k={k} 量化差 {diff} > 1")
    check(worst <= 1, f"全部 {len(ks)} 个时码最大量化差 {worst} ≤ 1")
    print(f"      对照 {len(ks)} 个时码，逐样点最大差 = {worst}")

    print("[5] 定位语义抽查")
    # 成片 0 号 -> 块 0 偏移 0；跳过撤块后第一个接缝 -> 块 5 偏移 0
    status, raw, _ = http("GET", f"{BASE}/api/p/{pid}/audition?k=0")
    m0 = json.loads(raw)
    check((m0["chunk"], m0["offset"]) == (0, 0), "k=0 -> (块0, 偏移0)")
    # 块 0..3 保留(3 在 off!) 实际 off=[3,4,9,17,25,32]
    # 保留序列：0,1,2 | 5,6,7,8 | 10,... 块5 的成片起点 = 3*4096
    k5 = 3 * 4096
    status, raw, _ = http("GET", f"{BASE}/api/p/{pid}/audition?k={k5}")
    m5 = json.loads(raw)
    check((m5["chunk"], m5["offset"]) == (5, 0), f"k={k5} -> (块5, 偏移0)，实得 ({m5['chunk']},{m5['offset']})")

    print("[6] 越界提示")
    for bad in (ref.shape[0], ref.shape[0] + 999):
        status, raw, _ = http("GET", f"{BASE}/api/p/{pid}/audition?k={bad}")
        d = json.loads(raw)
        check(status == 400 and "重新选择" in d["error"], f"k={bad} -> 400 提示重选")

    print()
    if failures:
        print(f"{len(failures)} 项失败")
        sys.exit(1)
    print("HTTP 端到端全部通过")


if __name__ == "__main__":
    main()
