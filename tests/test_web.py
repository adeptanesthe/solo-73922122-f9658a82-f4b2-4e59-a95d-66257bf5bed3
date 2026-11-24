"""Flask 路由层集成测试：上传 -> 撤/恢复 -> 跳听 -> 越界提示。

运行：PYTHONPATH=. python tests/test_web.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app  # noqa: E402
from app.wavio import build_mono_pcm16_wav  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok  " if cond else " FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def wav_bytes(pcm16: np.ndarray, sr: int = 16000) -> bytes:
    return build_mono_pcm16_wav(pcm16.astype("<i2").tobytes(), sr)


def main() -> None:
    app.config["CLIP_DATA_DIR"] = tempfile.mkdtemp(prefix="cliplab-web-")
    # main.py 在 import 时已建默认目录，这里把工程落盘目录指到临时目录
    import app.main as m
    m.DATA_DIR = app.config["CLIP_DATA_DIR"]
    client = app.test_client()

    print("[1] 非法上传被拒绝")
    r = client.post("/upload", data={"a": "0.8", "wav": (io.BytesIO(b"not wav"), "x.wav")},
                    content_type="multipart/form-data")
    check(r.status_code == 400, f"垃圾文件 -> 400（实得 {r.status_code}）")
    stereo = wav_bytes(np.zeros(10, dtype="<i2"))
    # 伪造成双声道：直接手写一个 2 声道 WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(np.zeros(20, dtype="<i2").tobytes())
    r = client.post("/upload", data={"a": "0.8", "wav": (io.BytesIO(buf.getvalue()), "s.wav")},
                    content_type="multipart/form-data")
    check(r.status_code == 400, f"立体声 -> 400（实得 {r.status_code}）")
    r = client.post("/upload", data={"a": "1.0", "wav": (io.BytesIO(stereo), "x.wav")},
                    content_type="multipart/form-data")
    check(r.status_code == 400, f"a=1 越界 -> 400（实得 {r.status_code}）")
    r = client.post("/upload", data={"a": "abc", "wav": (io.BytesIO(stereo), "x.wav")},
                    content_type="multipart/form-data")
    check(r.status_code == 400, f"a 非数字 -> 400（实得 {r.status_code}）")

    print("[2] 合法上传并分块")
    rng = np.random.default_rng(11)
    pcm = rng.integers(-20000, 20000, size=3 * 4096 + 7).astype(np.int16)
    r = client.post("/upload", data={"a": "0.85", "wav": (io.BytesIO(wav_bytes(pcm)), "d.wav")},
                    content_type="multipart/form-data", follow_redirects=False)
    check(r.status_code == 302, f"上传成功 302（实得 {r.status_code}）")
    pid = r.headers["Location"].rstrip("/").split("/")[-1]
    r = client.get(f"/api/p/{pid}/state")
    st = r.get_json()
    check(st["n_chunks"] == 4 and st["kept_samples"] == pcm.size, "4 块、保留全部样点")
    check(st["chunks"][-1]["samples"] == 7, "末块 7 个样点")

    print("[3] 撤下/恢复只访问摘要")
    r = client.post(f"/api/p/{pid}/chunks/1/toggle", json={"active": False})
    d = r.get_json()
    check(r.status_code == 200, "撤下 200")
    check(d["stats"]["chunk_loads"] == 0, f"撤块原始块读取 {d['stats']['chunk_loads']} == 0")
    check(d["stats"]["summary_touches"] <= 40, f"摘要访问 {d['stats']['summary_touches']} 为对数级")
    check(d["state"]["kept_samples"] == pcm.size - 4096, "成片样点数同步减少")
    r = client.post(f"/api/p/{pid}/chunks/1/toggle", json={"active": True})
    check(r.get_json()["state"]["kept_samples"] == pcm.size, "恢复后样点数复原")

    print("[4] 跳听定位与 WAV 内容（定位零读块，签名链接取声读 1 块）")
    client.post(f"/api/p/{pid}/chunks/1/toggle", json={"active": False})
    # 成片 4096 号样点 = 原块 #2 的块首
    r = client.get(f"/api/p/{pid}/audition?k=4096")
    meta = r.get_json()
    check(r.status_code == 200, "跳听 200")
    check((meta["chunk"], meta["offset"]) == (2, 0), f"k=4096 -> 块#2 偏移0（实得 {meta['chunk']},{meta['offset']}）")
    check(meta["chunk_loads"] == 0, f"定位不读原始块（实得 {meta['chunk_loads']}）")
    check(meta["frozen"] is True and meta["kept_at_locate"] == pcm.size - 4096,
          "定位返回冻结标记与定位当时成片长度")
    check(meta["out_samples"] == 4096, "试听长度到块末 = 4096")
    check("/audition.signed.wav/" in meta["wav_url"], "试听链接为签名快照链接")
    r = client.get(meta["wav_url"])
    check(r.status_code == 200 and r.data[:4] == b"RIFF", "签名 WAV 可下载且带 RIFF 头")
    check(r.headers.get("X-Chunk-Loads") == "1", "取声响应头标注只读 1 个原始块")
    with wave.open(io.BytesIO(r.data)) as wf:
        check(wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == 16000,
              "试听 WAV 为单声道 PCM16 同采样率")
        check(wf.getnframes() == 4096, "试听 WAV 帧数 4096")
    first_wav = r.data

    print("[4b] 两窗改剪：撤块/恢复后同一签名链接首次加载与重播字节不变")
    # 撤下定位命中的块 #2、恢复块 #1，再恢复 #2 —— 模拟另一窗口连续改剪
    client.post(f"/api/p/{pid}/chunks/2/toggle", json={"active": False})
    client.post(f"/api/p/{pid}/chunks/1/toggle", json={"active": True})
    r2 = client.get(meta["wav_url"])
    check(r2.status_code == 200 and r2.data == first_wav, "改剪后同一链接重放结果不变")
    client.post(f"/api/p/{pid}/chunks/2/toggle", json={"active": True})
    r3 = client.get(meta["wav_url"])
    check(r3.status_code == 200 and r3.data == first_wav, "恢复后同一链接仍可重播且不变")
    # 页面展示的原块/偏移仍是定位当时的值
    check((meta["chunk"], meta["offset"]) == (2, 0), "冻结的原块/偏移不漂移")

    print("[4c] 旧入口 audition.wav?k= 仍试听当前成片")
    # 当前剪法已全部恢复，k=4096 落回原块 #1 块首；与同剪法下新定位结果一致
    r = client.get(f"/api/p/{pid}/audition.wav?k=4096")
    check(r.status_code == 200 and r.data[:4] == b"RIFF", "旧入口仍可用")
    r_now = client.get(f"/api/p/{pid}/audition?k=4096")
    check(r_now.get_json()["chunk"] == 1, "当前成片 k=4096 定位到块 #1")
    r_now_wav = client.get(r_now.get_json()["wav_url"])
    check(r.data == r_now_wav.data, "旧入口 WAV 与当前剪法的新定位 WAV 相同")

    print("[4d] 伪造/串工程链接被拒")
    check(client.get(meta["wav_url"] + "x").status_code == 404, "损坏令牌 -> 404")
    r_other = client.post("/upload", data={"a": "0.85", "wav": (io.BytesIO(wav_bytes(pcm)), "o.wav")},
                          content_type="multipart/form-data", follow_redirects=False)
    other_pid = r_other.headers["Location"].rstrip("/").split("/")[-1]
    foreign_url = meta["wav_url"].replace(pid, other_pid)
    check(client.get(foreign_url).status_code == 404, "令牌串到别的工程 -> 404")

    print("[5] 越界时码提示重选")
    kept_now = client.get(f"/api/p/{pid}/state").get_json()["kept_samples"]
    for bad in (kept_now, kept_now + 10, -5):
        r = client.get(f"/api/p/{pid}/audition?k={bad}")
        check(r.status_code == 400 and "重新选择" in r.get_json()["error"],
              f"k={bad} -> 400 且提示重选")
    r = client.get(f"/api/p/{pid}/audition?k=abc")
    check(r.status_code == 400, "非整数时码 -> 400")

    print("[6] 未知工程 404")
    check(client.get("/api/p/nope/state").status_code == 404, "未知工程 404")

    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败")
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
