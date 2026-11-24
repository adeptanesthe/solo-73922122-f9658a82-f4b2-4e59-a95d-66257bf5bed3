"""Flask 网页：上传、分块撤/恢复、成片时码跳听。

跳听的“定位”与“取声”解耦
==========================
定位（``/audition``）只查询两棵摘要树，冻结定位当时的
:class:`~app.engine.AuditionSnapshot`（块号、偏移、入块滤波状态），并把快照
签进一个**无状态**令牌随 URL 下发；服务端不保存任何已发链接的音频或逐链接
数据，因此留存数据量不随已发链接数增长。之后无论另一窗口如何撤块/恢复，
``/audition.signed.wav`` 都凭令牌 + 不可变的原始 PCM 重建定位当时的剪法声音，
且只读取命中的那一个原始块。

只传样点号的旧入口 ``/audition.wav?k=...`` 保持原语义：每次都对**当前成片**
实时定位并试听。
"""
from __future__ import annotations

import io
import os
import secrets
import threading
import uuid

from flask import (
    Flask, abort, jsonify, redirect, render_template, request, send_file, url_for,
)
from itsdangerous import BadSignature, URLSafeSerializer

from .engine import CHUNK, AuditionSnapshot, Project, TimecodeOutOfRange
from .wavio import WavFormatError, build_mono_pcm16_wav, read_mono_pcm16_wav

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024  # 128 MiB

# 快照令牌只在本进程内有效；服务端不留存任何已发链接的数据
SNAPSHOT_SIGNER = URLSafeSerializer(secrets.token_hex(32), salt="audition-snapshot")

DATA_DIR = os.environ.get("CLIP_DATA_DIR", "/tmp/clip-lab-data")
os.makedirs(DATA_DIR, exist_ok=True)

PROJECTS: dict[str, Project] = {}
LOCKS: dict[str, threading.Lock] = {}


def get_project(pid: str) -> Project:
    project = PROJECTS.get(pid)
    if project is None:
        abort(404, description="工程不存在或已过期，请重新上传录音")
    return project


def get_lock(pid: str) -> threading.Lock:
    if pid not in LOCKS:
        LOCKS[pid] = threading.Lock()
    return LOCKS[pid]


def project_state(project: Project) -> dict:
    return {
        "id": project.id,
        "sample_rate": project.sample_rate,
        "total_samples": project.total_samples,
        "kept_samples": project.kept_samples(),
        "chunk_size": CHUNK,
        "n_chunks": project.n,
        "a": project.a,
        "chunks": [
            {
                "index": c.index,
                "samples": c.samples,
                "active": c.active,
                "start_sample": c.start_sample,
            }
            for c in project.list_chunks()
        ],
    }


@app.get("/")
def index():
    return render_template("index.html", projects=list(PROJECTS.values()))


@app.post("/upload")
def upload():
    f = request.files.get("wav")
    if f is None or not f.filename:
        return render_template("error.html", message="请选择要上传的 WAV 文件"), 400
    try:
        a = float(request.form.get("a", ""))
    except (TypeError, ValueError):
        return render_template("error.html", message="系数 a 必须是数字"), 400
    if not (0.0 <= a < 1.0):
        return render_template("error.html", message="系数 a 必须满足 0 ≤ a < 1"), 400

    data = f.read()
    try:
        pcm, sample_rate, n_frames = read_mono_pcm16_wav(data)
    except WavFormatError as exc:
        return render_template("error.html", message=str(exc)), 400

    pid = uuid.uuid4().hex[:12]
    pdir = os.path.join(DATA_DIR, pid)
    os.makedirs(pdir, exist_ok=True)
    pcm_path = os.path.join(pdir, "input.pcm")
    with open(pcm_path, "wb") as out:
        out.write(pcm)

    project = Project(pid, pcm_path, sample_rate, n_frames, a)
    PROJECTS[pid] = project
    LOCKS[pid] = threading.Lock()
    return redirect(url_for("project_page", pid=pid))


@app.get("/p/<pid>")
def project_page(pid: str):
    project = get_project(pid)
    return render_template("project.html", state=project_state(project))


@app.get("/api/p/<pid>/state")
def api_state(pid: str):
    project = get_project(pid)
    with get_lock(pid):
        return jsonify(project_state(project))


@app.post("/api/p/<pid>/chunks/<int:i>/toggle")
def api_toggle(pid: str, i: int):
    project = get_project(pid)
    payload = request.get_json(silent=True) or {}
    active = bool(payload.get("active"))
    with get_lock(pid):
        if not (0 <= i < project.n):
            return jsonify({"error": f"块号 {i} 越界，共 {project.n} 块"}), 400
        project.reset_counters()
        project.set_chunk_active(i, active)
        stats = {
            "summary_touches": project.fenwick.touches + project.segtree.touches,
            "chunk_loads": project.chunk_loads,
        }
        return jsonify({"state": project_state(project), "stats": stats})


def _dump_snapshot(project: Project, snap: AuditionSnapshot) -> str:
    """把快照连同工程 id 签成无状态令牌（不含任何音频字节）。"""
    return SNAPSHOT_SIGNER.dumps({
        "p": project.id,
        "c": snap.chunk,
        "o": snap.offset,
        "s": snap.state_in,
        "k": snap.k,
        "kept": snap.kept_at_locate,
    })


def _load_snapshot(project: Project, token: str) -> AuditionSnapshot:
    """验签并核对令牌属于当前工程；工程在则链接一直可重放。"""
    try:
        data = SNAPSHOT_SIGNER.loads(token)
    except BadSignature as exc:
        abort(404, description="试听链接无效或已损坏")
    if not isinstance(data, dict) or data.get("p") != project.id:
        abort(404, description="试听链接不属于该工程")
    try:
        return AuditionSnapshot(
            chunk=int(data["c"]),
            offset=int(data["o"]),
            state_in=float(data["s"]),
            k=int(data["k"]),
            kept_at_locate=int(data["kept"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        abort(404, description="试听链接内容不完整")


@app.get("/api/p/<pid>/audition")
def api_audition(pid: str):
    """定位：只访问摘要（原始块读取 0 次），返回冻结剪法的试听链接。"""
    project = get_project(pid)
    raw = request.args.get("k", "")
    try:
        k = int(raw)
    except (TypeError, ValueError):
        return jsonify({"error": "成片样点号必须是整数，请重新选择"}), 400
    with get_lock(pid):
        try:
            project.reset_counters()
            snap = project.locate_snapshot(k)
        except TimecodeOutOfRange as exc:
            return jsonify({"error": f"{exc}，请重新选择"}), 400
        token = _dump_snapshot(project, snap)
        out_samples = project.chunk_counts[snap.chunk] - snap.offset
        return jsonify({
            "k": snap.k,
            "chunk": snap.chunk,
            "offset": snap.offset,
            "original_sample": snap.original_sample,
            "out_samples": int(out_samples),
            "duration_ms": round(out_samples * 1000.0 / project.sample_rate, 2),
            "kept_at_locate": snap.kept_at_locate,
            "frozen": True,
            "summary_touches": project.fenwick.touches + project.segtree.touches,
            "chunk_loads": project.chunk_loads,
            "wav_url": url_for("api_audition_signed_wav", pid=project.id, token=token),
        })


@app.get("/api/p/<pid>/audition.signed.wav/<token>")
def api_audition_signed_wav(pid: str, token: str):
    """凭定位时签发的快照令牌重放：只读命中块，撤块/恢复后声音不变。"""
    project = get_project(pid)
    with get_lock(pid):
        snap = _load_snapshot(project, token)
        project.reset_counters()
        _chunk, _offset, pcm16 = project.render_snapshot(snap)
        wav = build_mono_pcm16_wav(pcm16.tobytes(), project.sample_rate)
        resp = send_file(
            io.BytesIO(wav),
            mimetype="audio/wav",
            as_attachment=False,
            download_name=f"audition_k{snap.k}.wav",
        )
        # 供演示/核对：重放一次只读取 1 个命中原始块
        resp.headers["X-Chunk-Loads"] = str(project.chunk_loads)
        resp.headers["X-Summary-Touches"] = str(
            project.fenwick.touches + project.segtree.touches
        )
        return resp


@app.get("/api/p/<pid>/audition.wav")
def api_audition_wav(pid: str):
    """旧入口（只传样点号）：每次都对当前成片实时定位并试听。"""
    project = get_project(pid)
    try:
        k = int(request.args.get("k", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "成片样点号必须是整数，请重新选择"}), 400
    with get_lock(pid):
        try:
            project.reset_counters()
            _chunk, _offset, pcm16 = project.audition(k)
        except TimecodeOutOfRange as exc:
            return jsonify({"error": f"{exc}，请重新选择"}), 400
        wav = build_mono_pcm16_wav(pcm16.tobytes(), project.sample_rate)
        return send_file(
            io.BytesIO(wav),
            mimetype="audio/wav",
            as_attachment=False,
            download_name=f"audition_k{k}.wav",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
