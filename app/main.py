"""Flask 网页：上传、分块撤/恢复、成片时码跳听。"""
from __future__ import annotations

import io
import os
import threading
import uuid

from flask import (
    Flask, abort, jsonify, redirect, render_template, request, send_file, url_for,
)

from .engine import CHUNK, Project, TimecodeOutOfRange
from .wavio import WavFormatError, build_mono_pcm16_wav, read_mono_pcm16_wav

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024  # 128 MiB

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


def _audition_result(project: Project, k: int) -> tuple[dict, bytes]:
    project.reset_counters()
    chunk, offset, pcm16 = project.audition(k)
    wav = build_mono_pcm16_wav(pcm16.tobytes(), project.sample_rate)
    meta = {
        "k": k,
        "chunk": chunk,
        "offset": offset,
        "original_sample": chunk * CHUNK + offset,
        "out_samples": int(pcm16.shape[0]),
        "duration_ms": round(pcm16.shape[0] * 1000.0 / project.sample_rate, 2),
        "summary_touches": project.fenwick.touches + project.segtree.touches,
        "chunk_loads": project.chunk_loads,
        "wav_url": url_for("api_audition_wav", pid=project.id, k=k),
    }
    return meta, wav


@app.get("/api/p/<pid>/audition")
def api_audition(pid: str):
    project = get_project(pid)
    raw = request.args.get("k", "")
    try:
        k = int(raw)
    except (TypeError, ValueError):
        return jsonify({"error": "成片样点号必须是整数，请重新选择"}), 400
    with get_lock(pid):
        try:
            meta, _wav = _audition_result(project, k)
        except TimecodeOutOfRange as exc:
            return jsonify({"error": f"{exc}，请重新选择"}), 400
        return jsonify(meta)


@app.get("/api/p/<pid>/audition.wav")
def api_audition_wav(pid: str):
    project = get_project(pid)
    try:
        k = int(request.args.get("k", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "成片样点号必须是整数，请重新选择"}), 400
    with get_lock(pid):
        try:
            _meta, wav = _audition_result(project, k)
        except TimecodeOutOfRange as exc:
            return jsonify({"error": f"{exc}，请重新选择"}), 400
        return send_file(
            io.BytesIO(wav),
            mimetype="audio/wav",
            as_attachment=False,
            download_name=f"audition_k{k}.wav",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
