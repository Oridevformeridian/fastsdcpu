import os
import json
import time
import urllib.request
import urllib.parse
import gradio as gr
from state import get_settings

API_BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")


def _api_get(path: str, params: dict = None):
    url = API_BASE.rstrip("/") + path
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.load(resp)
    except Exception:
        return None


def _api_post(path: str, data: dict):
    url = API_BASE.rstrip("/") + path
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.load(resp)
    except Exception:
        return None


app_settings = get_settings()


def _fmt(ts):
    if not ts:
        return ""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return str(ts)


def get_queue_ui():
    with gr.Blocks():
        with gr.Row():
            refresh = gr.Button("Refresh")
            cancel_id = gr.Number(value=None, label="Job ID to cancel", precision=0)
            cancel_btn = gr.Button("Cancel Job")
        status = gr.Markdown("")
        table = gr.Dataframe(headers=["id", "status", "created_at", "started_at", "finished_at", "result"], datatype=["number","str","str","str","str","str"], interactive=False)

        def _refresh():
            payload = _api_get("/api/queue")
            if not payload:
                return [], "(failed to fetch queue)"
            rows = []
            for j in payload.get("jobs", []):
                rows.append([
                    j.get("id"),
                    j.get("status"),
                    _fmt(j.get("created_at")),
                    _fmt(j.get("started_at")),
                    _fmt(j.get("finished_at")),
                    (j.get("result") or "")[:200],
                ])
            return rows, f"Loaded {len(rows)} jobs"

        def _cancel(job_id):
            if not job_id:
                return "No job id provided"
            resp = _api_post(f"/api/queue/{int(job_id)}/cancel", {})
            if not resp:
                return "Cancel failed"
            return f"Cancelled {job_id}"

        refresh.click(fn=_refresh, inputs=None, outputs=[table, status])
        cancel_btn.click(fn=_cancel, inputs=[cancel_id], outputs=[status])
        # load initially
        refresh.click(fn=_refresh, inputs=None, outputs=[table, status])
