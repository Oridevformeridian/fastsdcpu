import os
import json
import time
import urllib.request
import urllib.parse
import gradio as gr
from state import get_settings

API_BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")  # default to API server


def _api_get(path: str, params: dict = None):
    if not API_BASE:
        return None
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
    with gr.Blocks() as queue_block:
        with gr.Row():
            job_id_input = gr.Number(value=None, label="Job ID", precision=0)
            cancel_btn = gr.Button("Cancel")
            details_btn = gr.Button("Details")
            download_btn = gr.Button("Download Payload JSON")
        status = gr.Markdown("")
        table = gr.Dataframe(headers=["id", "status", "created_at", "started_at", "finished_at", "result"], datatype=["number","str","str","str","str","str"], interactive=False)
        details_area = gr.Markdown("")
        download_link = gr.Markdown("")
        
        # Hidden timer for auto-refresh every 3 seconds
        timer = gr.Timer(value=3, active=True)

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


        def _details(job_id):
            if not job_id:
                return "No job id provided", ""
            payload = _api_get(f"/api/queue/{int(job_id)}")
            if not payload or not payload.get("job"):
                return f"Job {job_id} not found", ""
            j = payload.get("job")
            text = (
                f"**Job {j.get('id')}**  \n"
                f"- status: {j.get('status')}  \n"
                f"- created: {_fmt(j.get('created_at'))}  \n"
                f"- started: {_fmt(j.get('started_at'))}  \n"
                f"- finished: {_fmt(j.get('finished_at'))}  \n"
                f"- result: {j.get('result') or ''}  \n"
            )
            return f"Loaded job {job_id}", text

        def _download_payload(job_id):
            if not job_id:
                return "No job id provided", ""
            url = f"{API_BASE}/api/queue/{int(job_id)}/payload"
            link = f"[Download Payload JSON for Job {job_id}]({url})"
            return f"Download link generated", link

        cancel_btn.click(fn=_cancel, inputs=[job_id_input], outputs=[status])
        details_btn.click(fn=_details, inputs=[job_id_input], outputs=[status, details_area])
        download_btn.click(fn=_download_payload, inputs=[job_id_input], outputs=[status, download_link])
        
        # Auto-refresh on tab load
        queue_block.load(fn=_refresh, inputs=None, outputs=[table, status])
        
        # Auto-refresh every 3 seconds via timer
        timer.tick(fn=_refresh, inputs=None, outputs=[table, status])
    
    return queue_block
