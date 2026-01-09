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
        # Current job status display
        with gr.Row():
            current_job_display = gr.Markdown("### Current Job: None", elem_id="current-job-status")
        
        with gr.Row():
            job_id_input = gr.Number(value=None, label="Job ID", precision=0)
            cancel_btn = gr.Button("Cancel")
            rerun_btn = gr.Button("♻️ Rerun")
            details_btn = gr.Button("Details")
            download_btn = gr.Button("Download Payload JSON")
        
        with gr.Row():
            show_completed = gr.Checkbox(label="Show Completed/Failed Jobs", value=True)
        
        status = gr.Markdown("")
        table = gr.Dataframe(headers=["id", "status", "created_at", "started_at", "finished_at", "result"], datatype=["number","str","str","str","str","str"], interactive=False)
        details_area = gr.Markdown("")
        download_file = gr.File(label="Downloaded Payload", visible=True)
        
        # Hidden timer for auto-refresh every 3 seconds
        timer = gr.Timer(value=3, active=True)

        def _get_current_job_status():
            """Fetch and format the current running job"""
            payload = _api_get("/api/queue", params={"status": "running"})
            if not payload or not payload.get("jobs"):
                return "### Current Job: None"
            
            jobs = payload.get("jobs", [])
            if not jobs:
                return "### Current Job: None"
            
            job = jobs[0]  # Get the first running job
            job_id = job.get("id")
            started_at = job.get("started_at")
            
            # Calculate elapsed time
            elapsed = ""
            if started_at:
                try:
                    elapsed_seconds = int(time.time() - float(started_at))
                    minutes = elapsed_seconds // 60
                    seconds = elapsed_seconds % 60
                    elapsed = f"{minutes}m {seconds}s"
                except Exception:
                    elapsed = "unknown"
            
            # Get job details
            try:
                job_payload = json.loads(job.get("payload", "{}"))
                job_type = job_payload.get("diffusion_task", "unknown").replace("_", " ").title()
                prompt = job_payload.get("prompt", "")
                # Truncate long prompts
                if len(prompt) > 100:
                    prompt = prompt[:100] + "..."
                
                status_text = (
                    f"### Current Job: #{job_id}\n"
                    f"**Type:** {job_type} | **Running for:** {elapsed}\n\n"
                    f"**Prompt:** {prompt}"
                )
                return status_text
            except Exception:
                return f"### Current Job: #{job_id} (Running for {elapsed})"

        def _refresh(show_completed_filter=True):
            payload = _api_get("/api/queue")
            if not payload:
                return [], "(failed to fetch queue)", _get_current_job_status()
            rows = []
            for j in payload.get("jobs", []):
                job_status = j.get("status")
                # Filter out completed/failed if checkbox is unchecked
                if not show_completed_filter and job_status in ("done", "failed", "cancelled"):
                    continue
                    
                rows.append([
                    j.get("id"),
                    job_status,
                    _fmt(j.get("created_at")),
                    _fmt(j.get("started_at")),
                    _fmt(j.get("finished_at")),
                    (j.get("result") or "")[:200],
                ])
            return rows, f"Loaded {len(rows)} jobs", _get_current_job_status()

        def _cancel(job_id):
            if not job_id:

        def _rerun(job_id):
            if not job_id:
                return "No job id provided"
            try:
                # Get the job payload
                job_payload = _api_get(f"/api/queue/{int(job_id)}")
                if not job_payload or not job_payload.get("job"):
                    return f"Job {job_id} not found"
                
                # Extract the original payload
                job = job_payload.get("job")
                payload_str = job.get("payload", "{}")
                payload = json.loads(payload_str)
                
                # Enqueue a new job with the same payload
                resp = _api_post("/api/queue", payload)
                if resp and resp.get("job_id"):
                    return f"Requeued as job {resp.get('job_id')}"
                return "Failed to requeue"
            except Exception as e:
                return f"Failed to rerun: {e}"
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
        rerun_btn.click(fn=_rerun, inputs=[job_id_input], outputs=[status])
        details_btn.click(fn=_details, inputs=[job_id_input], outputs=[status, details_area])
        download_btn.click(fn=_download_payload, inputs=[job_id_input], outputs=[status, download_file])
        
        # Auto-refresh on tab load
        queue_block.load(fn=_refresh, inputs=[show_completed], outputs=[table, status, current_job_display])
        
        # Auto-refresh every 3 seconds via timer (updates current job timer too)
        timer.tick(fn=_refresh, inputs=[show_completed], outputs=[table, status, current_job_display])
        
        # Refresh when filter checkbox changes
        show_completed.change(fn=_refresh, inputs=[show_completed]
                url = f"{API_BASE}/api/queue/{int(job_id)}/payload"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    content = resp.read()
                    # Save to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='wb', suffix=f'_job_{int(job_id)}_payload.json', delete=False) as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name
                return f"Downloaded payload for job {job_id}", tmp_path
            except Exception as e:
                return f"Failed to download payload: {e}", None

        cancel_btn.click(fn=_cancel, inputs=[job_id_input], outputs=[status])
        details_btn.click(fn=_details, inputs=[job_id_input], outputs=[status, details_area])
        download_btn.click(fn=_download_payload, inputs=[job_id_input], outputs=[status, download_file])
        
        # Auto-refresh on tab load
        queue_block.load(fn=_refresh, inputs=None, outputs=[table, status, current_job_display])
        
        # Auto-refresh every 3 seconds via timer (updates current job timer too)
        timer.tick(fn=_refresh, inputs=None, outputs=[table, status, current_job_display])
    
    return queue_block
