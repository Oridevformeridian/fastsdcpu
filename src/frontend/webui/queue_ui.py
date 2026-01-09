import os
import json
import time
import urllib.request
import urllib.parse
import gradio as gr
from state import get_settings
from frontend.webui.connection_manager import get_connection_state

API_BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")  # default to API server
conn_state = get_connection_state()


def _api_get(path: str, params: dict = None):
    if not API_BASE:
        return None
    url = API_BASE.rstrip("/") + path
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            result = json.load(resp)
            conn_state.mark_connected()
            return result
    except Exception:
        conn_state.mark_disconnected()
        return None


def _api_post(path: str, data: dict):
    url = API_BASE.rstrip("/") + path
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.load(resp)
            conn_state.mark_connected()
            return result
    except Exception:
        conn_state.mark_disconnected()
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
        
        # Action buttons and filters at the top
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    cancel_btn = gr.Button("🚫 Cancel", size="sm")
                    rerun_btn = gr.Button("♻️ Rerun", size="sm")
                    easyregen_btn = gr.Button("⚡ EasyRegen", size="sm")
                    details_btn = gr.Button("📋 Details", size="sm")
                    download_btn = gr.Button("💾 Download Payload", size="sm")
            with gr.Column(scale=1, min_width=120):
                job_id_input = gr.Number(value=None, label="Job ID", precision=0, scale=0, min_width=100)
        
        with gr.Row():
            show_completed = gr.Checkbox(label="Show Completed/Failed Jobs", value=True)
            gr.Markdown("💡 *Click the job ID to select*", elem_classes=["text-sm"])
        
        status = gr.Markdown("")
        
        # Queue table
        table = gr.Dataframe(
            headers=["id", "status", "created_at", "started_at", "finished_at", "result"], 
            datatype=["number", "str", "str", "str", "str", "str"], 
            interactive=False
        )
        
        details_area = gr.Markdown("")
        download_file = gr.File(label="Downloaded Payload", visible=True)
        
        # Hidden timer for auto-refresh every 3 seconds
        timer = gr.Timer(value=3, active=True)
        
        # Hidden component for triggering page reload on connection restore
        reload_trigger = gr.HTML(value="", visible=False)

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
            elapsed_seconds = 0
            if started_at:
                try:
                    elapsed_seconds = int(time.time() - float(started_at))
                    minutes = elapsed_seconds // 60
                    seconds = elapsed_seconds % 60
                    elapsed = f"{minutes}m {seconds}s"
                except Exception:
                    elapsed = "unknown"
            
            # Calculate average completion time from last 5 done jobs
            estimated_total = ""
            try:
                done_payload = _api_get("/api/queue", params={"status": "done"})
                if done_payload and done_payload.get("jobs"):
                    done_jobs = done_payload.get("jobs", [])[:5]  # Last 5 done jobs
                    durations = []
                    for dj in done_jobs:
                        start = dj.get("started_at")
                        finish = dj.get("finished_at")
                        if start and finish:
                            duration = float(finish) - float(start)
                            if duration > 0:
                                durations.append(duration)
                    
                    if durations:
                        avg_duration = int(sum(durations) / len(durations))
                        est_minutes = avg_duration // 60
                        est_seconds = avg_duration % 60
                        estimated_total = f" / {est_minutes}m {est_seconds}s"
            except Exception:
                pass  # If we can't calculate, just don't show estimate
            
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
                    f"**Type:** {job_type} | **Running for:** {elapsed}{estimated_total}\n\n"
                    f"**Prompt:** {prompt}"
                )
                return status_text
            except Exception:
                return f"### Current Job: #{job_id} (Running for {elapsed}{estimated_total})"

        def _refresh(show_completed_filter=True):
            # Check if we were disconnected before making the API call
            was_disconnected = not conn_state.is_connected and conn_state.error_shown
            
            payload = _api_get("/api/queue")
            reload_html = ""
            
            # If we just reconnected, trigger page reload after 2 seconds
            if was_disconnected and conn_state.is_connected and payload is not None:
                reload_html = '<script>setTimeout(function() { window.location.reload(); }, 2000);</script>'
            
            if not payload:
                return [], "(failed to fetch queue)", _get_current_job_status(), reload_html
            rows = []
            for j in payload.get("jobs", []):
                job_status = j.get("status")
                # Filter out completed/failed if checkbox is unchecked
                if not show_completed_filter and job_status in ("done", "failed", "cancelled"):
                    continue
                
                # Show retry count for queued jobs that have been retried
                retry_count = j.get("retry_count", 0)
                if job_status == "queued" and retry_count > 0:
                    status_display = f"rerunning({retry_count})"
                else:
                    status_display = job_status
                    
                rows.append([
                    j.get("id"),
                    status_display,
                    _fmt(j.get("created_at")),
                    _fmt(j.get("started_at")),
                    _fmt(j.get("finished_at")),
                    (j.get("result") or "")[:200],
                ])
            return rows, f"Loaded {len(rows)} jobs", _get_current_job_status(), reload_html

        def _cancel(job_id):
            if not job_id:
                return "No job id provided"
            resp = _api_post(f"/api/queue/{int(job_id)}/cancel", {})
            if not resp:
                return "Cancel failed"
            return f"Cancelled {job_id}"

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

        def _easyregen(job_id):
            if not job_id:
                return "No job id provided"
            try:
                # Get the job payload
                job_payload = _api_get(f"/api/queue/{int(job_id)}")
                if not job_payload or not job_payload.get("job"):
                    return f"Job {job_id} not found"
                
                # Extract and modify the payload
                job = job_payload.get("job")
                payload_str = job.get("payload", "{}")
                payload = json.loads(payload_str)
                
                # Modify for easy/fast regeneration
                payload["image_width"] = 512
                payload["image_height"] = 512
                payload["inference_steps"] = 8
                
                # Enqueue the modified job
                resp = _api_post("/api/queue", payload)
                if resp and resp.get("job_id"):
                    return f"EasyRegen queued as job {resp.get('job_id')} (512x512, 8 steps)"
                return "Failed to queue easyregen"
            except Exception as e:
                return f"Failed to easyregen: {e}"


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
                return "No job id provided", None
            try:
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

        def _on_row_select(evt: gr.SelectData):
            """When user clicks a row, populate the Job ID input"""
            try:
                # evt.index is (row, col) for Dataframe
                if evt.index is None:
                    return None, ""
                
                # evt.value contains the cell value that was clicked
                # evt.index[0] is the row number
                # For our table, we need to get the job ID from the first column
                # But evt.value gives us the clicked cell's value
                # We need to parse the row data differently
                
                # The cleanest approach: evt.value from first column IS the job_id
                # So we check if user clicked column 0 (id column)
                row_idx, col_idx = evt.index if isinstance(evt.index, tuple) else (evt.index, 0)
                
                # If they clicked the ID column (column 0), use that value directly
                if col_idx == 0 and evt.value is not None:
                    try:
                        job_id = int(evt.value)
                        return job_id, f"✓ Selected job #{job_id}"
                    except (ValueError, TypeError):
                        pass
                
                # For other columns, we can't easily get the row data without accessing the table state
                # So just show a message
                return None, "💡 Click the ID column (first column) to select a job"
                
            except Exception as e:
                return None, f"Selection error: {e}"

        cancel_btn.click(fn=_cancel, inputs=[job_id_input], outputs=[status])
        rerun_btn.click(fn=_rerun, inputs=[job_id_input], outputs=[status])
        easyregen_btn.click(fn=_easyregen, inputs=[job_id_input], outputs=[status])
        details_btn.click(fn=_details, inputs=[job_id_input], outputs=[status, details_area])
        download_btn.click(fn=_download_payload, inputs=[job_id_input], outputs=[status, download_file])
        
        # Row selection - click the ID column to populate Job ID
        table.select(fn=_on_row_select, outputs=[job_id_input, status])
        
        # Auto-refresh on tab load
        queue_block.load(fn=_refresh, inputs=[show_completed], outputs=[table, status, current_job_display, reload_trigger])
        
        # Auto-refresh every 3 seconds via timer (updates current job timer too)
        timer.tick(fn=_refresh, inputs=[show_completed], outputs=[table, status, current_job_display, reload_trigger])
        
        # Refresh when filter checkbox changes
        show_completed.change(fn=_refresh, inputs=[show_completed], outputs=[table, status, current_job_display, reload_trigger])
    
    return queue_block
