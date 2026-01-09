import platform
import os
import sqlite3
import logging
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Debug logging control
DEBUG_ENABLED = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")

from backend.api.models.response import StableDiffusionResponse
from backend.base64_image import base64_image_to_pil, pil_image_to_base64_str
from backend.device import get_device_name
from backend.models.device import DeviceInfo
from backend.models.lcmdiffusion_setting import DiffusionTask, LCMDiffusionSetting
from constants import APP_VERSION, DEVICE
from context import Context
from models.interface_types import InterfaceType
from paths import FastStableDiffusionPaths
from state import get_settings
from backend.queue_db import (
    init_db as init_queue_db,
    enqueue_job,
    get_job,
    list_jobs as list_queue_jobs,
    pop_next_job,
    complete_job,
    fail_job,
    cancel_job,
    reset_orphaned_jobs,
    update_job_progress,
)
import threading
import time
import json
import json

app_settings = get_settings()
app = FastAPI(
    title="FastSD CPU",
    description="Fast stable diffusion on CPU",
    version=APP_VERSION,
    license_info={
        "name": "MIT",
        "identifier": "MIT",
    },
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
print(app_settings.settings.lcm_diffusion_setting)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
context = Context(InterfaceType.API_SERVER)


@app.get("/api/")
async def root():
    return {"message": "Welcome to FastSD CPU API"}


@app.get(
    "/api/info",
    description="Get system information",
    summary="Get system information",
)
async def info():
    device_info = DeviceInfo(
        device_type=DEVICE,
        device_name=get_device_name(),
        os=platform.system(),
        platform=platform.platform(),
        processor=platform.processor(),
    )
    return device_info.model_dump()


@app.get(
    "/api/config",
    description="Get current configuration",
    summary="Get configurations",
)
async def config():
    return app_settings.settings


@app.get(
    "/api/models",
    description="Get available models",
    summary="Get available models",
)
async def models():
    return {
        "lcm_lora_models": app_settings.lcm_lora_models,
        "stable_diffusion": app_settings.stable_diffsuion_models,
        "openvino_models": app_settings.openvino_lcm_models,
        "lcm_models": app_settings.lcm_models,
    }


@app.post(
    "/api/generate",
    description="Generate image(Text to image,Image to Image)",
    summary="Generate image(Text to image,Image to Image)",
)
async def generate(diffusion_config: LCMDiffusionSetting) -> StableDiffusionResponse:
    app_settings.settings.lcm_diffusion_setting = diffusion_config
    if diffusion_config.diffusion_task == DiffusionTask.image_to_image:
        app_settings.settings.lcm_diffusion_setting.init_image = base64_image_to_pil(
            diffusion_config.init_image
        )

    images = context.generate_text_to_image(app_settings.settings)

    if images:
        images_base64 = [pil_image_to_base64_str(img) for img in images]
    else:
        images_base64 = []
    return StableDiffusionResponse(
        latency=round(context.latency, 2),
        images=images_base64,
        error=context.error,
    )


def start_web_server(port: int = 8000):
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )


@app.get(
    "/api/results/paged",
    description="List generated result files (paginated)",
    summary="List generated results (paged)",
)
async def list_results_paged(page: int = 0, size: int = 20):
    """Return paginated results. Uses a small in-memory cache with TTL and directory-mtime invalidation."""
    try:
        path = app_settings.settings.generated_images.path
        if not path:
            path = FastStableDiffusionPaths.get_results_path()

        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Results directory not found")

        # init cache on app
        if not hasattr(app, "_results_cache"):
            app._results_cache = {"dir_mtime": 0, "timestamp": 0.0, "ttl": 3.0, "pages": {}}

        cache = app._results_cache
        dir_mtime = os.stat(path).st_mtime

        # invalidate cache if directory changed
        if cache.get("dir_mtime") != dir_mtime:
            cache["dir_mtime"] = dir_mtime
            cache["pages"].clear()

        key = f"{page}:{size}"
        now = time.time()
        entry = cache["pages"].get(key)
        if entry and (now - entry["timestamp"] < cache["ttl"]):
            return entry["data"]

        # Only list image files (jpg, png)
        all_entries = [
            e for e in os.listdir(path) 
            if os.path.isfile(os.path.join(path, e)) 
            and (e.lower().endswith('.jpg') or e.lower().endswith('.png') or e.lower().endswith('.jpeg'))
        ]
        all_entries.sort(key=lambda e: os.stat(os.path.join(path, e)).st_mtime, reverse=True)

        start = page * size
        end = start + size
        page_entries = all_entries[start:end]

        results = []
        for entry_name in page_entries:
            try:
                full = os.path.join(path, entry_name)
                stat = os.stat(full)
                
                # Extract UUID from filename and look for corresponding JSON
                import re
                base_name = os.path.splitext(entry_name)[0]
                # Match UUID pattern in filename (with or without batch suffix)
                uuid_match = re.match(r'^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', base_name)
                meta = {}
                if uuid_match:
                    uuid = uuid_match.group(1)
                    json_path = os.path.join(path, uuid + ".json")
                    if os.path.exists(json_path):
                        # Retry logic to handle race conditions with file writes
                        for attempt in range(3):
                            try:
                                with open(json_path, "r", encoding="utf-8") as f:
                                    meta = json.load(f)
                                if DEBUG_ENABLED:
                                    print(f"[DEBUG] Loaded JSON for {entry_name}: UUID={uuid}, keys={list(meta.keys())}, prompt={meta.get('prompt', 'N/A')[:50]}")
                                break
                            except (json.JSONDecodeError, IOError) as e:
                                if attempt < 2:
                                    time.sleep(0.05)  # Wait 50ms before retry
                                else:
                                    if DEBUG_ENABLED:
                                        print(f"[DEBUG] Failed to load JSON for {entry_name} after 3 attempts: {e}")
                                    logging.warning(f"Failed to load JSON for {entry_name}: {e}")
                    else:
                        if DEBUG_ENABLED:
                            print(f"[DEBUG] JSON not found for {entry_name}: {json_path}")
                else:
                    if DEBUG_ENABLED:
                        print(f"[DEBUG] No UUID match for {entry_name}")

                results.append(
                    {
                        "name": entry_name,
                        "url": f"/results/{entry_name}",
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "meta": meta,
                        "review": None,
                    }
                )
            except Exception as e:
                # Don't let a single bad entry crash the entire results page
                logging.error(f"Error processing result entry {entry_name}: {e}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                # Continue to next entry without adding this one
                continue

        payload = {"page": page, "size": size, "results": results, "total": len(all_entries)}
        cache["pages"][key] = {"timestamp": now, "data": payload}
        return payload
    except Exception as e:
        logging.exception(f"Error in list_results_paged endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list results: {str(e)}")


@app.post(
    "/api/queue",
    description="Enqueue a generation task",
    summary="Enqueue generation",
)
async def enqueue_api(diffusion_config: LCMDiffusionSetting):
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()
    db_file = os.path.join(path, "queue.db")
    init_queue_db(db_file)
    try:
        payload = diffusion_config.model_dump()
    except Exception:
        try:
            payload = diffusion_config.dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid diffusion config: {e}")
    
    # Save payload JSON to disk immediately when enqueued
    queue_json_dir = os.path.join(path, "queue_payloads")
    os.makedirs(queue_json_dir, exist_ok=True)
    # Use timestamp for temporary filename until we get job_id
    temp_json_path = os.path.join(queue_json_dir, f"temp_{time.time()}.json")
    
    try:
        job_id = enqueue_job(db_file, payload, None)  # First create job to get ID
        # Rename to use actual job_id
        json_filename = f"job_{job_id}_payload.json"
        json_path = os.path.join(queue_json_dir, json_filename)
        
        # Write the JSON file
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        
        # Update the job with the json path
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("UPDATE queue SET payload_json_path = ? WHERE id = ?", (json_path, job_id))
        conn.commit()
        conn.close()
        
        print(f"Enqueued job {job_id} via api/web.py, saved payload to {json_path}")
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)
        raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {e}")
    
    return {"job_id": job_id, "status": "queued", "payload_json_path": json_path}


@app.get(
    "/api/queue",
    description="List queue jobs",
    summary="List queue",
)
async def list_queue_api(status: str = None):
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()
    db_file = os.path.join(path, "queue.db")
    init_queue_db(db_file)
    jobs = list_queue_jobs(db_file, status)
    return {"jobs": jobs}


@app.get(
    "/api/queue/{job_id}",
    description="Get queue job details",
    summary="Get job",
)
async def get_queue_job_api(job_id: int):
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()
    db_file = os.path.join(path, "queue.db")
    init_queue_db(db_file)
    job = get_job(db_file, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job}


@app.post(
    "/api/queue/{job_id}/cancel",
    description="Cancel a queued job",
    summary="Cancel job",
)
async def cancel_queue_job_api(job_id: int):
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()
    db_file = os.path.join(path, "queue.db")
    init_queue_db(db_file)
    ok = cancel_job(db_file, job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    return {"job_id": job_id, "status": "cancelled"}


@app.get(
    "/api/queue/{job_id}/payload",
    description="Download the payload JSON for a queue job",
    summary="Download job payload",
)
async def download_queue_payload_api(job_id: int):
    from fastapi.responses import FileResponse
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()
    db_file = os.path.join(path, "queue.db")
    init_queue_db(db_file)
    job = get_job(db_file, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    json_path = job.get("payload_json_path")
    if json_path and os.path.exists(json_path):
        return FileResponse(
            json_path,
            media_type="application/json",
            filename=f"job_{job_id}_payload.json"
        )
    else:
        # Fallback: generate JSON from stored payload
        payload_str = job.get("payload", "{}")
        temp_path = os.path.join(path, f"temp_job_{job_id}.json")
        try:
            payload_obj = json.loads(payload_str)
            with open(temp_path, "w") as f:
                json.dump(payload_obj, f, indent=2)
            return FileResponse(
                temp_path,
                media_type="application/json",
                filename=f"job_{job_id}_payload.json"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate payload file: {e}")


def _queue_worker_loop_api(poll_interval: float = 1.0):
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()
    db_file = os.path.join(path, "queue.db")
    init_queue_db(db_file)
    # Clean up any orphaned 'running' jobs from previous container runs
    orphaned_count = reset_orphaned_jobs(db_file)
    if orphaned_count > 0:
        print(f"⚠️  Found {orphaned_count} interrupted job(s) from restart - requeued for retry")
    
    print("✓ Queue worker started and ready to process jobs")
    
    while True:
        job = None
        job_id = None
        try:
            job = pop_next_job(db_file)
            if not job:
                time.sleep(poll_interval)
                continue
            
            job_id = job["id"]
            retry_count = job.get("retry_count", 0)
            retry_note = f" (retry #{retry_count})" if retry_count > 0 else ""
            print(f"Processing job {job_id}{retry_note}")
            
            update_job_progress(db_file, job_id, {"phase": "validating", "timestamp": time.time()})
            
            payload = json.loads(job["payload"]) if job.get("payload") else {}
            
            try:
                diffusion_config = LCMDiffusionSetting.model_validate(payload)
            except Exception as e:
                logging.warning(f"Job {job_id}: model_validate failed, trying parse_obj: {e}")
                try:
                    diffusion_config = LCMDiffusionSetting.parse_obj(payload)
                except Exception as e2:
                    error_msg = f"Failed to parse payload: {e2}"
                    logging.error(f"Job {job_id}: {error_msg}")
                    fail_job(db_file, job_id, error_msg)
                    continue
            
            # Check if cancelled before starting generation
            current_job = get_job(db_file, job_id)
            if current_job and current_job.get("status") == "cancelled":
                print(f"Job {job_id} cancelled before generation started")
                time.sleep(poll_interval)
                continue
            
            print(f"Job {job_id}: Starting image generation - {diffusion_config.image_width}x{diffusion_config.image_height}, steps={diffusion_config.inference_steps}")
            
            update_job_progress(db_file, job_id, {
                "phase": "loading_model",
                "model": diffusion_config.openvino_lcm_model_id,
                "timestamp": time.time()
            })
            
            try:
                app_settings.settings.lcm_diffusion_setting = diffusion_config
                if diffusion_config.diffusion_task == DiffusionTask.image_to_image:
                    try:
                        app_settings.settings.lcm_diffusion_setting.init_image = base64_image_to_pil(
                            diffusion_config.init_image
                        )
                    except Exception as e:
                        logging.warning(f"Job {job_id}: Failed to decode init_image: {e}")
                
                update_job_progress(db_file, job_id, {
                    "phase": "generating",
                    "size": f"{diffusion_config.image_width}x{diffusion_config.image_height}",
                    "steps": diffusion_config.inference_steps,
                    "timestamp": time.time()
                })
                
                images = context.generate_text_to_image(app_settings.settings)
                
                if images is None:
                    error_msg = context.error or "Image generation returned None"
                    logging.error(f"Job {job_id}: Generation failed - {error_msg}")
                    fail_job(db_file, job_id, error_msg)
                    time.sleep(0.1)
                    continue
                
            except Exception as e:
                error_msg = f"Image generation exception: {str(e)}"
                logging.error(f"Job {job_id}: {error_msg}")
                logging.error(f"Job {job_id}: Full traceback:\n{traceback.format_exc()}")
                fail_job(db_file, job_id, error_msg)
                time.sleep(0.1)
                continue
            
            # Check if job was cancelled during generation (before saving)
            current_job = get_job(db_file, job_id)
            if current_job and current_job.get("status") == "cancelled":
                print(f"Job {job_id} cancelled after generation, not saving files")
                time.sleep(poll_interval)
                continue
            
            try:
                update_job_progress(db_file, job_id, {
                    "phase": "saving",
                    "image_count": len(images) if images else 0,
                    "timestamp": time.time()
                })
                
                saved = context.save_images(images, app_settings.settings)
                print(f"Job {job_id}: Saved {len(saved) if saved else 0} images")
                
                # Final check before marking complete
                current_job = get_job(db_file, job_id)
                if current_job and current_job.get("status") == "cancelled":
                    print(f"Job {job_id} cancelled after saving, marking as cancelled")
                    time.sleep(poll_interval)
                    continue
                
                complete_job(db_file, job_id, {"saved": saved, "latency": context.latency})
                print(f"Job {job_id}: Completed successfully")
                
            except Exception as e:
                error_msg = f"Failed to save images: {str(e)}"
                logging.error(f"Job {job_id}: {error_msg}")
                logging.error(f"Job {job_id}: Full traceback:\n{traceback.format_exc()}")
                fail_job(db_file, job_id, error_msg)
            
            # Sleep briefly after processing to prevent tight loop
            time.sleep(0.1)
            
        except Exception as e:
            error_msg = f"Queue worker critical error: {str(e)}"
            logging.exception(error_msg)
            print(f"CRITICAL: {error_msg}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            
            try:
                if job_id:
                    fail_job(db_file, job_id, error_msg)
                    print(f"Marked job {job_id} as failed")
            except Exception as e2:
                logging.error(f"Failed to mark job as failed: {e2}")
            
            # Sleep longer after critical errors to prevent rapid failures
            time.sleep(poll_interval * 2)


# start background worker for API server
_worker_api = threading.Thread(target=_queue_worker_loop_api, daemon=True)
_worker_api.start()
