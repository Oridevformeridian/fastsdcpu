import platform
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
)
import threading
import time
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
                        print(f"[DEBUG] Loaded JSON for {entry_name}: UUID={uuid}, keys={list(meta.keys())}, prompt={meta.get('prompt', 'N/A')[:50]}")
                        break
                    except (json.JSONDecodeError, IOError) as e:
                        if attempt < 2:
                            time.sleep(0.05)  # Wait 50ms before retry
                        else:
                            print(f"[DEBUG] Failed to load JSON for {entry_name} after 3 attempts: {e}")
            else:
                print(f"[DEBUG] JSON not found for {entry_name}: {json_path}")
        else:
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

    payload = {"page": page, "size": size, "results": results, "total": len(all_entries)}
    cache["pages"][key] = {"timestamp": now, "data": payload}
    return payload


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
    try:
        job_id = enqueue_job(db_file, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {e}")
    print(f"Enqueued job {job_id} via api/web.py")
    return {"job_id": job_id, "status": "queued"}


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


def _queue_worker_loop_api(poll_interval: float = 1.0):
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()
    db_file = os.path.join(path, "queue.db")
    init_queue_db(db_file)
    # Clean up any orphaned 'running' jobs from previous container runs
    orphaned_count = reset_orphaned_jobs(db_file)
    if orphaned_count > 0:
        print(f"Reset {orphaned_count} orphaned job(s) from previous run")
    while True:
        job = None
        try:
            job = pop_next_job(db_file)
            if not job:
                time.sleep(poll_interval)
                continue
            job_id = job["id"]
            payload = json.loads(job["payload"]) if job.get("payload") else {}
            try:
                diffusion_config = LCMDiffusionSetting.model_validate(payload)
            except Exception:
                diffusion_config = LCMDiffusionSetting.parse_obj(payload)
            app_settings.settings.lcm_diffusion_setting = diffusion_config
            if diffusion_config.diffusion_task == DiffusionTask.image_to_image:
                try:
                    app_settings.settings.lcm_diffusion_setting.init_image = base64_image_to_pil(
                        diffusion_config.init_image
                    )
                except Exception:
                    pass
            images = context.generate_text_to_image(app_settings.settings)
            # Check if job was cancelled during generation
            current_job = get_job(db_file, job_id)
            if current_job and current_job.get("status") == "cancelled":
                time.sleep(poll_interval)  # Sleep before next iteration
                continue
            if images:
                saved = context.save_images(images, app_settings.settings)
                complete_job(db_file, job_id, {"saved": saved, "latency": context.latency})
            else:
                fail_job(db_file, job_id, context.error or "no images generated")
            # Sleep briefly after processing to prevent tight loop
            time.sleep(0.1)
        except Exception as e:
            print(f"Queue worker error: {e}")
            try:
                if job and job.get("id"):
                    fail_job(db_file, job.get("id"), str(e))
            except Exception:
                pass
            time.sleep(poll_interval)


# start background worker for API server
_worker_api = threading.Thread(target=_queue_worker_loop_api, daemon=True)
_worker_api.start()
