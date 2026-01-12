import platform
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import urllib.parse

from backend.api.models.response import StableDiffusionResponse
from backend.base64_image import base64_image_to_pil, pil_image_to_base64_str
from backend.device import get_device_name
from backend.models.device import DeviceInfo
from backend.models.lcmdiffusion_setting import DiffusionTask, LCMDiffusionSetting
from constants import APP_VERSION, DEVICE
from context import Context
from backend.pipeline_lock import pipeline_lock
from models.interface_types import InterfaceType
from state import get_settings
from paths import FastStableDiffusionPaths
from backend.api.models.review import ReviewRequest, ReviewResponse
from backend.reviews_db import (
    init_db,
    set_review,
    get_review,
    delete_review,
    list_reviews,
)
import shutil
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
    is_queue_paused,
)
import threading
import time
import json

# ---------------------------------------------------------------------------
# MONKEY-PATCH: Conv2d adapter no-op shim
#
# Some third-party adapter/runtime implementations (used by diffusers/peft)
# may attempt to call adapter management methods (e.g. `delete_adapter`,
# `add_adapter`, `merge_adapter`) on submodules. In some environments those
# calls end up resolving to plain `torch.nn.Conv2d` objects which do not
# implement those methods, causing errors like:
#     "'Conv2d' object has no attribute 'delete_adapter'"
#
# This temporary shim adds safe no-op adapter methods to `nn.Conv2d` so
# that adapter removal/add operations become no-ops instead of raising
# AttributeError. Remove this once the underlying library versions are
# upgraded/fixed (TODO: track upstream diffusers/peft fix).
# ---------------------------------------------------------------------------
try:
    import torch.nn as _nn

    def _lora_noop(self, *args, **kwargs):
        return None

    if not hasattr(_nn.Conv2d, "delete_adapter"):
        _nn.Conv2d.delete_adapter = _lora_noop
    if not hasattr(_nn.Conv2d, "add_adapter"):
        _nn.Conv2d.add_adapter = _lora_noop
    if not hasattr(_nn.Conv2d, "merge_adapter"):
        _nn.Conv2d.merge_adapter = _lora_noop
except Exception as _ex:
    print(f"Warning: Conv2d adapter monkey-patch failed: {_ex}")

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

# Mount generated results folder so images can be served by the API server
results_path = app_settings.settings.generated_images.path
if not results_path:
    results_path = FastStableDiffusionPaths.get_results_path()

if not os.path.exists(results_path):
    try:
        os.makedirs(results_path, exist_ok=True)
    except Exception:
        pass


# Serve generated result files with request-time logging so we can capture
# failures (connection resets, truncated reads). This replaces the previous
# StaticFiles mount to allow logging of client info and file metadata.
@app.get("/results/{file_path:path}")
async def serve_result(file_path: str, request: Request):
    # decode any URL-encoding and sanitize path
    file_path_decoded = urllib.parse.unquote(file_path)
    safe_path = os.path.normpath(file_path_decoded).lstrip(os.sep)
    full = os.path.join(results_path, safe_path)
    client = None
    try:
        client = request.client.host if request.client else "unknown"
    except Exception:
        client = "unknown"
    print(f"[RESULTS] request from {client} path={file_path_decoded} full={full}")

    if not os.path.isfile(full):
        print(f"[RESULTS] not found: {full}")
        raise HTTPException(status_code=404, detail="Result file not found")

    try:
        stat = os.stat(full)
        print(f"[RESULTS] serving {full} size={stat.st_size} mtime={stat.st_mtime}")
        # log a prefix of the file bytes to validate content on problematic requests
        try:
            with open(full, "rb") as f:
                prefix = f.read(128)
                if prefix:
                    print(f"[RESULTS] prefix_hex={prefix.hex()[:400]}")
                else:
                    print(f"[RESULTS] prefix_hex=(empty)")
        except Exception as e:
            print(f"[RESULTS] failed reading prefix: {e}")

        headers = {"X-Result-Size": str(stat.st_size)}
        return FileResponse(full, headers=headers)
    except Exception as e:
        print(f"[RESULTS] error serving {full}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    "/api/results",
    description="List generated result files",
    summary="List generated results",
)
async def list_results():
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Results directory not found")
    # Use SQLite for review persistence
    db_file = os.path.join(path, "reviews.db")
    init_db(db_file)
    reviews = list_reviews(db_file)

    files = []
    entries = [e for e in os.listdir(path) if os.path.isfile(os.path.join(path, e)) and os.stat(os.path.join(path, e)).st_size > 0]
    # sort by modification time (newest first)
    entries.sort(key=lambda e: os.stat(os.path.join(path, e)).st_mtime, reverse=True)

    for entry in entries:
        full = os.path.join(path, entry)
        stat = os.stat(full)
        file_review = reviews.get(entry)
        files.append(
            {
                "name": entry,
                "url": f"/results/{entry}",
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "review": file_review,
            }
        )

    return files



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
        and os.stat(os.path.join(path, e)).st_size > 0
        and (e.lower().endswith('.jpg') or e.lower().endswith('.png') or e.lower().endswith('.jpeg'))
    ]
    all_entries.sort(key=lambda e: os.stat(os.path.join(path, e)).st_mtime, reverse=True)

    start = page * size
    end = start + size
    page_entries = all_entries[start:end]

    # load reviews for each page entry
    db_file = os.path.join(path, "reviews.db")
    init_db(db_file)

    results = []
    for entry_name in page_entries:
        full = os.path.join(path, entry_name)
        stat = os.stat(full)
        file_review = get_review(db_file, entry_name)
        
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
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass

        results.append(
            {
                "name": entry_name,
                "url": f"/results/{entry_name}",
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "meta": meta,
                "review": file_review,
            }
        )

    payload = {"page": page, "size": size, "results": results, "total": len(all_entries)}
    cache["pages"][key] = {"timestamp": now, "data": payload}
    return payload



@app.get(
    "/api/results/{name}/review",
    description="Get review metadata for a generated result",
    summary="Get review metadata",
)
async def get_result_review(name: str):
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()

    full = os.path.join(path, name)
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="Result file not found")

    db_file = os.path.join(path, "reviews.db")
    entry = get_review(db_file, name)
    if not entry:
        raise HTTPException(status_code=404, detail="No review metadata for this file")

    return ReviewResponse(name=name, status=entry.get("status"), note=entry.get("note"))


@app.post(
    "/api/results/{name}/archive",
    description="Archive a generated result file",
    summary="Archive result",
)
async def archive_result(name: str):
    # Decode any URL-encoded filename components
    name = urllib.parse.unquote(name)
    # Guard against accidental absolute paths
    name = name.lstrip("/")

    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()

    full = os.path.join(path, name)
    # Debug info to help diagnose missing-file issues
    print(f"Archive request: name={name}, full={full}, exists={os.path.isfile(full)}")

    if not os.path.isfile(full):
        # Fallback: try to find a file whose basename matches the requested name
        try:
            for f in os.listdir(path):
                if f == name:
                    full = os.path.join(path, f)
                    break
        except Exception:
            pass

    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail=f"Result file not found: attempted {full}")

    archive_dir = os.path.join(path, "archive")
    try:
        os.makedirs(archive_dir, exist_ok=True)
    except Exception:
        pass

    dest = os.path.join(archive_dir, name)
    try:
        shutil.move(full, dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to archive file: {e}")

    # Attempt to move corresponding metadata JSON if present. The JSON file
    # uses the UUID base name (without -N suffix), so derive that name.
    name_without_ext = os.path.splitext(name)[0]
    if '-' in name_without_ext:
        parts = name_without_ext.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            name_without_ext = parts[0]

    json_name = name_without_ext + ".json"
    json_full = os.path.join(path, json_name)
    if os.path.isfile(json_full):
        try:
            shutil.move(json_full, os.path.join(archive_dir, json_name))
        except Exception:
            pass

    return {"archived": True, "path": dest}


@app.post(
    "/api/results/{name}/review",
    description="Set review metadata for a generated result",
    summary="Set review metadata",
)
async def set_result_review(name: str, review: ReviewRequest):
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()

    full = os.path.join(path, name)
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="Result file not found")

    db_file = os.path.join(path, "reviews.db")
    set_review(db_file, name, review.status.value, review.note)
    return ReviewResponse(name=name, status=review.status, note=review.note)


@app.delete(
    "/api/results/{name}/review",
    description="Delete review metadata for a generated result",
    summary="Delete review metadata",
)
async def delete_result_review(name: str):
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()

    full = os.path.join(path, name)
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="Result file not found")

    db_file = os.path.join(path, "reviews.db")
    deleted = delete_review(db_file, name)
    return {"deleted": deleted}


@app.post(
    "/api/queue",
    description="Enqueue a generation task",
    summary="Enqueue generation",
)
async def enqueue(diffusion_config: LCMDiffusionSetting):
    """Store the task in a persistent queue and return job id."""
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()
    db_file = os.path.join(path, "queue.db")
    init_queue_db(db_file)
    try:
        payload = diffusion_config.model_dump()
    except Exception as e:
        # try fallback to dict
        try:
            payload = diffusion_config.dict()
        except Exception as e2:
            raise HTTPException(status_code=400, detail=f"Invalid diffusion config: {e} / {e2}")

    try:
        job_id = enqueue_job(db_file, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {e}")

    print(f"Enqueued job {job_id}")
    return {"job_id": job_id, "status": "queued"}


@app.get(
    "/api/queue",
    description="List queue jobs",
    summary="List queue",
)
async def list_queue(status: str = None):
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()
    db_file = os.path.join(path, "queue.db")
    init_queue_db(db_file)
    jobs = list_queue_jobs(db_file, status)
    return {"jobs": jobs}


@app.post(
    "/api/queue/{job_id}/cancel",
    description="Cancel a queued job",
    summary="Cancel job",
)
async def cancel_queue_job(job_id: int):
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
    "/api/queue/{job_id}",
    description="Get queue job details",
    summary="Get job",
)
async def get_queue_job(job_id: int):
    path = app_settings.settings.generated_images.path
    if not path:
        path = FastStableDiffusionPaths.get_results_path()
    db_file = os.path.join(path, "queue.db")
    init_queue_db(db_file)
    job = get_job(db_file, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job}


def _queue_worker_loop(poll_interval: float = 1.0):
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
            # Check if queue is paused
            if is_queue_paused(db_file):
                time.sleep(poll_interval)
                continue
            
            job = pop_next_job(db_file)
            if not job:
                time.sleep(poll_interval)
                continue
            job_id = job["id"]
            payload = json.loads(job["payload"]) if job.get("payload") else {}
            # reconstruct LCMDiffusionSetting
            try:
                diffusion_config = LCMDiffusionSetting.model_validate(payload)
            except Exception:
                # backward compat for pydantic v1
                diffusion_config = LCMDiffusionSetting.parse_obj(payload)

            # Check if cancelled before starting generation
            current_job = get_job(db_file, job_id)
            if current_job and current_job.get("status") == "cancelled":
                print(f"Job {job_id} cancelled before generation started")
                time.sleep(poll_interval)
                continue

            # set into app settings and handle image init
            app_settings.settings.lcm_diffusion_setting = diffusion_config
            if diffusion_config.diffusion_task == DiffusionTask.image_to_image:
                try:
                    app_settings.settings.lcm_diffusion_setting.init_image = base64_image_to_pil(
                        diffusion_config.init_image
                    )
                except Exception:
                    pass

            # Ensure pipeline mutations and generation are serialized.
            with pipeline_lock:
                # Apply any requested LoRA into the live pipeline before generation
                try:
                    lora_cfg = diffusion_config.lora
                    if lora_cfg and getattr(lora_cfg, "enabled", False) and lora_cfg.path:
                        try:
                            from backend.lora import load_lora_weight, get_active_lora_weights
                            from pathlib import Path

                            if context.lcm_text_to_image.pipeline:
                                adapter_name = Path(str(lora_cfg.path)).stem
                                active = get_active_lora_weights()
                                if not any(a[0] == adapter_name for a in active):
                                    load_lora_weight(context.lcm_text_to_image.pipeline, diffusion_config)
                        except Exception as _ex:
                            print(f"Failed to apply saved LoRA before job start: {_ex}")
                except Exception:
                    pass

                images = context.generate_text_to_image(app_settings.settings)
            
            # Check if job was cancelled during generation (before saving)
            current_job = get_job(db_file, job_id)
            if current_job and current_job.get("status") == "cancelled":
                print(f"Job {job_id} cancelled after generation, not saving files")
                time.sleep(poll_interval)
                continue
                
            if images:
                saved = context.save_images(images, app_settings.settings)
                # Final check before marking complete
                current_job = get_job(db_file, job_id)
                if current_job and current_job.get("status") == "cancelled":
                    print(f"Job {job_id} cancelled after saving, marking as cancelled")
                    time.sleep(poll_interval)
                    continue
                complete_job(db_file, job_id, {"saved": saved, "latency": context.latency})
            else:
                fail_job(db_file, job_id, context.error or "no images generated")
            # Sleep briefly after processing to prevent tight loop
            time.sleep(0.1)
        except Exception as e:
            # avoid tight-loop on unexpected errors
            print(f"Queue worker error: {e}")
            try:
                if job and job.get("id"):
                    fail_job(db_file, job.get("id"), str(e))
            except Exception:
                pass
            time.sleep(poll_interval)


# Start background queue worker
_worker = threading.Thread(target=_queue_worker_loop, daemon=True)
_worker.start()
