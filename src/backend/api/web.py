import platform

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.models.response import StableDiffusionResponse
from backend.base64_image import base64_image_to_pil, pil_image_to_base64_str
from backend.device import get_device_name
from backend.models.device import DeviceInfo
from backend.models.lcmdiffusion_setting import DiffusionTask, LCMDiffusionSetting
from constants import APP_VERSION, DEVICE
from context import Context
from models.interface_types import InterfaceType
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
    while True:
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
            if images:
                saved = context.save_images(images, app_settings.settings)
                complete_job(db_file, job_id, {"saved": saved, "latency": context.latency})
            else:
                fail_job(db_file, job_id, context.error or "no images generated")
        except Exception as e:
            try:
                if job and job.get("id"):
                    fail_job(db_file, job.get("id"), str(e))
            except Exception:
                pass
            time.sleep(poll_interval)


# start background worker for API server
_worker_api = threading.Thread(target=_queue_worker_loop_api, daemon=True)
_worker_api.start()
