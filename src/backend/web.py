import platform
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.models.response import StableDiffusionResponse
from backend.base64_image import base64_image_to_pil, pil_image_to_base64_str
from backend.device import get_device_name
from backend.models.device import DeviceInfo
from backend.models.lcmdiffusion_setting import DiffusionTask, LCMDiffusionSetting
from constants import APP_VERSION, DEVICE
from context import Context
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

app.mount("/results", StaticFiles(directory=results_path), name="results")


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
    entries = [e for e in os.listdir(path) if os.path.isfile(os.path.join(path, e))]
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

    # build listing for this page
    all_entries = [e for e in os.listdir(path) if os.path.isfile(os.path.join(path, e))]
    all_entries.sort(key=lambda e: os.stat(os.path.join(path, e)).st_mtime, reverse=True)

    start = page * size
    end = start + size
    page_entries = all_entries[start:end]

    # load reviews and sidecar JSON for each page entry
    db_file = os.path.join(path, "reviews.db")
    init_db(db_file)

    results = []
    for entry_name in page_entries:
        full = os.path.join(path, entry_name)
        stat = os.stat(full)
        file_review = get_review(db_file, entry_name)
        meta = {}
        json_path = os.path.join(path, os.path.splitext(entry_name)[0] + ".json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

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
