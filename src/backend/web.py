import platform
import os

import uvicorn
import json
import threading
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

    # reviews stored in a simple JSON file in the results directory
    REVIEWS_FILENAME = ".reviews.json"
    reviews_file = os.path.join(path, REVIEWS_FILENAME)

    _reviews_lock = threading.Lock()

    def _load_reviews():
        if not os.path.exists(reviews_file):
            return {}
        try:
            with open(reviews_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_reviews(data: dict):
        tmp = reviews_file + ".tmp"
        with _reviews_lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            try:
                os.replace(tmp, reviews_file)
            except Exception:
                pass

    reviews = _load_reviews()

    files = []
    for entry in sorted(os.listdir(path), reverse=True):
        full = os.path.join(path, entry)
        if os.path.isfile(full):
            stat = os.stat(full)
            file_review = reviews.get(entry) if isinstance(reviews, dict) else None
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

    reviews_file = os.path.join(path, ".reviews.json")
    try:
        with open(reviews_file, "r", encoding="utf-8") as f:
            reviews = json.load(f)
    except Exception:
        reviews = {}

    entry = reviews.get(name)
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

    reviews_file = os.path.join(path, ".reviews.json")
    _reviews_lock = threading.Lock()

    def _load():
        if not os.path.exists(reviews_file):
            return {}
        try:
            with open(reviews_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(data: dict):
        tmp = reviews_file + ".tmp"
        with _reviews_lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            try:
                os.replace(tmp, reviews_file)
            except Exception:
                pass

    reviews = _load()
    reviews[name] = {"status": review.status.value, "note": review.note}
    _save(reviews)

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

    reviews_file = os.path.join(path, ".reviews.json")
    try:
        with open(reviews_file, "r", encoding="utf-8") as f:
            reviews = json.load(f)
    except Exception:
        reviews = {}

    if name in reviews:
        reviews.pop(name)
        try:
            with open(reviews_file, "w", encoding="utf-8") as f:
                json.dump(reviews, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return {"deleted": True}
