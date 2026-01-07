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

    files = []
    for entry in sorted(os.listdir(path), reverse=True):
        full = os.path.join(path, entry)
        if os.path.isfile(full):
            stat = os.stat(full)
            files.append(
                {
                    "name": entry,
                    "url": f"/results/{entry}",
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )

    return files
