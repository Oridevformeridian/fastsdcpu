import platform
import logging

import uvicorn
from backend.device import get_device_name
from backend.models.device import DeviceInfo
from constants import APP_VERSION, DEVICE
from context import Context
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi_mcp import FastApiMCP
from state import get_settings
from fastapi.middleware.cors import CORSMiddleware
from models.interface_types import InterfaceType
from fastapi.staticfiles import StaticFiles

SERVER_PORT = 8000

app_settings = get_settings()
app = FastAPI(
    title="FastSD CPU",
    description="Fast stable diffusion on CPU",
    version=APP_VERSION,
    license_info={
        "name": "MIT",
        "identifier": "MIT",
    },
    describe_all_responses=True,
    describe_full_response_schema=True,
)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to prevent server crashes"""
    logging.exception(f"Unhandled exception in {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": "Internal server error"}
    )

# Initialize context with error handling
context = None
try:
    logging.info("Initializing context...")
    context = Context(InterfaceType.API_SERVER)
    logging.info("Context initialized successfully")
except Exception as e:
    logging.exception(f"Failed to initialize context: {e}")
    # Continue starting server even if context initialization fails
    # This allows the server to stay up and report errors via API

app.mount("/results", StaticFiles(directory="results"), name="results")


@app.get(
    "/info",
    description="Get system information",
    summary="Get system information",
    operation_id="get_system_info",
)
async def info() -> dict:
    device_info = DeviceInfo(
        device_type=DEVICE,
        device_name=get_device_name(),
        os=platform.system(),
        platform=platform.platform(),
        processor=platform.processor(),
    )
    return device_info.model_dump()


@app.post(
    "/generate",
    description="Generate image from text prompt",
    summary="Text to image generation",
    operation_id="generate",
)
async def generate(
    prompt: str,
    request: Request,
) -> str:
    """
    Returns URL of the generated image for text prompt
    """
    try:
        if context is None:
            raise Exception("Context not initialized - server startup may have failed")
        
        app_settings.settings.lcm_diffusion_setting.prompt = prompt
        images = context.generate_text_to_image(app_settings.settings)
        
        if images is None:
            error_msg = context.error() or "Image generation failed"
            logging.error(f"Image generation failed: {error_msg}")
            raise Exception(f"Image generation failed: {error_msg}")
        
        image_names = context.save_images(
            images,
            app_settings.settings,
        )
        
        if not image_names:
            raise Exception("Failed to save generated images")
        
        # url = request.url_for("results", path=image_names[0]) - Claude Desktop returns api_server
        url = f"http://localhost:{SERVER_PORT}/results/{image_names[0]}"
        image_url = f"The generated image available at the URL {url}"
        return image_url
    except Exception as e:
        logging.exception(f"Error in generate endpoint: {e}")
        raise


def start_mcp_server(port: int = 8000):
    global SERVER_PORT
    SERVER_PORT = port
    
    # Configure logging to prevent systemd journal issues
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    print(f"Starting MCP server on port {port}...")
    mcp = FastApiMCP(
        app,
        name="FastSDCPU MCP",
        description="MCP server for FastSD CPU API",
    )

    mcp.mount()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_config=None,  # Disable uvicorn's default log config to prevent journal handler issues
    )
