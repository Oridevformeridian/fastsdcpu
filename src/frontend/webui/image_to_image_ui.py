from typing import Any
import gradio as gr
from backend.models.lcmdiffusion_setting import DiffusionTask
from models.interface_types import InterfaceType
from frontend.utils import is_reshape_required
from constants import DEVICE
from state import get_settings, get_context
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.request
import urllib.parse
import os

API_BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")
from frontend.webui.errors import show_error

app_settings = get_settings()

previous_width = 0
previous_height = 0
previous_model_id = ""
previous_num_of_images = 0


def generate_image_to_image(
    prompt,
    negative_prompt,
    init_image,
    strength,
) -> Any:
    context = get_context(InterfaceType.WEBUI)
    global \
        previous_height, \
        previous_width, \
        previous_model_id, \
        previous_num_of_images, \
        app_settings

    app_settings.settings.lcm_diffusion_setting.prompt = prompt
    app_settings.settings.lcm_diffusion_setting.negative_prompt = negative_prompt
    # If the UI did not provide an init_image, fall back to any pre-set image
    if init_image is None:
        init_image = app_settings.settings.lcm_diffusion_setting.init_image
    else:
        app_settings.settings.lcm_diffusion_setting.init_image = init_image
    app_settings.settings.lcm_diffusion_setting.strength = strength

    app_settings.settings.lcm_diffusion_setting.diffusion_task = (
        DiffusionTask.image_to_image.value
    )
    model_id = app_settings.settings.lcm_diffusion_setting.openvino_lcm_model_id
    reshape = False
    image_width = app_settings.settings.lcm_diffusion_setting.image_width
    image_height = app_settings.settings.lcm_diffusion_setting.image_height
    num_images = app_settings.settings.lcm_diffusion_setting.number_of_images
    if app_settings.settings.lcm_diffusion_setting.use_openvino:
        reshape = is_reshape_required(
            previous_width,
            image_width,
            previous_height,
            image_height,
            previous_model_id,
            model_id,
            previous_num_of_images,
            num_images,
        )

    def _enqueue():
        cfg = app_settings.settings.lcm_diffusion_setting
        try:
            payload = cfg.model_dump()
        except Exception:
            try:
                payload = cfg.dict()
            except Exception:
                payload = {}
        url = API_BASE.rstrip("/") + "/api/queue"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.load(resp)
        except Exception as e:
            return {"error": str(e)}

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_enqueue)
        resp = future.result()
        if resp and resp.get("job_id"):
            return [], f"Enqueued job {resp.get('job_id')}"
        else:
            err = resp.get("error") if resp else "failed to enqueue"
            show_error(err)
            return None, f"Error: {err}"

    previous_width = image_width
    previous_height = image_height
    previous_model_id = model_id
    previous_num_of_images = num_images
    return images


def get_image_to_image_ui() -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Init image", type="pil")
                with gr.Row():
                    prompt = gr.Textbox(
                        show_label=False,
                        lines=3,
                        placeholder="A fantasy landscape",
                        container=False,
                    )

                    generate_btn = gr.Button(
                        "Generate",
                        elem_id="generate_button",
                        scale=0,
                    )
                status = gr.Markdown("")
                negative_prompt = gr.Textbox(
                    label="Negative prompt (Works in LCM-LoRA mode, set guidance > 1.0):",
                    lines=1,
                    placeholder="",
                )
                strength = gr.Slider(
                    0.1,
                    1,
                    value=app_settings.settings.lcm_diffusion_setting.strength,
                    step=0.01,
                    label="Strength",
                )

                input_params = [
                    prompt,
                    negative_prompt,
                    input_image,
                    strength,
                ]

            with gr.Column():
                output = gr.Gallery(
                    label="Generated images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    height=512,
                )

    generate_btn.click(
        fn=generate_image_to_image,
        inputs=input_params,
        outputs=[output, status],
    )
