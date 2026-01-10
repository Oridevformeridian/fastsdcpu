import gradio as gr
from typing import Any
from backend.models.lcmdiffusion_setting import DiffusionTask
from models.interface_types import InterfaceType
from constants import DEVICE
from state import get_settings, get_context
from frontend.utils import is_reshape_required, get_valid_lora_model
from backend.lora import get_lora_models
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.request
import urllib.parse
import os

API_BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")  # default to API server
from frontend.webui.errors import show_error

app_settings = get_settings()

previous_width = 0
previous_height = 0
previous_model_id = ""
previous_num_of_images = 0


def generate_text_to_image(
    prompt,
    neg_prompt,
) -> Any:
    context = get_context(InterfaceType.WEBUI)
    global \
        previous_height, \
        previous_width, \
        previous_model_id, \
        previous_num_of_images, \
        app_settings
    app_settings.settings.lcm_diffusion_setting.prompt = prompt
    app_settings.settings.lcm_diffusion_setting.negative_prompt = neg_prompt
    app_settings.settings.lcm_diffusion_setting.diffusion_task = (
        DiffusionTask.text_to_image.value
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
        
        # Convert PIL Image to base64 if present (needed for JSON serialization)
        if cfg.init_image and hasattr(cfg.init_image, 'mode'):  # Check if it's a PIL Image
            from backend.base64_image import pil_image_to_base64_str
            cfg.init_image = pil_image_to_base64_str(cfg.init_image)
        
        # if no API configured, run local generation directly
        if not API_BASE:
            try:
                imgs = context.generate_text_to_image(app_settings.settings)
                if imgs:
                    saved = context.save_images(imgs, app_settings.settings)
                    return {"local": True, "saved": saved}
                return {"error": "local generation produced no images"}
            except Exception as le:
                return {"error": f"local generation failed: {le}"}
        # pydantic v2 -> model_dump, fallback to dict()
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
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode('utf-8')
            except Exception:
                body = str(e)
            return {"error": f"HTTPError: {e.code} {body}"}
        except urllib.error.URLError as e:
            # Connection refused -> run local generation as fallback
            reason = getattr(e, "reason", None)
            msg = str(e)
            if (isinstance(reason, ConnectionRefusedError)) or (hasattr(reason, "errno") and getattr(reason, "errno") == 111) or ("Connection refused" in msg):
                # run generation locally (we are already in a thread)
                try:
                    imgs = context.generate_text_to_image(app_settings.settings)
                    if imgs:
                        saved = context.save_images(imgs, app_settings.settings)
                        return {"local": True, "saved": saved}
                    return {"error": "local generation produced no images"}
                except Exception as le:
                    return {"error": f"local generation failed: {le}"}
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_enqueue)
        resp = future.result()
        if resp and resp.get("job_id"):
            return [], f"Enqueued job {resp.get('job_id')}"
        elif resp and resp.get("local"):
            saved = resp.get("saved") or []
            return [], f"Ran locally, saved: {saved}"
        else:
            err = resp.get("error") if resp else "failed to enqueue"
            show_error(err)
            return None, f"Error: {err}"

    previous_width = image_width
    previous_height = image_height
    previous_model_id = model_id
    previous_num_of_images = num_images
    return images


def get_text_to_image_ui() -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
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
                    label="Negative prompt (Works in LCM-LoRA mode, set guidance > 1.0) :",
                    lines=1,
                    placeholder="",
                )

                # LoRA controls (local to this generation)
                lora_models_map = get_lora_models(
                    app_settings.settings.lcm_diffusion_setting.lora.models_dir
                )
                valid_model = get_valid_lora_model(
                    list(lora_models_map.values()),
                    app_settings.settings.lcm_diffusion_setting.lora.path,
                    app_settings.settings.lcm_diffusion_setting.lora.models_dir,
                )
                lora_choices = ["None"] + list(lora_models_map.keys())
                lora_enabled = gr.Checkbox(
                    label="Use LoRA",
                    value=app_settings.settings.lcm_diffusion_setting.lora.enabled,
                    interactive=True,
                )
                lora_model = gr.Dropdown(
                    lora_choices,
                    label="LoRA model",
                    value=(valid_model if valid_model != "" else "None"),
                    interactive=True,
                )
                lora_weight = gr.Slider(
                    0.0,
                    1.0,
                    value=app_settings.settings.lcm_diffusion_setting.lora.weight,
                    step=0.05,
                    label="LoRA weight",
                )

                input_params = [prompt, negative_prompt, lora_enabled, lora_model, lora_weight]

            with gr.Column():
                output = gr.Gallery(
                    label="Generated images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    height=512,
                )
    def _wrap_generate(prompt, neg_prompt, lora_enabled_val, lora_model_val, lora_weight_val):
        # Apply LoRA selection to settings for this generation
        if app_settings.settings.lcm_diffusion_setting.use_openvino and lora_enabled_val:
            from frontend.webui.errors import show_error as _show_err

            _show_err("LoRA is not supported in OpenVINO mode.")
            return None, "Error: LoRA not supported in OpenVINO"

        lora_models_map_local = get_lora_models(
            app_settings.settings.lcm_diffusion_setting.lora.models_dir
        )
        if lora_model_val != "None":
            app_settings.settings.lcm_diffusion_setting.lora.path = lora_models_map_local.get(lora_model_val, "")
            app_settings.settings.lcm_diffusion_setting.lora.weight = lora_weight_val
            app_settings.settings.lcm_diffusion_setting.lora.enabled = lora_enabled_val
        else:
            app_settings.settings.lcm_diffusion_setting.lora.path = ""
            app_settings.settings.lcm_diffusion_setting.lora.enabled = False

        return generate_text_to_image(prompt, neg_prompt)

    generate_btn.click(
        fn=_wrap_generate,
        inputs=input_params,
        outputs=[output, status],
    )
