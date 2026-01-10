from datetime import datetime
import subprocess
import os

import gradio as gr
from backend.device import get_device_name
from constants import APP_VERSION
from frontend.webui.controlnet_ui import get_controlnet_ui
from frontend.webui.generation_settings_ui import get_generation_settings_ui
from frontend.webui.image_to_image_ui import get_image_to_image_ui
from frontend.webui.image_variations_ui import get_image_variations_ui
from frontend.webui.lora_models_ui import get_lora_models_ui
from frontend.webui.models_ui import get_models_ui
from frontend.webui.text_to_image_ui import get_text_to_image_ui
from frontend.webui.upscaler_ui import get_upscaler_ui
from frontend.webui.results_review_ui import get_results_review_ui
from frontend.webui.queue_ui import get_queue_ui
from state import get_settings

app_settings = get_settings()


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_footer_message() -> str:
    version = f"<center><p> {APP_VERSION} "
    current_year = datetime.now().year
    footer_msg = version + (
        f'  © 2023 - {current_year} <a href="https://github.com/rupeshs">'
        " Rupesh Sreeraman</a></p></center>"
    )
    return footer_msg


def get_web_ui() -> gr.Blocks:
    def change_mode(mode):
        global app_settings
        app_settings.settings.lcm_diffusion_setting.use_lcm_lora = False
        app_settings.settings.lcm_diffusion_setting.use_openvino = False
        app_settings.settings.lcm_diffusion_setting.use_gguf_model = False
        if mode == "LCM-LoRA":
            app_settings.settings.lcm_diffusion_setting.use_lcm_lora = True
        elif mode == "LCM-OpenVINO":
            app_settings.settings.lcm_diffusion_setting.use_openvino = True
        elif mode == "GGUF":
            app_settings.settings.lcm_diffusion_setting.use_gguf_model = True

    # Preserve saved LoRA state across restarts: only disable LoRA on
    # startup when there is no saved LoRA path. ControlNet remains
    # disabled by default in the WebUI to avoid surprise behavior.
    if app_settings.settings.lcm_diffusion_setting.lora:
        lora_cfg = app_settings.settings.lcm_diffusion_setting.lora
        if not lora_cfg.path:
            lora_cfg.enabled = False
    if app_settings.settings.lcm_diffusion_setting.controlnet:
        app_settings.settings.lcm_diffusion_setting.controlnet.enabled = False
    theme = gr.themes.Default(
        primary_hue="blue",
    )
    with gr.Blocks(
        title="FastSD CPU",
        theme=theme,
        css="footer {visibility: hidden}",
    ) as fastsd_web_ui:
        gr.HTML("<center><h2>Image Generator 3d Pro Max Mini Micro Manic</h2></center>")
        with gr.Row():
            with gr.Column(scale=8):
                gr.HTML("<center><H1>FastSD CPU</H1></center>")
            with gr.Column(scale=2):
                gr.Markdown(f"**Commit:** `{_get_git_commit()}`", elem_id="commit_hash")
        gr.Markdown(
            f"**Processor :  {get_device_name()}**",
            elem_id="processor",
        )
        current_mode = "LCM"
        if app_settings.settings.lcm_diffusion_setting.use_openvino:
            current_mode = "LCM-OpenVINO"
        elif app_settings.settings.lcm_diffusion_setting.use_lcm_lora:
            current_mode = "LCM-LoRA"
        elif app_settings.settings.lcm_diffusion_setting.use_gguf_model:
            current_mode = "GGUF"

        mode = gr.Radio(
            ["LCM", "LCM-LoRA", "LCM-OpenVINO", "GGUF"],
            label="Mode",
            info="Current working mode",
            value=current_mode,
        )
        mode.change(change_mode, inputs=mode)

        with gr.Tabs():
            with gr.TabItem("Text to Image"):
                get_text_to_image_ui()
            with gr.TabItem("Image to Image"):
                get_image_to_image_ui()
            with gr.TabItem("Image Variations"):
                get_image_variations_ui()
            with gr.TabItem("Upscaler"):
                get_upscaler_ui()
            with gr.TabItem("Generation Settings"):
                get_generation_settings_ui()
            with gr.TabItem("Models"):
                get_models_ui()
            with gr.TabItem("Lora Models"):
                get_lora_models_ui()
            with gr.TabItem("ControlNet"):
                get_controlnet_ui()
            with gr.TabItem("Results"):
                get_results_review_ui()
            with gr.TabItem("Queue"):
                get_queue_ui()

        gr.HTML(_get_footer_message())

    return fastsd_web_ui


def start_webui(
    share: bool = False,
):
    webui = get_web_ui()
    webui.queue()
    webui.launch(server_name="0.0.0.0", share=share)
