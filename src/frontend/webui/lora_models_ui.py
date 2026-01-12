import gradio as gr
from os import path
import os
import shutil
from backend.lora import (
    get_lora_models,
    get_active_lora_weights,
    update_lora_weights,
    load_lora_weight,
)
from state import get_settings, get_context
from frontend.utils import get_valid_lora_model
from models.interface_types import InterfaceType


_MAX_LORA_WEIGHTS = 5

_custom_lora_sliders = []
_custom_lora_names = []
_custom_lora_columns = []

app_settings = get_settings()


def on_click_update_weight(*lora_weights):
    update_weights = []
    active_weights = get_active_lora_weights()
    if not len(active_weights):
        # Try to auto-load saved LoRA into the current pipeline if possible
        ctx = get_context(InterfaceType.WEBUI)
        pipeline = ctx.lcm_text_to_image.pipeline
        settings = app_settings.settings.lcm_diffusion_setting
        if pipeline and settings.lora and settings.lora.enabled and settings.lora.path:
            try:
                from backend.lora import load_lora_weight

                load_lora_weight(pipeline, settings)
                active_weights = get_active_lora_weights()
            except Exception:
                pass
        if not len(active_weights):
            return [gr.Markdown.update(value="**LoRA enabled:** False (no adapters loaded)")]
    for idx, lora in enumerate(active_weights):
        update_weights.append(
            (
                lora[0],
                lora_weights[idx],
            )
        )
    if len(update_weights) > 0:
        update_lora_weights(
            get_context(InterfaceType.WEBUI).lcm_text_to_image.pipeline,
            app_settings.settings.lcm_diffusion_setting,
            update_weights,
        )
    # return status update
    return [gr.Markdown.update(value=f"**LoRA enabled:** {len(get_active_lora_weights())>0}")]


def on_click_load_lora(lora_name, lora_weight):
    if app_settings.settings.lcm_diffusion_setting.use_openvino:
        gr.Warning("Currently LoRA is not supported in OpenVINO.")
        return
    lora_models_map = get_lora_models(
        app_settings.settings.lcm_diffusion_setting.lora.models_dir
    )

    # Load a new LoRA
    settings = app_settings.settings.lcm_diffusion_setting
    settings.lora.fuse = False
    settings.lora.enabled = False
    print(f"Selected Lora Model :{lora_name}")
    print(f"Lora weight :{lora_weight}")
    # precompute dynamic outputs count for early returns
    outputs_count = len(_custom_lora_names) + len(_custom_lora_sliders) + len(_custom_lora_columns)
    def _empty_outputs_with_status(text):
        return [None] * outputs_count + [gr.Markdown.update(value=text)]

    # Handle unload request
    if lora_name == "None":
        settings.lora.path = ""
        settings.lora.enabled = False
        try:
            get_settings().save()
        except Exception:
            pass
        ctx = get_context(InterfaceType.WEBUI)
        pipeline = ctx.lcm_text_to_image.pipeline
        if pipeline:
            try:
                from backend.lora import remove_loaded_lora

                remove_loaded_lora(pipeline, settings)
                # return empty updates + status
                return _empty_outputs_with_status("**LoRA enabled:** False")
            except Exception:
                return _empty_outputs_with_status("**LoRA enabled:** False")
        else:
            return _empty_outputs_with_status("**LoRA enabled:** False")

    settings.lora.path = lora_models_map[lora_name]
    settings.lora.weight = lora_weight
    if not path.exists(settings.lora.path):
        gr.Warning("Invalid LoRA model path!")
        return
    ctx = get_context(InterfaceType.WEBUI)
    pipeline = ctx.lcm_text_to_image.pipeline
    # If there's no initialized pipeline in the WebUI context, persist the
    # selected LoRA into settings so it will be applied on the next
    # generation (or next queued job) instead of attempting an expensive
    # initialization here.
    if not pipeline:
        settings.lora.enabled = True
        try:
            get_settings().save()
        except Exception:
            pass
        return _empty_outputs_with_status("**LoRA enabled:** True (will apply on next generation)")

    settings.lora.enabled = True
    load_lora_weight(
        ctx.lcm_text_to_image.pipeline,
        settings,
    )

    # Update Gradio LoRA UI
    global _MAX_LORA_WEIGHTS
    values = []
    labels = []
    rows = []
    active_weights = get_active_lora_weights()
    for idx, lora in enumerate(active_weights):
        labels.append(f"{lora[0]}: ")
        values.append(lora[1])
        rows.append(gr.Row.update(visible=True))
    for i in range(len(active_weights), _MAX_LORA_WEIGHTS):
        labels.append(f"Update weight")
        values.append(0.0)
        rows.append(gr.Row.update(visible=False))
    # Append status update to outputs
    status_text = f"**LoRA enabled:** {settings.lora.enabled}"
    return labels + values + rows + [gr.Markdown.update(value=status_text)]


def get_lora_models_ui() -> None:
    with gr.Blocks() as ui:
        gr.HTML(
            "Download and place your LoRA model weights in <b>lora_models</b> folders and restart App"
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    lora_models_map = get_lora_models(
                        app_settings.settings.lcm_diffusion_setting.lora.models_dir
                    )
                    valid_model = get_valid_lora_model(
                        list(lora_models_map.values()),
                        app_settings.settings.lcm_diffusion_setting.lora.path,
                        app_settings.settings.lcm_diffusion_setting.lora.models_dir,
                    )
                    if valid_model != "":
                        valid_model_path = lora_models_map[valid_model]
                        app_settings.settings.lcm_diffusion_setting.lora.path = (
                            valid_model_path
                        )
                    else:
                        app_settings.settings.lcm_diffusion_setting.lora.path = ""

                    # Add a default "None" option so users can unload LoRAs
                    lora_choices = ["None"] + list(lora_models_map.keys())
                    lora_model = gr.Dropdown(
                        lora_choices,
                        label="LoRA model",
                        info="LoRA model weight to load (You can use LoRA models from Civitai or Hugging Face .safetensors format). Select 'None' to unload LoRAs.",
                        value=(valid_model if valid_model != "" else "None"),
                        interactive=True,
                    )

                    lora_weight = gr.Slider(
                        0.0,
                        1.0,
                        value=app_settings.settings.lcm_diffusion_setting.lora.weight,
                        step=0.05,
                        label="Initial Lora weight",
                        interactive=True,
                    )

                    # Status indicator showing whether saved LoRA is enabled
                    lora_status = gr.Markdown(
                        f"**LoRA enabled:** {app_settings.settings.lcm_diffusion_setting.lora.enabled}",
                        elem_id="lora_status",
                    )

                    # Place the Load button on its own row to improve layout
                    load_lora_btn = gr.Button(
                        "Load selected LoRA",
                        elem_id="load_lora_button",
                        scale=0,
                    )

                # New row for upload/default controls to avoid crowding the Load button
                with gr.Row():
                    upload_files = gr.File(
                        file_count="multiple",
                        label="Upload LoRA (.safetensors)",
                        interactive=True,
                    )
                    # Option to mark uploaded model as the default immediately
                    upload_set_default = gr.Checkbox(
                        label="Set uploaded as Default",
                        value=False,
                        interactive=True,
                    )
                    upload_btn = gr.Button("Upload LoRA(s)")
                    # Keep default action separate to reduce UI clutter
                    set_default_btn = gr.Button("Set selected as Default")

                with gr.Row():
                    gr.Markdown(
                        "## Loaded LoRA models",
                        show_label=False,
                    )
                    update_lora_weights_btn = gr.Button(
                        "Update LoRA weights",
                        elem_id="load_lora_button",
                        scale=0,
                    )

                global _MAX_LORA_WEIGHTS
                global _custom_lora_sliders
                global _custom_lora_names
                global _custom_lora_columns
                for i in range(0, _MAX_LORA_WEIGHTS):
                    new_row = gr.Column(visible=False)
                    _custom_lora_columns.append(new_row)
                    with new_row:
                        lora_name = gr.Markdown(
                            "Lora Name",
                            show_label=True,
                        )
                        lora_slider = gr.Slider(
                            0.0,
                            1.0,
                            step=0.05,
                            label="LoRA weight",
                            interactive=True,
                            visible=True,
                        )

                        _custom_lora_names.append(lora_name)
                        _custom_lora_sliders.append(lora_slider)

    # The load handler updates the dynamic LoRA rows plus the status element
    load_lora_btn.click(
        fn=on_click_load_lora,
        inputs=[lora_model, lora_weight],
        outputs=[
            *_custom_lora_names,
            *_custom_lora_sliders,
            *_custom_lora_columns,
            lora_status,
        ],
    )

    def _on_upload_lora(*args, **kwargs):
        """Save uploaded safetensors into the LoRA models directory and refresh list.

        Signature tolerant wrapper: accepts either `(files)` or `(files, set_default)`.
        If `set_default` is True, also set the first uploaded file as the default
        LoRA and enable it in settings.
        """
        # Accept both positional and keyword call signatures for robustness
        if len(args) > 0:
            files = args[0]
            set_default = args[1] if len(args) > 1 else kwargs.get("set_default", False)
        else:
            files = kwargs.get("files")
            set_default = kwargs.get("set_default", False)
        if not files:
            return gr.Dropdown.update(), gr.Markdown.update(value="No files uploaded")
        dest_dir = app_settings.settings.lcm_diffusion_setting.lora.models_dir
        saved = []
        try:
            os.makedirs(dest_dir, exist_ok=True)
            # files may be a list of temp file dicts or a single file
            file_list = files if isinstance(files, list) else [files]
            for f in file_list:
                # Gradio file object can be a dict-like or have .name and .file
                try:
                    fname = f.name
                    fobj = f.file
                except Exception:
                    # fallback for older gradio versions
                    fname = os.path.basename(f["name"]) if isinstance(f, dict) else getattr(f, "name", "")
                    fobj = f["file"] if isinstance(f, dict) else getattr(f, "file", None)
                if not fname.lower().endswith(".safetensors"):
                    continue
                dest = os.path.join(dest_dir, fname)
                # move/copy file
                try:
                    if hasattr(fobj, "name") and os.path.exists(fobj.name):
                        shutil.copyfile(fobj.name, dest)
                    else:
                        # fobj may be a file-like object
                        with open(dest, "wb") as out_f:
                            shutil.copyfileobj(fobj, out_f)
                    saved.append(fname)
                except Exception as ex:
                    print(f"Failed to save uploaded LoRA {fname}: {ex}")
            # rebuild model list
            lora_models_map = get_lora_models(dest_dir)
            lora_choices = ["None"] + list(lora_models_map.keys())
            # choose first saved as selected
            sel = None
            if len(saved) > 0 and saved[0] in lora_models_map:
                sel = saved[0]
            else:
                # try to keep current valid
                valid_model = get_valid_lora_model(
                    list(lora_models_map.values()),
                    app_settings.settings.lcm_diffusion_setting.lora.path,
                    dest_dir,
                )
                sel = valid_model if valid_model != "" else "None"

            status = f"Uploaded: {saved}"
            # If requested, set the uploaded model as default and enable it
            if set_default and sel and sel in lora_models_map:
                try:
                    app_settings.settings.lcm_diffusion_setting.lora.path = lora_models_map[sel]
                    app_settings.settings.lcm_diffusion_setting.lora.enabled = True
                    try:
                        get_settings().save()
                    except Exception:
                        pass
                    status += f"; Default set to: {sel} (enabled)"
                except Exception as ex:
                    status += f"; Failed to set default: {ex}"

            return gr.Dropdown.update(choices=lora_choices, value=sel), gr.Markdown.update(value=status)
        except Exception as ex:
            return gr.Dropdown.update(), gr.Markdown.update(value=f"Upload failed: {ex}")

    def _on_set_default(model_name):
        # Set the selected model as default and enable it for future runs.
        if not model_name or model_name == "None":
            app_settings.settings.lcm_diffusion_setting.lora.path = ""
            app_settings.settings.lcm_diffusion_setting.lora.enabled = False
            try:
                get_settings().save()
            except Exception:
                pass
            return gr.Markdown.update(value="Default LoRA cleared (LoRA disabled)")
        lora_models_map = get_lora_models(app_settings.settings.lcm_diffusion_setting.lora.models_dir)
        if model_name not in lora_models_map:
            return gr.Markdown.update(value="Model not found to set as default")
        app_settings.settings.lcm_diffusion_setting.lora.path = lora_models_map[model_name]
        app_settings.settings.lcm_diffusion_setting.lora.enabled = True
        try:
            get_settings().save()
        except Exception:
            pass
        return gr.Markdown.update(value=f"Default LoRA set to: {model_name} (enabled)")

    upload_btn.click(
        fn=_on_upload_lora,
        inputs=[upload_files, upload_set_default],
        outputs=[lora_model, lora_status],
    )

    set_default_btn.click(
        fn=_on_set_default,
        inputs=[lora_model],
        outputs=[lora_status],
    )

    update_lora_weights_btn.click(
        fn=on_click_update_weight,
        inputs=[*_custom_lora_sliders],
        outputs=None,
    )
