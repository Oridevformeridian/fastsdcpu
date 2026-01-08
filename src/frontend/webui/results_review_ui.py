import os
import time
import json
import gradio as gr
from PIL import Image
from state import get_settings
from backend.reviews_db import get_review, set_review, delete_review, init_db


app_settings = get_settings()


def _list_results_paths():
    path = app_settings.settings.generated_images.path
    if not path:
        from paths import FastStableDiffusionPaths

        path = FastStableDiffusionPaths.get_results_path()

    if not os.path.exists(path):
        return []

    entries = [e for e in os.listdir(path) if os.path.isfile(os.path.join(path, e))]
    entries.sort(key=lambda e: os.stat(os.path.join(path, e)).st_mtime, reverse=True)
    return [os.path.join(path, e) for e in entries]


def get_results_review_ui():
    with gr.Blocks():
        with gr.Row():
            refresh = gr.Button("Refresh")
            files_gallery = gr.Gallery(label="Generated results", columns=3, height=240)

        status_area = gr.Markdown("")

        # create per-file cards
        paths = _list_results_paths()
        for p in paths:
            name = os.path.basename(p)
            stat = os.stat(p)
            m = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            # read sidecar json
            prompt_val = ""
            model_val = ""
            json_path = os.path.join(os.path.dirname(p), os.path.splitext(name)[0] + ".json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        prompt_val = data.get("prompt", "")
                        model_val = data.get("model", "") or data.get("openvino_model", "")
                except Exception:
                    pass

            db_path = os.path.join(os.path.dirname(p), "reviews.db")
            init_db(db_path)
            entry = get_review(db_path, name)
            status_val = entry.get("status") if entry else "pending"
            note_val = entry.get("note") if entry else ""

            with gr.Row(variant="panel"):
                img = gr.Image(value=p, type="filepath", interactive=False)
                with gr.Column():
                    name_tb = gr.Textbox(value=name, label="File", interactive=False)
                    mtime_tb = gr.Textbox(value=m, label="Modified", interactive=False)
                    prompt_tb = gr.Textbox(value=prompt_val, label="Prompt", interactive=False, lines=2)
                    model_tb = gr.Textbox(value=model_val, label="Model", interactive=False)
                    status_radio = gr.Radio(["pending", "approved", "rejected"], value=status_val, label="Status")
                    note_tb = gr.Textbox(value=note_val, label="Note", lines=2)
                    path_state = gr.State(value=p)
                    approve_btn = gr.Button("Approve")
                    reject_btn = gr.Button("Reject")
                    use_img2img_btn = gr.Button("Use in Image to Image")
                    use_var_btn = gr.Button("Use in Image Variations")
                    clear_btn = gr.Button("Clear")

                    def _approve(path, note_text):
                        name = os.path.basename(path)
                        dbp = os.path.join(os.path.dirname(path), "reviews.db")
                        init_db(dbp)
                        set_review(dbp, name, "approved", note_text)
                        return "approved", note_text, f"Approved {name}"

                    def _reject(path, note_text):
                        name = os.path.basename(path)
                        dbp = os.path.join(os.path.dirname(path), "reviews.db")
                        init_db(dbp)
                        set_review(dbp, name, "rejected", note_text)
                        return "rejected", note_text, f"Rejected {name}"

                    def _clear_row(path):
                        name = os.path.basename(path)
                        dbp = os.path.join(os.path.dirname(path), "reviews.db")
                        init_db(dbp)
                        delete_review(dbp, name)
                        return "pending", "", f"Cleared {name}"

                    def _use_img2img(path):
                        try:
                            pil = Image.open(path).convert("RGB")
                            app_settings.settings.lcm_diffusion_setting.init_image = pil
                            return f"Set {os.path.basename(path)} as init image"
                        except Exception:
                            return "(failed to load image)"

                    def _use_variations(path):
                        try:
                            pil = Image.open(path).convert("RGB")
                            app_settings.settings.lcm_diffusion_setting.init_image = pil
                            app_settings.settings.lcm_diffusion_setting.diffusion_task = "image_variations"
                            return f"Set {os.path.basename(path)} for variations"
                        except Exception:
                            return "(failed)"

                    approve_btn.click(fn=_approve, inputs=[path_state, note_tb], outputs=[status_radio, note_tb, status_area])
                    reject_btn.click(fn=_reject, inputs=[path_state, note_tb], outputs=[status_radio, note_tb, status_area])
                    clear_btn.click(fn=_clear_row, inputs=[path_state], outputs=[status_radio, note_tb, status_area])
                    use_img2img_btn.click(fn=_use_img2img, inputs=[path_state], outputs=[status_area])
                    use_var_btn.click(fn=_use_variations, inputs=[path_state], outputs=[status_area])

        # wire refresh to update the gallery
        refresh.click(fn=_list_results_paths, inputs=None, outputs=[files_gallery])
