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
    PAGE_SIZE = 6

    with gr.Blocks():
        with gr.Row():
            refresh = gr.Button("Refresh")
            prev_btn = gr.Button("Prev")
            next_btn = gr.Button("Next")
            page_state = gr.State(value=0)
            files_gallery = gr.Gallery(label="Generated results", columns=3, height=240)

        status_area = gr.Markdown("")

        # create fixed slots for page items
        image_slots = []
        name_slots = []
        mtime_slots = []
        prompt_slots = []
        model_slots = []
        status_slots = []
        note_slots = []
        path_states = []

        for i in range(PAGE_SIZE):
            with gr.Row(variant="panel"):
                img = gr.Image(value=None, type="filepath", interactive=False)
                name_tb = gr.Textbox(value="", label="File", interactive=False)
                mtime_tb = gr.Textbox(value="", label="Modified", interactive=False)
                prompt_tb = gr.Textbox(value="", label="Prompt", interactive=False, lines=2)
                model_tb = gr.Textbox(value="", label="Model", interactive=False)
                status_radio = gr.Radio(["pending", "approved", "rejected"], value="pending", label="Status")
                note_tb = gr.Textbox(value="", label="Note", lines=2)
                path_state = gr.State(value="")
                approve_btn = gr.Button("Approve")
                reject_btn = gr.Button("Reject")
                use_img2img_btn = gr.Button("Use in Image to Image")
                use_var_btn = gr.Button("Use in Image Variations")
                clear_btn = gr.Button("Clear")

                image_slots.append(img)
                name_slots.append(name_tb)
                mtime_slots.append(mtime_tb)
                prompt_slots.append(prompt_tb)
                model_slots.append(model_tb)
                status_slots.append(status_radio)
                note_slots.append(note_tb)
                path_states.append(path_state)

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

        def _populate_page(page_index: int):
            paths = _list_results_paths()
            total = len(paths)
            start = page_index * PAGE_SIZE
            page_paths = []
            out_values = [page_index]
            # for each slot, compute outputs
            for i in range(PAGE_SIZE):
                idx = start + i
                if idx < total:
                    p = paths[idx]
                    page_paths.append(p)
                    name = os.path.basename(p)
                    stat = os.stat(p)
                    m = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                    json_path = os.path.join(os.path.dirname(p), os.path.splitext(name)[0] + ".json")
                    prompt_val = ""
                    model_val = ""
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
                    out_values.extend([p, name, m, prompt_val, model_val, status_val, note_val, p])
                else:
                    out_values.extend([None, "", "", "", "", "pending", "", ""]) 
            return (page_paths, ) + tuple(out_values)

        def _prev(page_index: int):
            new_page = max(0, page_index - 1)
            return _populate_page(new_page)

        def _next(page_index: int):
            paths = _list_results_paths()
            max_page = max(0, (len(paths) - 1) // PAGE_SIZE)
            new_page = min(max_page, page_index + 1)
            return _populate_page(new_page)

        # wire pagination controls: outputs are files_gallery, page_state + per-slot component values
        outputs = [files_gallery, page_state]
        for i in range(PAGE_SIZE):
            outputs.extend([image_slots[i], name_slots[i], mtime_slots[i], prompt_slots[i], model_slots[i], status_slots[i], note_slots[i], path_states[i]])
        refresh.click(fn=lambda: _populate_page(0), inputs=None, outputs=outputs)
        prev_btn.click(fn=_prev, inputs=[page_state], outputs=outputs)
        next_btn.click(fn=_next, inputs=[page_state], outputs=outputs)
