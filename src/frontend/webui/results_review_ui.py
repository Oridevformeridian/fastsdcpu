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
            files_gallery = gr.Gallery(label="Generated results", columns=3, height=300)

        # tabular summary of generated files with metadata
        with gr.Row():
            files_table = gr.Dataframe(headers=["name", "mtime", "size", "prompt", "model", "status"], interactive=False)

        with gr.Row():
            filename = gr.Textbox(label="Filename", interactive=False)
            mtime = gr.Textbox(label="Modified", interactive=False)
            metadata = gr.Textbox(label="Metadata", interactive=False, lines=4)
        with gr.Row():
            status = gr.Radio(["pending", "approved", "rejected"], label="Review status")
            note = gr.Textbox(label="Note", lines=2)
        with gr.Row():
            approve_btn = gr.Button("Approve")
            reject_btn = gr.Button("Reject")
            clear_btn = gr.Button("Clear")
            use_img2img_btn = gr.Button("Use in Image to Image")
            use_variations_btn = gr.Button("Use in Image Variations")

    def _load_files():
        paths = _list_results_paths()
        # build table rows: name, mtime, size, prompt, model, status
        table_rows = []
        for p in paths:
            name = os.path.basename(p)
            stat = os.stat(p)
            m = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            size = stat.st_size
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
            table_rows.append([name, m, size, prompt_val, model_val, status_val])
        # outputs: gallery paths and table rows
        return paths, table_rows

    def _select_file(img_path):
        if not img_path:
            return "", "", "", "pending", ""
        name = os.path.basename(img_path)
        stat = os.stat(img_path)
        m = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        db_path = os.path.join(os.path.dirname(img_path), "reviews.db")
        init_db(db_path)
        entry = get_review(db_path, name)

        # try to read sidecar json metadata file (same basename + .json)
        meta_text = ""
        json_path = os.path.join(os.path.dirname(img_path), os.path.splitext(name)[0] + ".json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # display some common keys if present
                    parts = []
                    for k in ("prompt", "negative_prompt", "model", "settings"):
                        if k in data:
                            parts.append(f"{k}: {data[k]}")
                    if not parts:
                        meta_text = json.dumps(data)
                    else:
                        meta_text = " | ".join(parts)
            except Exception:
                meta_text = "(error reading metadata)"

        status_val = entry.get("status") if entry else "pending"
        note_val = entry.get("note") if entry else ""
        return name, m, meta_text, status_val, note_val

    def _set_status(img_path, new_status, note_text):
        if not img_path:
            return
        name = os.path.basename(img_path)
        db_path = os.path.join(os.path.dirname(img_path), "reviews.db")
        init_db(db_path)
        set_review(db_path, name, new_status, note_text)
        return _load_files()


    def _use_in_img2img(img_path):
        if not img_path:
            return _load_files()
        try:
            pil = Image.open(img_path).convert("RGB")
            app_settings.settings.lcm_diffusion_setting.init_image = pil
        except Exception:
            pass
        return _load_files()

    def _clear(img_path):
        if not img_path:
            return _load_files(), "", "", "", "pending", ""
        name = os.path.basename(img_path)
        db_path = os.path.join(os.path.dirname(img_path), "reviews.db")
        init_db(db_path)
        delete_review(db_path, name)
        # return gallery+table and clear fields
        paths, table = _load_files()
        return paths, table, "", "", "", "pending", ""

    refresh.click(fn=_load_files, inputs=None, outputs=[files_gallery, files_table])
    files_gallery.select(fn=_select_file, inputs=files_gallery, outputs=[filename, mtime, metadata, status, note])
    approve_btn.click(lambda p, n: _set_status(p, "approved", n), inputs=[files_gallery, note], outputs=[files_gallery, files_table])
    reject_btn.click(lambda p, n: _set_status(p, "rejected", n), inputs=[files_gallery, note], outputs=[files_gallery, files_table])
    clear_btn.click(fn=_clear, inputs=files_gallery, outputs=[files_gallery, files_table, filename, mtime, metadata, status, note])
    use_img2img_btn.click(fn=_use_in_img2img, inputs=files_gallery, outputs=[files_gallery, files_table])
    use_variations_btn.click(fn=lambda p: (_use_in_img2img(p)[0], _use_in_img2img(p)[1]), inputs=files_gallery, outputs=[files_gallery, files_table])
