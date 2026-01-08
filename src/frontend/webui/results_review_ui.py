import os
import time
import gradio as gr
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

        with gr.Row():
            filename = gr.Textbox(label="Filename", interactive=False)
            mtime = gr.Textbox(label="Modified", interactive=False)
        with gr.Row():
            status = gr.Radio(["pending", "approved", "rejected"], label="Review status")
            note = gr.Textbox(label="Note", lines=2)
        with gr.Row():
            approve_btn = gr.Button("Approve")
            reject_btn = gr.Button("Reject")
            clear_btn = gr.Button("Clear")

    def _load_files():
        paths = _list_results_paths()
        # return file paths to gallery
        return paths

    def _select_file(img_path):
        if not img_path:
            return "", "", "pending", ""
        name = os.path.basename(img_path)
        stat = os.stat(img_path)
        m = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        db_path = os.path.join(os.path.dirname(img_path), "reviews.db")
        init_db(db_path)
        entry = get_review(db_path, name)
        if entry:
            return name, m, entry.get("status"), entry.get("note") or ""
        return name, m, "pending", ""

    def _set_status(img_path, new_status, note_text):
        if not img_path:
            return
        name = os.path.basename(img_path)
        db_path = os.path.join(os.path.dirname(img_path), "reviews.db")
        init_db(db_path)
        set_review(db_path, name, new_status, note_text)
        return _load_files()

    def _clear(img_path):
        if not img_path:
            return
        name = os.path.basename(img_path)
        db_path = os.path.join(os.path.dirname(img_path), "reviews.db")
        init_db(db_path)
        delete_review(db_path, name)
        return _load_files(), "", "", "pending", ""

    refresh.click(fn=lambda: _load_files(), inputs=None, outputs=files_gallery)
    files_gallery.select(fn=_select_file, inputs=files_gallery, outputs=[filename, mtime, status, note])
    approve_btn.click(lambda p, n: _set_status(p, "approved", n), inputs=[files_gallery, note], outputs=files_gallery)
    reject_btn.click(lambda p, n: _set_status(p, "rejected", n), inputs=[files_gallery, note], outputs=files_gallery)
    clear_btn.click(fn=_clear, inputs=files_gallery, outputs=[files_gallery, filename, mtime, status, note])
