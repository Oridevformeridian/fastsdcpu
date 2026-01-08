import os
import time
import json
import gradio as gr
from PIL import Image
from state import get_settings
import urllib.request
import urllib.parse
import os

API_BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")

def _api_get(path: str, params: dict = None):
    url = API_BASE.rstrip("/") + path
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.load(resp)
    except Exception:
        return None

def _api_post(path: str, data: dict):
    url = API_BASE.rstrip("/") + path
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.load(resp)
    except Exception:
        return None

def _api_delete(path: str):
    url = API_BASE.rstrip("/") + path
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.load(resp)
    except Exception:
        return None


app_settings = get_settings()


def _list_results_paths():
    # not used when API-backed; kept for fallback
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
                    # call backend API to set review
                    if not path:
                        return "pending", note_text, "(no file)"
                    name = os.path.basename(path)
                    _api_post(f"/api/results/{urllib.parse.quote(name)}/review", {"status": "approved", "note": note_text})
                    return "approved", note_text, f"Approved {name}"

                def _reject(path, note_text):
                    if not path:
                        return "pending", note_text, "(no file)"
                    name = os.path.basename(path)
                    _api_post(f"/api/results/{urllib.parse.quote(name)}/review", {"status": "rejected", "note": note_text})
                    return "rejected", note_text, f"Rejected {name}"

                def _clear_row(path):
                    if not path:
                        return "pending", "", "(no file)"
                    name = os.path.basename(path)
                    _api_delete(f"/api/results/{urllib.parse.quote(name)}/review")
                    return "pending", "", f"Cleared {name}"

                def _use_img2img(path):
                    try:
                        # use backend-served URL so Gradio can load it anywhere
                        name = os.path.basename(path)
                        url = API_BASE.rstrip("/") + f"/results/{urllib.parse.quote(name)}"
                        pil = Image.open(url).convert("RGB")
                        app_settings.settings.lcm_diffusion_setting.init_image = pil
                        return f"Set {name} as init image"
                    except Exception:
                        # fallback: try local path
                        try:
                            pil = Image.open(path).convert("RGB")
                            app_settings.settings.lcm_diffusion_setting.init_image = pil
                            return f"Set {os.path.basename(path)} as init image"
                        except Exception:
                            return "(failed to load image)"

                def _use_variations(path):
                    try:
                        name = os.path.basename(path)
                        url = API_BASE.rstrip("/") + f"/results/{urllib.parse.quote(name)}"
                        pil = Image.open(url).convert("RGB")
                        app_settings.settings.lcm_diffusion_setting.init_image = pil
                        app_settings.settings.lcm_diffusion_setting.diffusion_task = "image_variations"
                        return f"Set {name} for variations"
                    except Exception:
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
            payload = _api_get("/api/results/paged", {"page": page_index, "size": PAGE_SIZE})
            if not payload:
                # fallback to local listing
                paths = _list_results_paths()
                total = len(paths)
                start = page_index * PAGE_SIZE
                page_paths = paths[start : start + PAGE_SIZE]
                # build out_values using minimal info
                out = [page_index]
                for i in range(PAGE_SIZE):
                    if i < len(page_paths):
                        p = page_paths[i]
                        name = os.path.basename(p)
                        stat = os.stat(p)
                        m = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                        out.extend([p, name, m, "", "", "pending", "", p])
                    else:
                        out.extend([None, "", "", "", "", "pending", "", ""])
                return (page_paths,) + tuple(out)

            page_paths = []
            out = [page_index]
            for i in range(PAGE_SIZE):
                if i < len(payload.get("results", [])):
                    item = payload["results"][i]
                    name = item.get("name")
                    url = API_BASE.rstrip("/") + item.get("url")
                    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.get("mtime", 0)))
                    prompt_val = item.get("meta", {}).get("prompt", "")
                    model_val = item.get("meta", {}).get("model", "") or item.get("meta", {}).get("openvino_model", "")
                    status_val = item.get("review", {}).get("status") if item.get("review") else "pending"
                    note_val = item.get("review", {}).get("note") if item.get("review") else ""
                    page_paths.append(url)
                    out.extend([url, name, mtime, prompt_val, model_val, status_val, note_val, url])
                else:
                    out.extend([None, "", "", "", "", "pending", "", ""]) 
            return (page_paths,) + tuple(out)

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
