import os
import time
import json
import gradio as gr
from PIL import Image
from state import get_settings
import urllib.request
import urllib.parse
import os

API_BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")  # default to API server

def _api_get(path: str, params: dict = None):
    if not API_BASE:
        return None
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

    entries = [
        e for e in os.listdir(path) 
        if os.path.isfile(os.path.join(path, e))
        and (e.lower().endswith('.jpg') or e.lower().endswith('.png') or e.lower().endswith('.jpeg'))
    ]
    entries.sort(key=lambda e: os.stat(os.path.join(path, e)).st_mtime, reverse=True)
    return [os.path.join(path, e) for e in entries]


def get_results_review_ui():
    PAGE_SIZE = 6

    with gr.Blocks() as results_block:
        # Top navigation
        with gr.Row():
            prev_btn_top = gr.Button("←", scale=0, min_width=50)
            page_indicator = gr.Markdown("Page 1")
            next_btn_top = gr.Button("→", scale=0, min_width=50)
            
        # Preview gallery at top
        files_gallery = gr.Gallery(label="Generated results", columns=3, height=240)
        
        status_area = gr.Markdown("")
        page_state = gr.State(value=0)
        
        # Hidden timer for auto-refresh every 10 seconds
        timer = gr.Timer(value=10, active=True)

        # create fixed slots for page items (image, filename, modified, prompt, model, actions)
        image_slots = []
        name_slots = []
        mtime_slots = []
        prompt_slots = []
        model_slots = []
        path_states = []

        for i in range(PAGE_SIZE):
            with gr.Row(variant="panel"):
                img = gr.Image(value=None, type="filepath", interactive=False)
                name_tb = gr.Textbox(value="", label="File", interactive=False)
                mtime_tb = gr.Textbox(value="", label="Modified", interactive=False)
                prompt_tb = gr.Textbox(value="", label="Prompt", interactive=False, lines=2)
                model_tb = gr.Textbox(value="", label="Model", interactive=False)
                path_state = gr.State(value="")
                # action buttons
                use_img2img_btn = gr.Button("Use in Image to Image")
                use_var_btn = gr.Button("Use in Image Variations")
                regen_btn = gr.Button("Regenerate")
                show_json_btn = gr.Button("Show JSON")

                image_slots.append(img)
                name_slots.append(name_tb)
                mtime_slots.append(mtime_tb)
                prompt_slots.append(prompt_tb)
                model_slots.append(model_tb)
                path_states.append(path_state)

                def _use_img2img(path):
                    try:
                        name = os.path.basename(path)
                        url = API_BASE.rstrip("/") + f"/results/{urllib.parse.quote(name)}"
                        pil = Image.open(url).convert("RGB")
                        app_settings.settings.lcm_diffusion_setting.init_image = pil
                        return f"Set {name} as init image"
                    except Exception:
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

                def _show_json(path):
                    if not path:
                        return "(no file)"
                    name = os.path.basename(path)
                    json_url = API_BASE.rstrip("/") + f"/results/{urllib.parse.quote(os.path.splitext(name)[0] + '.json')}"
                    try:
                        with urllib.request.urlopen(json_url, timeout=2) as f:
                            data = json.load(f)
                            pretty = json.dumps(data, indent=2)
                            return f"Loaded JSON for {name}:\n```json\n{pretty}\n```"
                    except Exception as e:
                        return f"failed to load json: {e}"

                def _regenerate(path):
                    if not path:
                        return "(no file)"
                    name = os.path.basename(path)
                    json_url = API_BASE.rstrip("/") + f"/results/{urllib.parse.quote(os.path.splitext(name)[0] + '.json')}"
                    payload = None
                    try:
                        with urllib.request.urlopen(json_url, timeout=2) as f:
                            payload = json.load(f)
                    except Exception:
                        payload = None

                    if not payload:
                        # try to construct minimal payload
                        payload = {"prompt": "", "diffusion_task": "text_to_image"}
                    # enqueue
                    resp = _api_post("/api/queue", payload)
                    if resp and resp.get("job_id"):
                        return f"Enqueued regenerate job {resp.get('job_id')}"
                    return "failed to enqueue regenerate"

                use_img2img_btn.click(fn=_use_img2img, inputs=[path_state], outputs=[status_area])
                use_var_btn.click(fn=_use_variations, inputs=[path_state], outputs=[status_area])
                show_json_btn.click(fn=_show_json, inputs=[path_state], outputs=[status_area])
                regen_btn.click(fn=_regenerate, inputs=[path_state], outputs=[status_area])

        # Bottom navigation (same as top)
        with gr.Row():
            prev_btn_bottom = gr.Button("←", scale=0, min_width=50)
            page_indicator_bottom = gr.Markdown("Page 1")
            next_btn_bottom = gr.Button("→", scale=0, min_width=50)

        def _populate_page(page_index: int):
            payload = _api_get("/api/results/paged", {"page": page_index, "size": PAGE_SIZE})
            if not payload:
                # fallback to local listing
                paths = _list_results_paths()
                total = len(paths)
                start = page_index * PAGE_SIZE
                page_paths = paths[start : start + PAGE_SIZE]
                # build out_values using minimal info
                page_text = f"Page {page_index + 1} of {max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)}"
                out = [page_paths, page_index, page_text, page_text]
                for i in range(PAGE_SIZE):
                    if i < len(page_paths):
                        p = page_paths[i]
                        name = os.path.basename(p)
                        stat = os.stat(p)
                        m = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                        out.extend([p, name, m, "", "", p])
                    else:
                        out.extend([None, "", "", "", "", ""])
                return tuple(out)

            page_paths = []
            total_results = payload.get("total", 0)
            page_text = f"Page {page_index + 1} of {max(1, (total_results + PAGE_SIZE - 1) // PAGE_SIZE)}"
            out = [page_index, page_text, page_text]
            for i in range(PAGE_SIZE):
                if i < len(payload.get("results", [])):
                    item = payload["results"][i]
                    name = item.get("name")
                    url = API_BASE.rstrip("/") + item.get("url")
                    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.get("mtime", 0)))
                    prompt_val = item.get("meta", {}).get("prompt", "")
                    model_val = item.get("meta", {}).get("model", "") or item.get("meta", {}).get("openvino_model", "")
                    page_paths.append(url)
                    out.extend([url, name, mtime, prompt_val, model_val, url])
                else:
                    out.extend([None, "", "", "", "", ""]) 
            return tuple([page_paths] + out)

        def _prev(page_index: int):
            new_page = max(0, page_index - 1)
            return _populate_page(new_page)

        def _next(page_index: int):
            paths = _list_results_paths()
            max_page = max(0, (len(paths) - 1) // PAGE_SIZE)
            new_page = min(max_page, page_index + 1)
            return _populate_page(new_page)

        # wire pagination controls: outputs are files_gallery, page_state, page_indicator (top & bottom) + per-slot component values
        outputs = [files_gallery, page_state, page_indicator, page_indicator_bottom]
        for i in range(PAGE_SIZE):
            outputs.extend([image_slots[i], name_slots[i], mtime_slots[i], prompt_slots[i], model_slots[i], path_states[i]])
        
        # Wire all navigation buttons
        prev_btn_top.click(fn=_prev, inputs=[page_state], outputs=outputs)
        next_btn_top.click(fn=_next, inputs=[page_state], outputs=outputs)
        prev_btn_bottom.click(fn=_prev, inputs=[page_state], outputs=outputs)
        next_btn_bottom.click(fn=_next, inputs=[page_state], outputs=outputs)
        
        # Auto-refresh on tab load
        results_block.load(fn=lambda: _populate_page(0), inputs=None, outputs=outputs)
        
        # Auto-refresh every 10 seconds via timer (refresh current page)
        timer.tick(fn=lambda p: _populate_page(p), inputs=[page_state], outputs=outputs)
    
    return results_block
