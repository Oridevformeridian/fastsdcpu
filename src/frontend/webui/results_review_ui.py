import os
import time
import json
import gradio as gr
from PIL import Image
from state import get_settings
import urllib.request
import urllib.parse
from frontend.webui.connection_manager import get_connection_state

# Debug logging control
DEBUG_ENABLED = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")

API_BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")  # default to API server
conn_state = get_connection_state()

def _api_get(path: str, params: dict = None):
    if not API_BASE:
        return None
    url = API_BASE.rstrip("/") + path
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            result = json.load(resp)
            conn_state.mark_connected()
            return result
    except Exception:
        conn_state.mark_disconnected()
        return None

def _api_post(path: str, data: dict):
    url = API_BASE.rstrip("/") + path
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.load(resp)
            conn_state.mark_connected()
            return result
    except Exception:
        conn_state.mark_disconnected()
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

    def _is_valid_image(pth: str) -> bool:
        try:
            st = os.stat(pth)
            if st.st_size <= 16:
                if DEBUG_ENABLED:
                    print(f"[DEBUG-UI] skipping small file: {pth} ({st.st_size} bytes)")
                return False
            with open(pth, "rb") as fh:
                prefix = fh.read(8)
            if prefix.startswith(b"\x89PNG\r\n\x1a\n") or prefix.startswith(b"\xff\xd8"):
                return True
            if DEBUG_ENABLED:
                print(f"[DEBUG-UI] skipping invalid-header file: {pth}")
            return False
        except Exception:
            return False

    entries = [
        e for e in os.listdir(path)
        if os.path.isfile(os.path.join(path, e))
        and (e.lower().endswith('.jpg') or e.lower().endswith('.png') or e.lower().endswith('.jpeg'))
        and _is_valid_image(os.path.join(path, e))
    ]
    entries.sort(key=lambda e: os.stat(os.path.join(path, e)).st_mtime, reverse=True)
    return [os.path.join(path, e) for e in entries]


def get_results_review_ui():
    PAGE_SIZE = 6

    with gr.Blocks(css="""
        /* Desktop: show taller gallery (2.25x) and keep full width */
        @media (min-width: 768px) {
            #results-gallery-desktop-hide { display: block !important; height: 540px !important; }
        }
        /* Mobile: keep the original, smaller gallery height */
        @media (max-width: 767px) {
            #results-gallery-desktop-hide { display: block !important; height: 240px !important; }
        }
        .download-image-btn { cursor: pointer; }
    """) as results_block:
        # Top navigation with filter
        with gr.Row():
            prev_btn_top = gr.Button("←", scale=0, min_width=50)
            page_indicator = gr.Markdown("Page 1")
            next_btn_top = gr.Button("→", scale=0, min_width=50)
            reload_btn = gr.Button("🔄 Reload Images", size="sm")
            show_failed = gr.Checkbox(label="Show Failed Images", value=True)
            
        # Preview gallery at top (hidden on desktop)
        files_gallery = gr.Gallery(label="Generated results", columns=3, height=1040, elem_id="results-gallery-desktop-hide")
        
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
                with gr.Column(scale=2):
                    # Image as download button
                    img = gr.Image(value=None, type="filepath", interactive=False, height=300, show_download_button=True, show_label=False)
                
                with gr.Column(scale=3):
                    name_tb = gr.Textbox(value="", label="File", interactive=False)
                    mtime_tb = gr.Textbox(value="", label="Modified", interactive=False)
                    prompt_tb = gr.Textbox(value="", label="Prompt", interactive=False, lines=3)
                    model_tb = gr.Textbox(value="", label="Model", interactive=False)
                    
                    # Compact action buttons row
                    with gr.Row():
                        use_img2img_btn = gr.Button("📷 Img2Img", size="sm", scale=1)
                        use_var_btn = gr.Button("🔄 Variations", size="sm", scale=1)
                        regen_btn = gr.Button("♻️ Regen", size="sm", scale=1)
                        show_json_btn = gr.Button("{ } JSON", size="sm", scale=1)
                        archive_btn = gr.Button("📦 Archive", size="sm", scale=1)
                    
                    path_state = gr.State(value="")

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
                    name_without_ext = os.path.splitext(name)[0]
                    
                    # Remove the -N suffix if present (e.g., "uuid-1" -> "uuid")
                    # Images: uuid-1.png, uuid-2.png, etc.
                    # JSON: uuid.json (no index suffix)
                    if '-' in name_without_ext:
                        parts = name_without_ext.rsplit('-', 1)
                        # Only strip if last part is a number (the index)
                        if len(parts) == 2 and parts[1].isdigit():
                            name_without_ext = parts[0]
                    
                    json_url = API_BASE.rstrip("/") + f"/results/{urllib.parse.quote(name_without_ext + '.json')}"
                    try:
                        with urllib.request.urlopen(json_url, timeout=2) as f:
                            data = json.load(f)
                            pretty = json.dumps(data, indent=2)
                            return f"Loaded JSON for {name}:\n```json\n{pretty}\n```"
                    except Exception as e:
                        return f"failed to load json from {json_url}: {e}"

                def _regenerate(path):
                    if not path:
                        return "(no file)"
                    name = os.path.basename(path)
                    name_without_ext = os.path.splitext(name)[0]
                    
                    # Remove the -N suffix if present
                    if '-' in name_without_ext:
                        parts = name_without_ext.rsplit('-', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            name_without_ext = parts[0]
                    
                    json_url = API_BASE.rstrip("/") + f"/results/{urllib.parse.quote(name_without_ext + '.json')}"
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
                # Pass the filename textbox to _show_json (robust when path state is empty)
                show_json_btn.click(fn=_show_json, inputs=[name_tb], outputs=[status_area])
                regen_btn.click(fn=_regenerate, inputs=[path_state], outputs=[status_area])
                def _archive(path):
                    if not path:
                        return "(no file)"
                    name = os.path.basename(path)
                    api_path = f"/api/results/{urllib.parse.quote(name)}/archive"
                    resp = _api_post(api_path, {})
                    if resp and resp.get("archived"):
                        return f"Archived {name}"
                    return f"Failed to archive {name}"

                archive_btn.click(fn=_archive, inputs=[path_state], outputs=[status_area])

        # Bottom navigation (same as top)
        with gr.Row():
            prev_btn_bottom = gr.Button("←", scale=0, min_width=50)
            page_indicator_bottom = gr.Markdown("Page 1")
            next_btn_bottom = gr.Button("→", scale=0, min_width=50)

        def _populate_page(page_index: int, show_failed_filter: bool):
            try:
                try:
                    payload = _api_get("/api/results/paged", {"page": page_index, "size": PAGE_SIZE})
                except Exception:
                    payload = None
                    if DEBUG_ENABLED:
                        import traceback
                        print("[DEBUG-UI] exception calling _api_get:")
                        traceback.print_exc()

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

                # Get local file paths instead of URLs for gallery (Gradio doesn't like 127.0.0.1 URLs)
                results_path = app_settings.settings.generated_images.path
                if not results_path:
                    from paths import FastStableDiffusionPaths
                    results_path = FastStableDiffusionPaths.get_results_path()

                page_paths = []
                total_results = payload.get("total", 0)
                page_text = f"Page {page_index + 1} of {max(1, (total_results + PAGE_SIZE - 1) // PAGE_SIZE)}"
                out = [page_index, page_text, page_text]
                for i in range(PAGE_SIZE):
                    if i < len(payload.get("results", [])):
                        item = payload["results"][i]
                        name = item.get("name")
                        # Prefer cache-busted HTTP URL to force browser reloads
                        local_path = os.path.join(results_path, name)
                        file_exists = False
                        image_url = None
                        try:
                            if os.path.exists(local_path):
                                st = os.stat(local_path)
                                if st.st_size > 16:
                                    # quick header check
                                    try:
                                        with open(local_path, "rb") as fh:
                                            prefix = fh.read(8)
                                        if prefix.startswith(b"\x89PNG\r\n\x1a\n") or prefix.startswith(b"\xff\xd8"):
                                            file_exists = True
                                            image_url = local_path
                                        else:
                                            if DEBUG_ENABLED:
                                                print(f"[DEBUG-UI] skipping result with bad header: {local_path}")
                                    except Exception:
                                        if DEBUG_ENABLED:
                                            print(f"[DEBUG-UI] couldn't read header for: {local_path}")
                                else:
                                    if DEBUG_ENABLED:
                                        print(f"[DEBUG-UI] skipping result too small: {local_path} ({st.st_size} bytes)")
                        except Exception:
                            file_exists = False
                        if DEBUG_ENABLED:
                            print(f"[DEBUG-UI] Image {i}: name={name}, path={local_path}, exists={file_exists}")
                        
                        # Filter out missing files if show_failed is False
                        if not show_failed_filter and not file_exists:
                            out.extend([None, "", "", "", "", ""])
                            continue
                        
                        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.get("mtime", 0)))
                        prompt_val = item.get("meta", {}).get("prompt", "")
                        model_val = item.get("meta", {}).get("model", "") or item.get("meta", {}).get("openvino_model", "")
                        if DEBUG_ENABLED:
                            print(f"[DEBUG-UI]   prompt={prompt_val[:50] if prompt_val else 'EMPTY'}, model={model_val}")
                        page_paths.append(image_url if file_exists else None)
                        out.extend([image_url if file_exists else None, name, mtime, prompt_val, model_val, local_path])
                    else:
                        out.extend([None, "", "", "", "", ""]) 
                    # Gradio Gallery cannot accept None entries; filter them for the gallery view
                    gallery_paths = [p for p in page_paths if p]
                    return tuple([gallery_paths] + out)
            except Exception as e:
                # Catch-all to prevent Gradio from showing "Error" in the UI.
                if DEBUG_ENABLED:
                    import traceback
                    print("[DEBUG-UI] exception in _populate_page:")
                    traceback.print_exc()
                # Build an empty safe response matching the expected outputs
                page_paths = []
                page_text = f"Page {page_index + 1} of 1"
                safe = [page_paths, page_index, page_text, page_text]
                for _ in range(PAGE_SIZE):
                    safe.extend([None, "", "", "", "", ""])
                return tuple(safe)

        def _prev(page_index: int, show_failed_filter: bool):
            new_page = max(0, page_index - 1)
            return _populate_page(new_page, show_failed_filter)

        def _next(page_index: int, show_failed_filter: bool):
            paths = _list_results_paths()
            max_page = max(0, (len(paths) - 1) // PAGE_SIZE)
            new_page = min(max_page, page_index + 1)
            return _populate_page(new_page, show_failed_filter)

        # wire pagination controls: outputs are files_gallery, page_state, page_indicator (top & bottom) + per-slot component values
        outputs = [files_gallery, page_state, page_indicator, page_indicator_bottom]
        for i in range(PAGE_SIZE):
            outputs.extend([image_slots[i], name_slots[i], mtime_slots[i], prompt_slots[i], model_slots[i], path_states[i]])
        
        # Wire all navigation buttons
        prev_btn_top.click(fn=_prev, inputs=[page_state, show_failed], outputs=outputs)
        next_btn_top.click(fn=_next, inputs=[page_state, show_failed], outputs=outputs)
        prev_btn_bottom.click(fn=_prev, inputs=[page_state, show_failed], outputs=outputs)
        next_btn_bottom.click(fn=_next, inputs=[page_state, show_failed], outputs=outputs)
        
        # Wire filter toggle
        show_failed.change(fn=_populate_page, inputs=[page_state, show_failed], outputs=outputs)
        
        # Wire timer refresh
        timer.tick(fn=_populate_page, inputs=[page_state, show_failed], outputs=outputs)

        # Initialize on load
        results_block.load(fn=_populate_page, inputs=[page_state, show_failed], outputs=outputs)
    
    return results_block
