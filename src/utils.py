from os import path, listdir
import platform
import os
import time
import logging
from typing import List

logger = logging.getLogger(__name__)


def show_system_info():
    try:
        print(f"Running on {platform.system()} platform")
        print(f"OS: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
    except Exception as ex:
        print(f"Error occurred while getting system information {ex}")


def get_models_from_text_file(file_path: str) -> List:
    models = []
    with open(file_path, "r") as file:
        lines = file.readlines()
    for repo_id in lines:
        if repo_id.strip() != "":
            models.append(repo_id.strip())
    return models


def get_image_file_extension(image_format: str) -> str:
    if image_format == "JPEG":
        return ".jpg"
    elif image_format == "PNG":
        return ".png"


def get_files_in_dir(root_dir: str) -> List:
    models = []
    models.append("None")
    for file in listdir(root_dir):
        if file.endswith((".gguf", ".safetensors")):
            models.append(path.join(root_dir, file))
    return models


def atomic_save_image(image, dest_path: str, jpeg_quality: int = 90, save_kwargs: dict | None = None, max_attempts: int = 3) -> bool:
    """
    Atomically save a PIL Image to disk with validation.

    - Writes to a hidden temp file, fsyncs, validates magic header and size,
      then atomically replaces the final file.
    - Returns True on success, False otherwise.
    """
    if save_kwargs is None:
        save_kwargs = {}

    out_dir = os.path.dirname(dest_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(dest_path)
    temp_name = f".{base}.tmp"
    temp_path = os.path.join(out_dir, temp_name)

    # Only provide JPEG quality for jpg/jpeg extensions
    ext = os.path.splitext(dest_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        save_kwargs.setdefault("quality", jpeg_quality)

    attempts = 0
    saved_ok = False
    while attempts < max_attempts and not saved_ok:
        attempts += 1
        try:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            image.save(temp_path, **save_kwargs)
        except Exception as e:
            logger.error("atomic save attempt %d failed for %s: %s", attempts, dest_path, e)
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            time.sleep(0.1)
            continue

        # Flush and sync temp file
        try:
            with open(temp_path, "rb") as tf:
                tf.flush()
                try:
                    os.fsync(tf.fileno())
                except Exception:
                    pass
        except Exception:
            pass

        # Quick validation: non-empty and magic header (PNG/JPEG/GIF)
        try:
            stat_tmp = os.stat(temp_path)
            if stat_tmp.st_size > 16:
                with open(temp_path, "rb") as hf:
                    prefix = hf.read(8)
                if (
                    prefix.startswith(b"\x89PNG\r\n\x1a\n")
                    or prefix.startswith(b"\xff\xd8")
                    or prefix.startswith(b"GIF89a")
                    or prefix.startswith(b"GIF87a")
                ):
                    saved_ok = True
                    break
        except Exception:
            pass

        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        time.sleep(0.1)

    if not saved_ok:
        logger.error("atomic_save_image failed to produce valid temp file for %s after %d attempts", dest_path, max_attempts)
        return False

    # Promote to final filename
    try:
        os.replace(temp_path, dest_path)
    except Exception:
        try:
            os.rename(temp_path, dest_path)
        except Exception as e:
            logger.error("atomic promotion failed for %s: %s", dest_path, e)
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
            except Exception:
                pass
            return False

    # Sync directory so file becomes visible
    try:
        dir_fd = os.open(out_dir, os.O_RDONLY)
        os.fsync(dir_fd)
        os.close(dir_fd)
    except Exception:
        pass

    return True
