import json
import time
import os
import logging
from os import path, mkdir
from typing import Any
from uuid import uuid4
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
from utils import get_image_file_extension

logger = logging.getLogger(__name__)


def get_exclude_keys():
    exclude_keys = {
        "init_image": True,
        "generated_images": True,
        "lora": {
            "models_dir": True,
            "path": True,
        },
        "dirs": True,
        "controlnet": {
            "adapter_path": True,
        },
    }
    return exclude_keys


class ImageSaver:
    @staticmethod
    def save_images(
        output_path: str,
        images: Any,
        folder_name: str = "",
        format: str = "PNG",
        jpeg_quality: int = 90,
        lcm_diffusion_setting: LCMDiffusionSetting = None,
    ) -> list[str]:
        gen_id = uuid4()
        image_ids = []

        if images:
            image_seeds = []

            for index, image in enumerate(images):

                image_seed = image.info.get('image_seed')
                if image_seed is not None:
                    image_seeds.append(image_seed)

                if not path.exists(output_path):
                    mkdir(output_path)

                if folder_name:
                    out_path = path.join(
                        output_path,
                        folder_name,
                    )
                else:
                    out_path = output_path

                if not path.exists(out_path):
                    mkdir(out_path)
                image_extension = get_image_file_extension(format)
                image_file_name = f"{gen_id}-{index+1}{image_extension}"
                image_path = path.join(out_path, image_file_name)
                # Save to a temp file first, fsync the file, then atomically replace
                # the final filename. This avoids a race where a reader may observe
                # a truncated/zero-length file while PIL is writing.
                try:
                    temp_name = f".{image_file_name}.tmp"
                    temp_path = path.join(out_path, temp_name)

                    # Ensure any existing temp is removed
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception:
                        pass

                    # Attempt to save and validate the temp file before promoting
                    attempts = 0
                    max_attempts = 3
                    saved_ok = False
                    while attempts < max_attempts and not saved_ok:
                        attempts += 1
                        try:
                            # When possible, pass format to save to avoid ambiguity
                            save_kwargs = {}
                            try:
                                # prefer using extension-derived format if available
                                fmt = format
                                if fmt:
                                    save_kwargs["format"] = fmt
                            except Exception:
                                pass
                            # JPEG quality only when appropriate
                            if str(image_extension).lower() in (".jpg", ".jpeg"):
                                save_kwargs.setdefault("quality", jpeg_quality)

                            image.save(temp_path, **save_kwargs)
                        except Exception as e:
                            logger.exception("[ImageSaver] save attempt %d failed for %s: %s", attempts, image_file_name, e)
                            try:
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                            except Exception:
                                pass
                            time.sleep(0.1)
                            continue

                        # Flush and sync temp file to disk
                        try:
                            with open(temp_path, "rb") as tf:
                                tf.flush()
                                try:
                                    os.fsync(tf.fileno())
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Quick validation: check file non-empty and magic header
                        try:
                            stat_tmp = os.stat(temp_path)
                            if stat_tmp.st_size > 16:
                                with open(temp_path, "rb") as hf:
                                    prefix = hf.read(8)
                                # PNG signature or JPEG SOI
                                if prefix.startswith(b"\x89PNG\r\n\x1a\n") or prefix.startswith(b"\xff\xd8") or prefix.startswith(b"GIF89a") or prefix.startswith(b"GIF87a"):
                                    saved_ok = True
                                    break
                                else:
                                    logger.debug("[ImageSaver] temp file %s header mismatch: %r", temp_path, prefix)
                            else:
                                logger.debug("[ImageSaver] temp file %s too small: %d bytes", temp_path, stat_tmp.st_size)
                        except Exception:
                            logger.exception("[ImageSaver] error validating temp file %s", temp_path)
                        except Exception:
                            pass

                        # If validation failed, remove and retry
                        try:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        except Exception:
                            pass
                        time.sleep(0.1)

                    if not saved_ok:
                        # Log some filesystem diagnostics
                        try:
                            statv = os.statvfs(out_path)
                            free_bytes = statv.f_bavail * statv.f_frsize
                        except Exception:
                            free_bytes = None
                        logger.error("failed to produce valid temp file for %s after %d attempts; free_bytes=%s", image_file_name, max_attempts, free_bytes)

                    # Atomically move temp into final path if present (only if validated)
                    try:
                        if saved_ok and os.path.exists(temp_path):
                            os.replace(temp_path, image_path)
                        else:
                            # temp missing or not validated; attempt best-effort direct save
                            try:
                                image.save(image_path, quality=jpeg_quality)
                            except Exception as e:
                                logger.error("direct save failed for %s: %s", image_file_name, e)
                                # ensure no invalid file left
                                try:
                                    if os.path.exists(image_path):
                                        os.remove(image_path)
                                except Exception:
                                    pass
                                raise
                    except Exception:
                        # fallback to rename or direct save
                        try:
                            if saved_ok and os.path.exists(temp_path):
                                os.rename(temp_path, image_path)
                            else:
                                image.save(image_path, quality=jpeg_quality)
                        except Exception as e:
                            logger.error("promotion failed for %s: %s", image_file_name, e)
                            # ensure no invalid file left
                            try:
                                if os.path.exists(image_path):
                                    os.remove(image_path)
                            except Exception:
                                pass

                    # Sync directory metadata so new file is visible to readers
                    try:
                        dir_fd = os.open(out_path, os.O_RDONLY)
                        os.fsync(dir_fd)
                        os.close(dir_fd)
                    except Exception:
                        pass
                except Exception as e:
                    logger.exception("unexpected error while saving %s: %s", image_file_name, e)

                # Validate final file exists and looks reasonable before reporting success
                try:
                    if os.path.exists(image_path):
                        stat_final = os.stat(image_path)
                        if stat_final.st_size > 16:
                            # basic magic header check
                            try:
                                with open(image_path, "rb") as fh:
                                    prefix = fh.read(8)
                                if prefix.startswith(b"\x89PNG\r\n\x1a\n") or prefix.startswith(b"\xff\xd8"):
                                    image_ids.append(image_file_name)
                                else:
                                    logger.error("saved file %s has invalid header, removing", image_path)
                                    try:
                                        os.remove(image_path)
                                    except Exception:
                                        pass
                            except Exception:
                                logger.exception("error validating saved file %s", image_path)
                        else:
                            logger.error("saved file %s too small (%d bytes), removing", image_path, stat_final.st_size)
                            try:
                                os.remove(image_path)
                            except Exception:
                                pass
                    else:
                        logger.error("final image file missing after save attempts: %s", image_path)
                except Exception:
                    logger.exception("error checking final file %s", image_path)

            if lcm_diffusion_setting:
                data = lcm_diffusion_setting.model_dump(exclude=get_exclude_keys())
                if image_seeds:
                    data['image_seeds'] = image_seeds
                json_path = path.join(out_path, f"{gen_id}.json")
                with open(json_path, "w") as json_file:
                    json.dump(
                        data,
                        json_file,
                        indent=4,
                    )
                    json_file.flush()
                    os.fsync(json_file.fileno())
                # Additional sync to ensure directory metadata is updated
                try:
                    dir_fd = os.open(out_path, os.O_RDONLY)
                    os.fsync(dir_fd)
                    os.close(dir_fd)
                except Exception:
                    pass  # Best effort
        return image_ids
            
