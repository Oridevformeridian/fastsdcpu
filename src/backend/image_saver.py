import json
from os import path, mkdir
from typing import Any
from uuid import uuid4
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
from utils import get_image_file_extension


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
                image_ids.append(image_file_name)
                image_path = path.join(out_path, image_file_name)
                # Save to a temp file first, fsync the file, then atomically replace
                # the final filename. This avoids a race where a reader may observe
                # a truncated/zero-length file while PIL is writing.
                try:
                    import os
                    temp_name = f".{image_file_name}.tmp"
                    temp_path = path.join(out_path, temp_name)
                    # Ensure any existing temp is removed
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception:
                        pass

                    # Use PIL to save to the temp path
                    image.save(temp_path, quality=jpeg_quality)

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

                    # Atomically move temp into final path
                    try:
                        os.replace(temp_path, image_path)
                    except Exception:
                        # fallback to rename
                        try:
                            os.rename(temp_path, image_path)
                        except Exception:
                            pass

                    # Sync directory metadata so new file is visible to readers
                    try:
                        dir_fd = os.open(out_path, os.O_RDONLY)
                        os.fsync(dir_fd)
                        os.close(dir_fd)
                    except Exception:
                        pass
                except Exception:
                    # If anything goes wrong, attempt a best-effort save to final path
                    try:
                        image.save(image_path, quality=jpeg_quality)
                    except Exception:
                        pass
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
                    import os
                    dir_fd = os.open(out_path, os.O_RDONLY)
                    os.fsync(dir_fd)
                    os.close(dir_fd)
                except Exception:
                    pass  # Best effort
        return image_ids
            
