#!/usr/bin/env python3
"""Simple test script to POST a sample LCMDiffusionSetting to /api/queue

Usage:
  python scripts/test_enqueue.py --url http://127.0.0.1:8000
"""
import argparse
import json
import urllib.request
import urllib.error


SAMPLE = {
    "lcm_model_id": "Lykon/dreamshaper-8",
    "prompt": "A fantasy landscape with mountains and a river",
    "negative_prompt": "lowres, bad anatomy",
    "diffusion_task": "text_to_image",
    "image_width": 512,
    "image_height": 512,
    "inference_steps": 10,
    "guidance_scale": 7.5,
    "number_of_images": 1,
    "seed": 42,
}


def post_queue(url: str, payload: dict):
    full = url.rstrip("/") + "/api/queue"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(full, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
            print("Response:", json.dumps(data, indent=2))
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        print(f"HTTPError {e.code}: {body}")
    except Exception as e:
        print("Error:", e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    p.add_argument("--sample", action="store_true", help="Print sample payload and exit")
    args = p.parse_args()
    if args.sample:
        print(json.dumps(SAMPLE, indent=2))
        return
    print(f"Posting sample job to {args.url}/api/queue")
    post_queue(args.url, SAMPLE)


if __name__ == "__main__":
    main()
