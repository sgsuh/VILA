"""Collect SmoothQuant activation scales for the NVILA vision tower (W8A8 in TinyChat).

Equivalent to `awq.quantize.get_smooth_scale` from mit-han-lab/llm-awq, adapted to the
current `llava.utils.media.extract_media`, which returns video frames under media["image"].
"""

import argparse
import os

import torch
from awq.quantize.smooth import get_act_scales

import llava
from llava.media import Image, Video
from llava.mm_utils import process_images
from llava.utils.media import extract_media

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--media", nargs="+", required=True, help="Calibration images/videos (paths or URLs)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = llava.load(args.model_path, devices=[0])
    del model.llm
    del model.mm_projector
    torch.cuda.empty_cache()
    model = model.cuda().eval()

    prompt = [Video(m) if m.lower().endswith(VIDEO_EXTENSIONS) else Image(m) for m in args.media]
    media = extract_media([{"from": "human", "value": prompt}], model.config)

    vision_tower = model.vision_tower
    dtype = next(vision_tower.parameters()).dtype
    frames = process_images(media["image"], vision_tower.image_processor, model.config)
    frames = frames.to(device="cuda", dtype=dtype)
    print(f"Collecting activation scales from {frames.shape[0]} frames")

    act_scales = get_act_scales(vision_tower.eval(), frames)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(act_scales, args.output)
    print(f"Saved {len(act_scales)} activation scales to {args.output}")


if __name__ == "__main__":
    main()
