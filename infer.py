#!/usr/bin/env python3
"""Generate views end-to-end from raw images, or reuse a prepared VGGT case."""
import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import torch
from PIL import Image, ImageDraw, ImageOps

from viewweaver.inputs import case_file, load_case, load_raw_case, read_rgb
from viewweaver.preprocessing import add_preparation_arguments, preparation_options


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--mode", choices=("end-to-end", "prepared"), default="end-to-end")
    parser.add_argument("--prepared-dir", type=Path, help="End-to-end cache location; defaults to a new timestamp directory")
    add_preparation_arguments(parser)
    parser.add_argument("--base-model", type=Path, default=Path("checkpoints/flux-kontext"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/viewweaver"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--prompt", help="Override the prompt stored in each case")
    parser.add_argument("--seed", type=int, help="Override each case's seed")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--save-comparison", action=argparse.BooleanOptionalAction, default=True,
                        help="Save reference–target comparisons (enabled by default)")
    parser.add_argument("--check-inputs", action="store_true", help="Validate cases without loading models")
    args = parser.parse_args(argv)
    if min(args.steps, args.top_k) < 1 or min(args.width, args.height) < 32:
        parser.error("steps/top-k must be positive; width/height must be at least 32")
    if args.width % 16 or args.height % 16:
        parser.error("width and height must be multiples of 16")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("prompt cannot be empty")
    return args


def save_comparison(references, target, path):
    """Keep the target at full resolution below labeled reference thumbnails."""
    if not references:
        raise ValueError("Comparison requires reference images")
    width = max(target.width, 128 * len(references))
    cell = width // len(references)
    target_y = cell + 48
    canvas = Image.new("RGB", (width, target_y + target.height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, reference in enumerate(references):
        thumbnail = ImageOps.contain(reference, (cell, cell), Image.Resampling.LANCZOS)
        canvas.paste(thumbnail, (index * cell + (cell - thumbnail.width) // 2,
                                 24 + (cell - thumbnail.height) // 2))
        draw.text((index * cell + 4, 4), f"Reference {index}", fill="black")
    draw.text((4, cell + 28), "Target", fill="black")
    canvas.paste(target, ((width - target.width) // 2, target_y))
    canvas.save(path)


@torch.inference_mode()
def main():
    args = parse_args()
    raw_path = args.case
    options = preparation_options(args)
    if args.mode == "end-to-end":
        _, images, _ = load_raw_case(args.case, expected_views=args.input_views)
        print(f"Validated raw case {args.case.name}: {len(images)} reference images", flush=True)
        if args.check_inputs:
            return
        from viewweaver.preprocessing import prepare_one_case
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        prepared_dir = args.prepared_dir or args.output_dir / "prepared" / run_id
        args.case = prepare_one_case(args.case, prepared_dir, args.vggt_model, options)
    case_path = args.case
    metadata, features, source = load_case(case_path)
    if features.shape[0] != args.input_views:
        raise ValueError(f"Expected {args.input_views} input views, found {features.shape[0]}")
    references = []
    if args.save_comparison:
        paths = metadata.get("references")
        if not isinstance(paths, list) or len(paths) != args.input_views or not all(isinstance(p, str) for p in paths):
            raise ValueError("Comparison requires one reference image path per input view in case.json")
        references = [read_rgb(case_file(case_path, path)) for path in paths]
    if not (case_path / "geometry.npz").is_file():
        raise FileNotFoundError(case_path / "geometry.npz")
    print(f"Validated {case_path.name}: {features.shape[0]} references, {len(options['azimuth'])} requested targets", flush=True)
    if args.check_inputs:
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Inference requires a CUDA GPU with bfloat16 support")
    start = time.monotonic()
    seed = args.seed if args.seed is not None else metadata.get("seed", 123)
    prompt = args.prompt if args.prompt is not None else metadata["prompt"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir / case_path.resolve().name / run_id
    output.mkdir(parents=True, exist_ok=False)
    if args.mode == "prepared":
        from viewweaver.preprocessing import render_cached_case
        metadata["targets"], _ = render_cached_case(case_path, source, output, options)
    views = list(range(len(metadata["targets"])))
    from viewweaver.loading import load_pipeline
    pipeline = load_pipeline(args.base_model, args.checkpoint, top_k=args.top_k)
    features = features.to(device="cuda", dtype=torch.bfloat16)
    source = source.to(device="cuda", dtype=torch.bfloat16)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for index in views:
        target = metadata["targets"][index]
        render = read_rgb(target["image"]).resize((args.width, args.height), Image.Resampling.BICUBIC)
        # Each generated image uses its own target camera, not target 000's pose.
        pose = torch.from_numpy(target["w2c"]).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = pipeline(
                prompt=prompt, image=render, recon_feats=features, source_w2c=source,
                target_w2c=pose, height=args.height, width=args.width,
                num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                true_cfg_scale=1.0, generator=generator, max_sequence_length=512,
                _auto_resize=False, joint_attention_kwargs={"attention_mask": None},
            ).images[0]
        filename = f"{index:03d}.png"
        result.save(output / filename)
        if args.save_comparison:
            save_comparison(references, result, output / f"{index:03d}_comparison.png")
        print(f"Saved {output / filename}", flush=True)
    record = {
        "mode": args.mode, "raw_case": str(raw_path.resolve()) if args.mode == "end-to-end" else None,
        "case": str(case_path.resolve()), "prompt": prompt, "seed": seed,
        "base_model": str(args.base_model.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "steps": args.steps, "width": args.width, "height": args.height,
        "guidance_scale": args.guidance_scale, "true_cfg_scale": 1.0, "top_k": args.top_k,
        "views": views, "reference_views": features.shape[0],
        "comparison_images": [f"{i:03d}_comparison.png" for i in views] if args.save_comparison else [],
        "target_camera_options": options,
        "targets": [{"index": i, "azimuth": target["azimuth"], "elevation": target["elevation"],
                     "w2c": target["w2c"].tolist(), "intrinsic": target["intrinsic"],
                     "conditioning_image": str(target["image"].resolve())}
                    for i, target in enumerate(metadata["targets"])],
        "target_pose_mode": "per-view", "elapsed_seconds": round(time.monotonic() - start, 2),
        "case_sha256": hashlib.sha256((case_path / "case.json").read_bytes()).hexdigest(),
        "preparation": metadata.get("preparation"),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "diffusers", "transformers", "peft")},
    }
    (output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
