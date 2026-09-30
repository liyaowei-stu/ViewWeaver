"""Run VGGT and render fresh conditioning from raw reference images."""
import gc
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .geometry import angle_pairs, normalize_scene, orbit_cameras, render_points, unproject_depth
from .inputs import load_raw_case, load_case


def add_preparation_arguments(parser):
    parser.add_argument("--vggt-model", type=Path, default=Path("checkpoints/vggt"))
    parser.add_argument("--input-views", type=int, default=8, help="Expected reference count; all references are used")
    parser.add_argument("--reference-view", type=int, default=0, help="Zero-based input index defining azimuth zero and up; all inputs still participate")
    parser.add_argument("--azimuth", type=float, nargs="+", default=[-15.0], help="Horizontal angles in degrees, relative to the selected reference camera")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--camera-distance", type=float, default=4.5, help="Distance in normalized object units; increase to make the subject smaller")
    parser.add_argument("--elevation", type=float, nargs="+", default=[15.0], help="Elevation degrees; positive looks down, singleton broadcasts")
    parser.add_argument("--fov", type=float, default=55.0, help="Horizontal field of view in degrees")
    parser.add_argument("--offset-x", "--principal-x", dest="principal_x", type=float, default=0.0,
                        help="Horizontal framing shift / image width; positive moves the subject right")
    parser.add_argument("--offset-y", "--principal-y", dest="principal_y", type=float, default=0.18,
                        help="Vertical framing shift / image height; positive moves the subject down")
    parser.add_argument("--point-radius", type=float, default=1.5, help="RGB splat radius in render pixels")
    parser.add_argument("--conf-percentile", type=float, default=20.0)
    parser.add_argument("--max-points", type=int, default=200000)


def preparation_options(args):
    names = ("input_views", "reference_view", "render_size", "camera_distance", "fov", "principal_x", "principal_y",
             "point_radius", "conf_percentile", "max_points")
    options = {name: getattr(args, name) for name in names}
    if not all(np.isfinite(value) for value in options.values()):
        raise ValueError("Preprocessing options must be finite")
    if args.input_views < 1 or args.render_size < 32 or args.max_points < 100:
        raise ValueError("input-views >= 1, render-size >= 32, max-points >= 100 are required")
    if not 0 <= args.reference_view < args.input_views:
        raise ValueError(f"reference-view must be between 0 and {args.input_views - 1}")
    if args.camera_distance <= 1 or not 5 <= args.fov <= 150:
        raise ValueError("Invalid orbit distance, elevation or field of view")
    if (not 0 <= args.conf_percentile < 100 or not 0 <= args.point_radius <= 8
            or not abs(args.principal_x) < 0.5 or not abs(args.principal_y) < 0.5):
        raise ValueError("Invalid confidence percentile, point radius or principal-point shift")
    options["azimuth"], options["elevation"] = angle_pairs(args.azimuth, args.elevation)
    return options


def preprocess_images(images, masks):
    rgb_tensors, mask_tensors = [], []
    for image, mask in zip(images, masks):
        width, height = image.size
        if width >= height:
            size = (518, max(14, round(height * 518 / width / 14) * 14))
        else:
            size = (max(14, round(width * 518 / height / 14) * 14), 518)
        image = image.resize(size, Image.Resampling.BICUBIC)
        arr = (np.asarray(mask) > 127).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
        arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
        mask = Image.fromarray(arr * 255).resize(size, Image.Resampling.NEAREST)
        rgb = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)
        foreground = torch.from_numpy(np.array(mask, dtype=np.float32) / 255)[None]
        dw, dh = 518 - size[0], 518 - size[1]
        pad = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
        rgb_tensors.append(F.pad(rgb, pad, value=1))
        mask_tensors.append(F.pad(foreground, pad, value=0))
    return torch.stack(rgb_tensors), torch.stack(mask_tensors)


@torch.no_grad()
def render_targets(points, colors, source, scale, output, options):
    """Shared by fresh VGGT preparation and new angles from cached geometry."""
    targets, intrinsics = orbit_cameras(
        source, scale, azimuth=options["azimuth"], elevation=options["elevation"],
        distance=options["camera_distance"], fov=options["fov"],
        principal_x=options["principal_x"], principal_y=options["principal_y"], image_size=options["render_size"],
        reference_view=options["reference_view"],
    )
    (output / "renders").mkdir(parents=True, exist_ok=False)
    entries, coverage = [], []
    for i, (target, matrix) in enumerate(zip(targets, intrinsics)):
        image, mask = render_points(points, colors, target, matrix,
                                    options["render_size"], options["point_radius"])
        path = f"renders/{i:03d}.png"
        Image.fromarray((image.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(output / path)
        Image.fromarray(mask.cpu().numpy().astype(np.uint8) * 255).save(output / f"renders/{i:03d}_mask.png")
        entries.append({"image": path, "w2c": target.cpu().tolist(), "intrinsic": matrix.cpu().tolist(),
                        "azimuth": options["azimuth"][i], "elevation": options["elevation"][i]})
        coverage.append(round(mask.float().mean().item(), 4))
    return entries, coverage


@torch.no_grad()
def render_cached_case(case_path, source, output, options):
    with np.load(case_path / "geometry.npz", allow_pickle=False) as data:
        points, colors, scale = (np.array(data[key], copy=True) for key in
                                ("points", "colors", "normalization_scale"))
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
            or colors.shape != points.shape or scale.shape != ()
            or not all(np.isfinite(v).all() for v in (points, colors, scale)) or scale <= 0):
        raise ValueError("Invalid cached geometry: expected finite points/colors [N,3] and positive scalar scale")
    tensors = [torch.as_tensor(value, dtype=torch.float32, device="cuda")
               for value in (points, colors, scale)]
    entries, coverage = render_targets(tensors[0], tensors[1], source.cuda().float(), tensors[2], output, options)
    for entry in entries:
        entry["image"] = output / entry["image"]
        entry["w2c"] = np.asarray(entry["w2c"], dtype=np.float32)
    return entries, coverage


@torch.inference_mode()
def prepare_case(case_path, output, model, model_path, options):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(f"Prepared case already exists; choose another output directory: {output}")
    metadata, images, masks = load_raw_case(case_path, expected_views=options["input_views"])
    rgb, foreground = preprocess_images(images, masks)
    rgb, foreground = rgb.cuda(), foreground.cuda()
    print(f"VGGT: {case_path.name}, {len(images)} input views", flush=True)
    with torch.autocast("cuda", enabled=False):
        tokens, patch_start = model.aggregator(rgb[None])
        pose = model.camera_head(tokens)[-1]
        depth, confidence = model.depth_head(tokens, images=rgb[None], patch_start_idx=patch_start)
        source, intrinsic = pose_encoding_to_extri_intri(pose, rgb.shape[-2:])
    source, intrinsic = source[0].float(), intrinsic[0].float()
    depth, confidence = depth[0, ..., 0].float(), confidence[0].float()
    features = tokens[-1][0, :, patch_start:].reshape(len(images), 37, 37, 2048).permute(0, 3, 1, 2)
    features = features * F.interpolate(foreground, size=(37, 37), mode="nearest")
    features = features.cpu().to(torch.float16).numpy()
    del tokens, pose
    points = unproject_depth(depth, source, intrinsic)
    valid = ((foreground[:, 0] > 0.5) & torch.isfinite(points).all(-1)
             & torch.isfinite(confidence) & (confidence >= 1.0) & (depth > 0))
    if valid.sum() < 100:
        raise ValueError("Too few confident foreground points; check input images and masks")
    threshold = torch.quantile(confidence[valid], options["conf_percentile"] / 100)
    valid &= confidence >= threshold
    points, colors = points[valid], rgb.permute(0, 2, 3, 1)[valid]
    if len(points) > options["max_points"]:
        # Evenly retain samples across all input views after confidence filtering.
        keep = torch.linspace(0, len(points) - 1, options["max_points"], device=points.device).long()
        points, colors = points[keep], colors[keep]
    points, normalized_source, center, scale = normalize_scene(points, source)
    entries, coverage = render_targets(points, colors, normalized_source, scale, output, options)
    (output / "references").mkdir()
    for i, image in enumerate(images):
        image.save(output / "references" / f"{i:03d}.png")
        masks[i].save(output / "references" / f"{i:03d}_mask.png")
    np.savez_compressed(output / "conditioning.npz", recon_feats=features,
                        source_w2c=normalized_source.cpu().numpy())
    np.savez_compressed(output / "geometry.npz", points=points.cpu().numpy(), colors=colors.cpu().numpy(),
                        depth=depth.cpu().numpy(), depth_confidence=confidence.cpu().numpy(),
                        source_w2c_raw=source.cpu().numpy(), source_intrinsic=intrinsic.cpu().numpy(),
                        normalization_center=center.cpu().numpy(), normalization_scale=scale.cpu().numpy())
    hashes = {}
    for ref in metadata["references"]:
        for kind in ("image", "mask"):
            if ref.get(kind):
                path = case_path / ref[kind]
                hashes[ref[kind]] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "id": metadata.get("id", case_path.name), "prompt": metadata["prompt"], "seed": metadata.get("seed", 123),
        "references": [f"references/{i:03d}.png" for i in range(len(images))], "targets": entries,
        "preparation": {"method": "VGGT from RGB images", "vggt_model": str(model_path.resolve()),
                        "feature_layer": -1, "precision": "float32", "options": options,
                        "source_case": str(case_path.resolve()), "source_sha256": hashes,
                        "points": len(points), "render_coverage": coverage,
                        "elapsed_seconds": round(time.monotonic() - start, 2)},
    }
    (output / "case.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    load_case(output)
    print(f"Prepared {output}: {len(points)} points, {len(entries)} fresh target renders", flush=True)
    return output


def prepare_one_case(case_path, output_root, model_path, options):
    from vggt.models.vggt import VGGT
    from safetensors.torch import load_file
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT preprocessing requires CUDA")
    weight = model_path / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    case_path = case_path.resolve()
    # Validate inputs before loading VGGT; discard the data here because
    # prepare_case() reloads it for processing after the model is ready.
    load_raw_case(case_path, expected_views=options["input_views"])
    output = output_root / case_path.name
    if output.exists():
        raise FileExistsError(output)
    model = VGGT()
    model.load_state_dict(load_file(str(weight)), strict=True)
    model = model.eval().requires_grad_(False).cuda().float()
    try:
        return prepare_case(case_path, output, model, model_path, options)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
