"""Read portable raw or prepared cases without a dataset dependency."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def case_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input file must be inside the case directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_raw_case(case_dir, expected_views=None):
    """Only images and masks are read; no geometry or feature cache is accepted."""
    root = Path(case_dir).resolve()
    metadata = json.loads((root / "case.json").read_text())
    if not isinstance(metadata.get("prompt"), str) or not metadata["prompt"].strip():
        raise ValueError("case.json must contain a nonempty prompt")
    references = metadata.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("Raw case requires a nonempty references list")
    if expected_views is not None and len(references) != expected_views:
        raise ValueError(f"Expected {expected_views} input views, found {len(references)} in {root}")
    images, masks = [], []
    for reference in references:
        if not isinstance(reference, dict) or "image" not in reference:
            raise ValueError("Raw references must be objects with image and mask paths")
        path = case_file(root, reference["image"])
        image = read_rgb(path)
        if reference.get("mask"):
            with Image.open(case_file(root, reference["mask"])) as mask_image:
                mask = mask_image.convert("L")
        else:
            with Image.open(path) as original:
                if original.mode != "RGBA":
                    raise ValueError(f"Provide a foreground mask or RGBA alpha for {path}")
                mask = original.getchannel("A").copy()
        if mask.size != image.size:
            raise ValueError(f"Image and mask sizes differ: {path}")
        if not np.asarray(mask).any():
            raise ValueError(f"Empty foreground mask: {path}")
        images.append(image)
        masks.append(mask)
    return metadata, images, masks


def read_rgb(path):
    with Image.open(path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, "white")
            return Image.alpha_composite(background, image).convert("RGB")
        return image.convert("RGB")


def validate_pose(value, shape, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite {shape}, got {value.shape}")
    if np.any(np.abs(np.linalg.det(value[..., :3])) < 1e-6):
        raise ValueError(f"{name}: singular camera rotation")
    return value


def load_case(case_dir):
    root = Path(case_dir).resolve()
    metadata = json.loads((root / "case.json").read_text())
    if not isinstance(metadata.get("prompt"), str) or not metadata["prompt"].strip():
        raise ValueError("case.json must contain a nonempty prompt")
    with np.load(root / "conditioning.npz", allow_pickle=False) as data:
        features = data["recon_feats"].copy()
        source = data["source_w2c"].copy()
    if features.ndim != 4 or features.shape[1] != 2048 or min(features.shape) == 0:
        raise ValueError(f"recon_feats must have shape [S, 2048, H, W], got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("recon_feats contains NaN or infinity")
    source = validate_pose(source, (features.shape[0], 3, 4), "source_w2c")
    targets = metadata.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("case.json must contain at least one target")
    for target in targets:
        target["w2c"] = validate_pose(target["w2c"], (3, 4), "target w2c")
        image_path = case_file(root, target["image"])
        with Image.open(image_path) as image:
            image.verify()
        target["image"] = image_path
    return metadata, torch.from_numpy(features), torch.from_numpy(source)
