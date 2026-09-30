"""Run from the project root: python tests/smoke_test.py (no model weights needed)."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viewweaver.inputs import load_raw_case
from viewweaver.geometry import angle_pairs, unproject_depth, normalize_scene, orbit_cameras, render_points
from viewweaver.preprocessing import add_preparation_arguments, preparation_options, preprocess_images
from viewweaver.models.view_moe import ViewMoE
from infer import parse_args, save_comparison


def main():
    torch.set_num_threads(2)
    defaults = parse_args(['--case', 'examples/mario'])
    options = preparation_options(defaults)
    assert defaults.mode == 'end-to-end' and defaults.save_comparison
    assert options['input_views'] == 8
    assert options['reference_view'] == 0 and options['azimuth'] == [-15.0] and options['elevation'] == [15.0]
    assert options['camera_distance'] == 4.5 and options['principal_y'] == 0.18
    custom = parse_args(['--case', 'examples/mario', '--azimuth', '20',
                         '--camera-distance', '6', '--offset-y', '0', '--no-save-comparison'])
    assert not custom.save_comparison and custom.azimuth == [20.0]
    assert custom.camera_distance == 6 and custom.principal_y == 0
    # Comparison preserves every reference, aspect ratios, and full target pixels.
    with tempfile.TemporaryDirectory() as temp:
        target = Image.new("RGB", (160, 96), "blue")
        for count in (1, 8):
            references = [Image.new("RGB", (80, 40), (30 * i, 200, 0)) for i in range(count)]
            path = Path(temp) / "comparison.png"
            save_comparison(references, target, path)
            with Image.open(path) as comparison:
                width = max(target.width, 128 * count)
                cell = width // count
                target_y = cell + 48
                assert comparison.size == (width, target_y + target.height)
                x = (width - target.width) // 2
                assert comparison.crop((x, target_y, x + target.width, comparison.height)).tobytes() == target.tobytes()
                for i in range(count):
                    assert comparison.getpixel((i * cell + cell // 2, 24 + cell // 2)) == (30 * i, 200, 0)
                    assert comparison.getpixel((i * cell + cell // 2, 24)) == (255, 255, 255)
        try:
            save_comparison([], target, path)
        except ValueError:
            pass
        else:
            raise AssertionError("Empty comparison references were accepted")
    root = Path(__file__).resolve().parents[1]
    assert all((root / "examples" / name / "case.json").is_file()
               for name in ("mario", "backpack"))
    for case in (root / "examples").iterdir():
        if not case.is_dir():
            continue
        metadata, images, masks = load_raw_case(case)
        assert len(images) == len(masks) == len(metadata["references"]) == 8
        assert len({image.tobytes() for image in images}) == 8, "References must not be duplicates"
        assert not list(case.rglob('*.npz')) and 'targets' not in metadata
        rgb, foreground = preprocess_images(images, masks)
        assert rgb.shape == (8, 3, 518, 518) and foreground.shape == (8, 1, 518, 518)
    # A missing view must fail on the requested case, not silently select another.
    with tempfile.TemporaryDirectory() as temp:
        case = Path(temp) / "case"
        shutil.copytree(root / "examples/backpack", case)
        metadata = json.loads((case / "case.json").read_text())
        (case / metadata["references"][0]["image"]).unlink()
        try:
            load_raw_case(case)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing reference image was silently accepted")
    # Normalization must leave camera-space positions (and therefore projections) unchanged.
    identity = torch.eye(4)[:3]
    points = torch.tensor([[-0.5, -0.2, 2.0], [0.7, 0.8, 3.0], [0.1, 0.3, 2.5]])
    normalized, cameras, center, scale = normalize_scene(points, identity[None])
    assert torch.allclose(normalized @ cameras[0, :, :3].T + cameras[0, :, 3], points, atol=1e-6)
    intrinsic = torch.tensor([[10., 0., 1.], [0., 10., 1.], [0., 0., 1.]])
    depth = torch.full((1, 3, 3), 2.)
    world = unproject_depth(depth, identity[None], intrinsic[None])
    assert torch.allclose(world[0, 1, 1], torch.tensor([0., 0., 2.]))
    targets, matrices = orbit_cameras(cameras, scale, azimuth=[0, 90, 180, 270], image_size=32)
    assert torch.equal(matrices[:, :2, 2], torch.full((4, 2), 16.))
    assert not torch.allclose(targets[0], targets[1])
    for target, matrix in zip(targets, matrices):
        rotation = target[:, :3] / scale
        assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-5)
        projection = matrix @ target[:, 3]
        assert torch.allclose(projection[:2] / projection[2], matrix[:2, 2], atol=1e-4)
    assert angle_pairs([0, 90, -45], [15]) == ([0, 90, -45], [15, 15, 15])
    assert angle_pairs([30], [-15, 15]) == ([30, 30], [-15, 15])
    for az, el in (([0, 30], [0, 15, 30]), ([float("nan")], [0]), ([0], [90])):
        try:
            angle_pairs(az, el)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid target angle list was accepted")
    known_source = identity.clone()
    known_source[2, 3] = 3
    # Framing offsets translate projections without moving cameras; distance controls size.
    base, k = orbit_cameras(known_source[None], torch.tensor(1.), elevation=[0], image_size=100)
    shifted, shifted_k = orbit_cameras(known_source[None], torch.tensor(1.), elevation=[0],
                                      image_size=100, principal_x=0.1, principal_y=0.15)
    assert torch.equal(base, shifted)
    sample = torch.tensor([[0., 0., 0.], [0.2, 0.3, 0.]])
    camera_points = sample @ base[0, :, :3].T + base[0, :, 3]
    uv, shifted_uv = camera_points @ k[0].T, camera_points @ shifted_k[0].T
    delta = shifted_uv[:, :2] / shifted_uv[:, 2:] - uv[:, :2] / uv[:, 2:]
    assert torch.allclose(delta, torch.tensor([[10., 15.], [10., 15.]]), atol=1e-5)
    far, far_k = orbit_cameras(known_source[None], torch.tensor(1.), elevation=[0], distance=7., image_size=100)
    far_points = sample @ far[0, :, :3].T + far[0, :, 3]
    far_uv = far_points @ far_k[0].T
    projected, far_projected = uv[:, :2] / uv[:, 2:], far_uv[:, :2] / far_uv[:, 2:]
    assert torch.allclose(far_projected[1] - far_projected[0], (projected[1] - projected[0]) / 2, atol=1e-5)
    parser = argparse.ArgumentParser()
    add_preparation_arguments(parser)
    options = preparation_options(parser.parse_args(['--offset-x', '0.1', '--offset-y', '0.15']))
    legacy = preparation_options(parser.parse_args(['--principal-x', '0.1', '--principal-y', '0.15']))
    assert options == legacy and options['principal_x'] == 0.1 and options['principal_y'] == 0.15
    for option, value in (('--offset-x', '0.5'), ('--offset-y', '-0.5'), ('--offset-x', 'nan')):
        try:
            preparation_options(parser.parse_args([option, value]))
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid framing offset was accepted')
    explicit, _ = orbit_cameras(known_source[None], torch.tensor(1.),
                                azimuth=[0, 90, -90], elevation=[0, 30, -30])
    positions = -torch.linalg.solve(explicit[:, :, :3], explicit[:, :, 3])
    assert torch.allclose(positions.norm(dim=-1), torch.full((3,), 3.5), atol=1e-5)
    assert torch.allclose(positions[0], torch.tensor([0., 0., -3.5]), atol=1e-5)
    assert positions[1, 0] > 0 and positions[1, 1] < 0  # right and above
    assert positions[2, 0] < 0 and positions[2, 1] > 0  # left and below
    # Selecting a camera must change both the azimuth origin and up, without reordering inputs.
    original = explicit.clone()
    selected, _ = orbit_cameras(explicit, torch.tensor(1.), reference_view=1)
    expected, _ = orbit_cameras(explicit[1:2], torch.tensor(1.))
    default, _ = orbit_cameras(explicit, torch.tensor(1.))
    zero, _ = orbit_cameras(explicit, torch.tensor(1.), reference_view=0)
    assert torch.equal(selected, expected) and not torch.allclose(selected, default)
    assert torch.equal(default, zero) and torch.equal(explicit, original)
    for index in (-1, len(explicit)):
        try:
            orbit_cameras(explicit, torch.tensor(1.), reference_view=index)
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-range reference view was accepted")
    try:
        load_raw_case(root / "examples/mario", expected_views=4)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong reference count was accepted")
    # A nearer red point must occlude the blue point at the same projected pixel.
    pixels, mask = render_points(torch.tensor([[0., 0., 1.], [0., 0., 2.]]),
        torch.tensor([[1., 0., 0.], [0., 0., 1.]]), identity, intrinsic, image_size=4, radius=0)
    assert mask.sum() == 1 and torch.equal(pixels[1, 1], torch.tensor([1., 0., 0.]))
    assert torch.equal(pixels[0, 0], torch.ones(3))
    # Preserve routing behavior on one and several reference views, including k > S.
    torch.manual_seed(123)
    model = ViewMoE(inner_dim=16, cond_dim=8, num_attention_heads=2, attention_head_dim=8,
                    geo_freq_dim=4, top_k=4).eval()
    x = torch.randn(1, 5, 16)
    identity = torch.eye(4)[:3]
    for count in (1, 8):
        feats = torch.randn(1, count, 4, 8)
        pose = identity[None, None].repeat(1, count, 1, 1)
        out, logits = model(x, feats, identity[None, None], pose)
        assert out.shape == x.shape and logits.shape == (1, count)
        assert torch.equal(out, x), "Zero-initialized attention must preserve the residual"
        assert torch.isfinite(logits).all()
    # Exercise the complete denoising path with tiny random components, no downloads.
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from viewweaver.models.transformer_flux import FluxTransformer2DModel
    from viewweaver.pipeline import ViewWeaverPipeline
    transformer = FluxTransformer2DModel(
        in_channels=16, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
        pooled_projection_dim=16, axes_dims_rope=(4, 6, 6), guidance_embeds=True,
    ).eval()
    for name in ("view_moe_double", "view_moe_single"):
        setattr(transformer, name, ViewMoE(inner_dim=32, cond_dim=8,
                num_attention_heads=2, attention_head_dim=16, geo_freq_dim=4, top_k=2))
    vae = AutoencoderKL(block_out_channels=(8, 8, 8, 8), latent_channels=4,
                       down_block_types=("DownEncoderBlock2D",) * 4,
                       up_block_types=("UpDecoderBlock2D",) * 4,
                       norm_num_groups=4, shift_factor=0.0, scaling_factor=1.0)
    pipeline = ViewWeaverPipeline(
        transformer=transformer, vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
    )
    with torch.inference_mode():
        result = pipeline(image=Image.new("RGB", (64, 64), "gray"),
            prompt_embeds=torch.randn(1, 4, 24), pooled_prompt_embeds=torch.randn(1, 16),
            recon_feats=torch.randn(8, 8, 2, 2), source_w2c=identity[None].repeat(8, 1, 1),
            target_w2c=identity[None], height=64, width=64, num_inference_steps=2,
            _auto_resize=False, joint_attention_kwargs={"attention_mask": None}).images[0]
    assert result.size == (64, 64)
    print("PASS: reference–target comparisons, raw eight-view cases, missing-file handling, geometry normalization, orbit cameras, z-buffer")
    print("PASS: tiny FLUX + ViewMoE + VAE denoising pipeline (random weights; not a quality test)")


if __name__ == "__main__":
    main()
