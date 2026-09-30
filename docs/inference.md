# Inference Guide

[README](../README.md) · [Data formats](data.md)

Run the commands below from the project root after completing the installation and checkpoint download steps in the README.

## Python entry point

```bash
python infer.py \
  --mode end-to-end \
  --case examples/mario \
  --vggt-model checkpoints/vggt \
  --base-model checkpoints/flux-kontext \
  --checkpoint checkpoints/viewweaver \
  --reference-view 0 \
  --azimuth 0 --elevation 15 \
  --prompt "Place the Mario figure on a wooden desk." \
  --seed 123 \
  --output-dir outputs
```

## Two-stage inference

The default `scripts/run_case.sh --case examples/mario` command handles preparation and generation together. Use the optional two-stage workflow below to reuse geometry across runs.

First, extract the reference features, cameras, and point cloud. Preparation also saves an initial target render for inspection:

```bash
PREPARED_DIR=outputs/prepared_8views \
  bash scripts/prepare_case.sh --case examples/mario
```

Then reuse the cache with new camera angles or prompts:

```bash
bash scripts/run_prepared.sh --case outputs/prepared_8views/mario \
  --reference-view 0 --azimuth 0 --elevation 15

bash scripts/run_prepared.sh --case outputs/prepared_8views/mario \
  --reference-view 0 --azimuth -30 0 30 --elevation 20 \
  --prompt "Place the Mario figure on a wooden desk." --seed 42
```

The second stage renders the requested views from the cached point cloud without rerunning VGGT. Camera settings come from the **current command**, with defaults of reference view `0`, azimuth `−15°`, elevation `15°`, camera distance `4.5`, and vertical offset `0.18`. New conditioning renders are saved under the current result directory.

Preparation refuses to overwrite an existing case cache. Choose a new `PREPARED_DIR` when extracting features again.

## Camera control

Target cameras look toward the reconstructed object center. Select the coordinate frame with `--reference-view`, a **zero-based index** into the input list in `case.json`. For eight inputs, valid indices are `0–7`; the default is `0`.

| Control                | Convention                                                                                                                                                                                                                              |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Azimuth**      | `0°` is the selected reference camera's direction projected onto the reference horizontal plane. Positive angles move the camera to the right around the object: `90°` to the right, `180°` behind, and `-90°` to the left. |
| **Elevation**    | `0°` is level with the object center in the reference frame. Positive values look down from above; negative values look up from below.                                                                                               |
| **Up direction** | Taken from the selected reference camera. A tilted reference image produces a tilted coordinate frame.                                                                                                                                  |
| **Distance**     | Camera distance in normalized object coordinates. Increase it to make the object smaller in the frame.                                                                                                                                  |

Changing the reference view only changes the target camera frame; **all eight input views still participate in inference**. Angles are relative to the selected camera, not an automatically detected semantic front or gravity direction. `(0°, 0°)` does not reproduce the reference image's original distance, elevation, or intrinsics.

To generate several target views, supply angle lists:

```bash
# Three views at a shared elevation.
bash scripts/run_case.sh --case examples/mario --reference-view 0 --azimuth -30 0 30 --elevation 15

# Three paired targets: (0°, 0°), (30°, 15°), and (60°, 30°).
bash scripts/run_case.sh --case examples/mario --reference-view 0 --azimuth 0 30 60 --elevation 0 15 30
```

Lists are paired in order. A single azimuth or elevation is broadcast across the other list; otherwise, lengths must match. The lists do not form a Cartesian product.

### Subject size and placement

Use `--camera-distance` to control subject size: a larger distance makes the subject smaller. Use `--offset-x` and `--offset-y` to shift its placement as fractions of the image width and height. Positive values move the subject right and down; negative values move it left and up. The horizontal offset defaults to `0` and the vertical offset to `0.18`; both must be strictly between `-0.5` and `0.5`.

```bash
# A smaller subject, placed below the image center.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_case.sh --case examples/mario \
  --reference-view 0 --azimuth 0 --elevation 15 \
  --camera-distance 5 --offset-x 0 --offset-y 0.15 --save-comparison
```

This moves the projected reconstruction center to `(50%, 65%)` of the image. The object's visible bounding-box center may differ. Distance changes are approximate size controls because perspective and object depth affect the projection.

The same options work with `scripts/run_prepared.sh --case outputs/prepared_8views/mario`, reusing the cached features and point cloud. Offsets update the target camera intrinsics used for conditioning renders; the generator may slightly change the final placement or silhouette. `--principal-x` and `--principal-y` are equivalent aliases. The effective shifts are recorded as `principal_x` and `principal_y` in `result.json`.

### Reference–target comparisons

Both inference launchers save comparisons by default; use `--no-save-comparison` to disable them. The files are named `000_comparison.png`, `001_comparison.png`, etc. Each image shows all input references in their original order above the generated target. Reference thumbnails preserve their aspect ratios; the target retains its full output resolution. Individual target images are also saved. Comparison filenames are recorded in `result.json`.

### Optional controls

| Argument                                 | Default             | Description                                                                  |
| ---------------------------------------- | ------------------- | ---------------------------------------------------------------------------- |
| `--input-views`                        | `8`               | Expected number of inputs; validates the count without selecting a subset.   |
| `--reference-view`                     | `0`               | Input camera defining azimuth zero and the up direction.                     |
| `--azimuth`                            | `-15`               | One or more horizontal angles in degrees.                                    |
| `--elevation`                          | `15`              | One or more elevation angles in degrees, strictly between`-89` and `89`. |
| `--camera-distance`                    | `4.5`             | Distance from the reconstructed object center.                               |
| `--fov`                                | `55`              | Horizontal field of view in degrees.                                         |
| `--offset-x` / `--offset-y` | `0` / `0.18` | Subject placement shift as a fraction of image width / height; positive moves right / down. |
| `--save-comparison` / `--no-save-comparison` | On | Enable or disable reference–target comparisons. |
| `--steps`                              | `32`              | Number of denoising steps.                                                   |
| `--guidance-scale`                     | `3.5`             | Text guidance scale.                                                         |
| `--width` / `--height`               | `1024` / `1024` | Output dimensions; both must be multiples of 16.                             |
| `--prompt` / `--seed`                | From the case       | Override the text prompt or random seed.                                     |
| `--top-k`                              | `4`               | Number of reference feature views selected by ViewMoE routing.               |
| `--render-size` / `--point-radius`   | `512` / `1.5`   | Conditioning render size and point radius in pixels.                         |
| `--conf-percentile` / `--max-points` | `20` / `200000` | Confidence filtering and point count limit during preparation.               |

Changing point filtering requires a new preparation run. `--top-k` controls feature routing inside ViewMoE; it does not change the number of VGGT input images. See `python infer.py --help` and `python prepare.py --help` for the complete CLI.

### Launcher configuration

The Bash scripts resolve relative paths from the project root. They use `.venv/bin/python` unless `PYTHON` is set; when using Conda, keep `export PYTHON="$(command -v python)"` in the active shell.

```bash
CUDA_VISIBLE_DEVICES=1 OUTPUT_DIR=outputs/custom \
  bash scripts/run_case.sh --case examples/mario --reference-view 0 --azimuth 0 --elevation 15
```

Supported environment variables are `PYTHON`, `CUDA_VISIBLE_DEVICES`, `VGGT_MODEL`, `FLUX_MODEL`, `VIEWWEAVER_CHECKPOINT`, and `OUTPUT_DIR`. Use `PREPARED_DIR` for preparation output. All launchers require an explicit `--case` directory; no example is selected by default.

## Environment and input checks

```bash
python -m pip check
python - <<'CHECK'
import torch
from vggt.models.vggt import VGGT
from viewweaver.loading import load_pipeline

assert torch.cuda.is_available(), 'CUDA is required'
assert torch.cuda.is_bf16_supported(), 'bfloat16 support is required'
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
CHECK
```

Validate raw inputs without loading model weights:

```bash
bash scripts/run_case.sh --case examples/mario --reference-view 0 --check-inputs
```

Run the lightweight geometry and small-model pipeline checks:

```bash
python tests/smoke_test.py
```

The smoke test uses randomly initialized miniature generation components and does not download model weights. Reconstruction quality affects the rendered conditions: poorly observed or occluded regions can produce holes or distortions at novel viewpoints.
