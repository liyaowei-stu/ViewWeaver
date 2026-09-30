# Input and Output Formats

[README](../README.md) · [Inference guide](inference.md)

Paths and commands below are relative to the project root.

## Input cases

Organize each object as a self-contained case:

```text
examples/my_object/
├── case.json
├── images/000.png ... 007.png
└── masks/000.png ... 007.png
```

Example `case.json`:

```json
{
  "id": "my_object",
  "prompt": "Place the object on a wooden desk.",
  "seed": 123,
  "references": [
    {"image": "images/000.png", "mask": "masks/000.png"},
    {"image": "images/001.png", "mask": "masks/001.png"},
    {"image": "images/002.png", "mask": "masks/002.png"},
    {"image": "images/003.png", "mask": "masks/003.png"},
    {"image": "images/004.png", "mask": "masks/004.png"},
    {"image": "images/005.png", "mask": "masks/005.png"},
    {"image": "images/006.png", "mask": "masks/006.png"},
    {"image": "images/007.png", "mask": "masks/007.png"}
  ]
}
```

Masks must match their image dimensions, with white foreground and black background. Alternatively, provide RGBA images and omit the mask field to use the alpha channel. All paths are relative to the case directory and must remain inside it. Reference order determines the indices used by `--reference-view`.

Foreground segmentation is supplied by the input masks. Reconstruction is computed from the reference images; the raw case does not require precomputed features, depth maps, or cameras.

## Generated outputs and caches

Generated images are numbered in the requested angle order. Each run creates a timestamped directory containing the target images and `result.json`. The record includes the prompt, seed, model paths, reference index, target angles, camera matrices, and conditioning image paths.

By default, each target also has a `000_comparison.png` (and so on) containing all reference images above the target. The `comparison_images` list in `result.json` records these files. Use `--no-save-comparison` to disable them.

Prepared cases contain:

```text
outputs/prepared_8views/<case>/
├── conditioning.npz      # Reference features and source cameras
├── geometry.npz          # Point cloud, depth, and reconstruction metadata
├── case.json
├── references/
└── renders/
```

End-to-end inference saves its cache under `outputs/prepared/<timestamp>/<case>/` by default. Override this with `--prepared-dir`, and reuse the resulting case directory with `--mode prepared`.

For reproducible outputs, keep the inputs, seed, reference view, generation settings, and target angle order unchanged. Random noise is sampled sequentially across the requested targets.

## Checkpoint layout

The model loaders expect the following files and directories:

```text
checkpoints/
├── viewweaver/
│   ├── lora_dit.safetensors
│   ├── view_moe_double.safetensors
│   └── view_moe_single.safetensors
├── vggt/
│   └── model.safetensors
└── flux-kontext/
    ├── model_index.json
    ├── transformer/
    ├── vae/
    ├── text_encoder/
    ├── text_encoder_2/
    ├── tokenizer/
    ├── tokenizer_2/
    └── scheduler/
```

Model paths can be overridden with `--checkpoint`, `--vggt-model`, and `--base-model`. Inference loads local files; download instructions are in the [README](../README.md#model-weights).
