"""Run `python tests/pipeline_test.py`; uses CPU and random weights, no downloads."""
from pathlib import Path
import sys

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viewweaver.models.transformer_flux import FluxTransformer2DModel
from viewweaver.models.view_moe import ViewMoE
from viewweaver.pipeline import ViewWeaverPipeline


def check_pipeline():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    transformer = FluxTransformer2DModel(
        in_channels=16, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
        pooled_projection_dim=16, axes_dims_rope=(4, 6, 6), guidance_embeds=True,
    ).eval()
    for name in ("view_moe_double", "view_moe_single"):
        moe = ViewMoE(inner_dim=32, cond_dim=8, num_attention_heads=2,
                      attention_head_dim=16, geo_freq_dim=4, top_k=2)
        # Nonzero output weights ensure reference attention affects the test.
        torch.nn.init.normal_(moe.attn.to_out.weight, std=0.02)
        setattr(transformer, name, moe)
    vae = AutoencoderKL(
        block_out_channels=(8, 8, 8, 8), latent_channels=4,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        norm_num_groups=4, shift_factor=0.0, scaling_factor=1.0,
    )
    pipeline = ViewWeaverPipeline(
        transformer=transformer, vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
    )
    pipeline.set_progress_bar_config(disable=True)
    grid_ids = pipeline._prepare_latent_image_ids(1, 2, 3, "cpu", torch.float32)
    expected_grid = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                  [0, 1, 0], [0, 1, 1], [0, 1, 2]])
    assert torch.equal(grid_ids, expected_grid)
    identity = torch.eye(4)[:3]
    snapshots = {"parameter_keys": tuple(transformer.state_dict())}

    # Rectangular reference grids and unequal generated/rendered token counts.
    for num_views in (1, 8):
        features = torch.randn(num_views, 8, 2, 3)
        cameras = identity[None].repeat(num_views, 1, 1)
        cameras[:, 0, 3] = torch.arange(num_views) * 0.1
        kwargs = dict(
            image=torch.randn(1, 4, 8, 12), height=64, width=64,
            prompt_embeds=torch.randn(1, 4, 24), pooled_prompt_embeds=torch.randn(1, 16),
            negative_prompt_embeds=torch.randn(1, 4, 24),
            negative_pooled_prompt_embeds=torch.randn(1, 16),
            recon_feats=features, source_w2c=cameras, target_w2c=identity[None],
            num_inference_steps=2, _auto_resize=False, output_type="latent",
        )
        captured_ids = []
        handle = transformer.pos_embed.register_forward_pre_hook(
            lambda module, args: captured_ids.append(args[0].detach().clone())
        )
        for cfg in (1.0, 2.5):
            with torch.no_grad():
                result = pipeline(**kwargs, true_cfg_scale=cfg,
                                  generator=torch.Generator().manual_seed(456)).images
            assert result.shape == (1, 16, 16) and torch.isfinite(result).all()
            snapshots[f"pipeline_{num_views}_{cfg}"] = result.clone()
        handle.remove()
        for joint_ids, reference_ids in zip(captured_ids[::2], captured_ids[1::2]):
            assert joint_ids.shape == (4 + 16 + 24, 3)
            assert torch.count_nonzero(joint_ids[:, 0]) == 0
            reference_ids = reference_ids.reshape(num_views, 6, 3)
            for index in range(num_views):
                assert torch.equal(reference_ids[index, :, 1:], expected_grid[:, 1:])
                assert (reference_ids[index, :, 0] == index + 1).all()

        # Both accepted input layouts must generate identical results.
        kwargs.update(recon_feats=features[None], source_w2c=cameras[None], target_w2c=identity[None, None])
        with torch.no_grad():
            batched = pipeline(**kwargs, true_cfg_scale=1.0,
                               generator=torch.Generator().manual_seed(456)).images
        assert torch.equal(batched, snapshots[f"pipeline_{num_views}_1.0"])

        inputs = dict(
            hidden_states=torch.randn(1, 6, 16, requires_grad=True),
            render_latents=torch.randn(1, 4, 16),
            encoder_hidden_states=kwargs["prompt_embeds"],
            pooled_projections=kwargs["pooled_prompt_embeds"],
            timestep=torch.tensor([0.5]), guidance=torch.tensor([3.5]),
            img_ids=grid_ids, txt_ids=torch.zeros(4, 3),
            render_ids=grid_ids[:4], recon_ids=grid_ids,
            recon_feats=features.flatten(2).transpose(1, 2)[None].contiguous().requires_grad_(),
            target_w2c=identity[None, None], source_w2c=cameras[None], return_dict=False,
        )
        for checkpointing in (False, True):
            if checkpointing:
                transformer.enable_gradient_checkpointing()
            else:
                transformer.disable_gradient_checkpointing()
            output = transformer(**inputs)[0]
            gradients = torch.autograd.grad(output.square().mean(),
                                            (inputs["hidden_states"], inputs["recon_feats"]))
            assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in gradients)
            key = f"transformer_{num_views}_{checkpointing}"
            snapshots[key] = output.detach().clone()
            snapshots[key + "_gradients"] = tuple(grad.clone() for grad in gradients)
        assert torch.equal(grid_ids, expected_grid), "Reference IDs must not be mutated"
        torch.testing.assert_close(snapshots[f"transformer_{num_views}_False"],
                                   snapshots[f"transformer_{num_views}_True"], rtol=0, atol=0)
        for normal, checkpointed in zip(snapshots[f"transformer_{num_views}_False_gradients"],
                                       snapshots[f"transformer_{num_views}_True_gradients"]):
            torch.testing.assert_close(normal, checkpointed, rtol=0, atol=0)
        transformer.disable_gradient_checkpointing()
    print("PASS: token IDs, 1/8 references, CFG, batched inputs, checkpointed outputs and gradients")
    return snapshots


if __name__ == "__main__":
    check_pipeline()
