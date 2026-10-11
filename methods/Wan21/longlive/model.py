"""Short-history LongLive router with the paper's two-term loss."""
from types import SimpleNamespace

import torch

from methods.Wan21.shared.model.router import RouterModel

from .cache import install_recache_attention
from .config import resolve_path
from .weights import load_merged


def router_args(cfg):
    return SimpleNamespace(
        num_frame_per_block=3, num_training_frames=21, context_noise=0,
        mixed_precision=True, router_nfe_weight=float(cfg.nfe_weight),
        router_exit_threshold=float(cfg.exit_threshold), router_initial_exit_bias=float(cfg.initial_exit_bias),
        router_head_dim=int(cfg.head_dim), first_chunk_step=4,
        router_train_chunk_min=1, router_train_chunk_max=6, router_train_step_choices=[1, 2, 3],
        denoising_step_list=list(cfg.denoising_steps), warp_denoising_step=True,
        generator_ckpt=str(resolve_path(cfg.resume or cfg.trained_path or cfg.merged_checkpoint)),
        gradient_checkpointing=False,
        model_kwargs=dict(model_name="Wan2.1-T2V-1.3B", model_path=cfg.assets_path,
                          timestep_shift=5.0, local_attn_size=12, sink_size=3,
                          causal_model_variant="adaptive", load_dtype="bf16", num_train_timesteps=1000),
    )


class LongLiveRouterModel(RouterModel):
    def __init__(self, cfg, device):
        self.public_config = cfg
        super().__init__(router_args(cfg), device=device)
        install_recache_attention(self.generator)

    def _load_generator_checkpoint(self, path):
        self.longlive_metadata = load_merged(self.generator, path, allow_router=True)

    def _sample_chunk_and_step(self, num_chunks):
        choices = [int(x) for x in self.public_config.train_chunk_choices]
        if not choices or any(x < 1 or x >= num_chunks for x in choices):
            raise ValueError("Training targets must be noninitial chunks within the 21-frame clip")
        chunk = choices[int(torch.randint(len(choices), (), device=self.device))]
        step = int(torch.randint(1, 4, (), device=self.device))
        return chunk, step
