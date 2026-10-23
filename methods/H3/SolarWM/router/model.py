"""Compact the target video rows before any Transformer Q/FFN computation."""
import torch
from torch import nn
from .attention import SparseControl, install_processors
from .cache import PatchWorkspace


class H3Backbone:
    def __init__(self, model):
        self.model = model
        self._hidden = None
        self.capture = False
        install_processors(model)
        self.hook = model.norm_out.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        if self.capture:
            self._hidden = args[0].detach()

    def wrap(self, model):
        self.model = model

    @torch.no_grad()
    def forward(self, *, inputs, rows, timestep, chunk_index, cache,
                active_ids, refresh_ids, workspace, commit=False, features=False):
        from ..backbone.torch_layout import build_row_timesteps
        from ..backbone.distributed import sequence_all_gather as sequence_model_parallel_all_gather
        layout = inputs.layout("rollout")
        device = rows.device
        compute_ids = torch.unique(torch.cat((active_ids, refresh_ids)), sorted=True)
        prefix = int(layout.condition_indices.numel() + layout.audio_indices.numel())
        keep = torch.cat((torch.arange(prefix, device=device), prefix + compute_ids))
        times, indices = build_row_timesteps(
            layout, timestep, inputs.audio_timestep, text_timestep=1.0,
            condition_video_timestep=inputs.keyframe_timestep, device=device)
        row_times = times[indices]
        row_times[prefix + refresh_ids] = 1.0
        times, indices = torch.unique(row_times[keep], sorted=True, return_inverse=True)
        video_ids = torch.cat((layout.condition_video_indices,
                               prefix + torch.arange(compute_ids.numel(), device=device)))
        control = SparseControl(layout, inputs.camera_viewmats, inputs.camera_K,
                                chunk_index=chunk_index, cache=cache, commit_cache=commit,
                                compute_ids=compute_ids, refresh_ids=refresh_ids, workspace=workspace)
        self.capture, self._hidden = features, None
        try:
            output, _ = self.model(
                hidden_states=torch.cat((inputs.anchor_rows, rows[:, compute_ids]), 1),
                audio_hidden_states=inputs.audio_rows, encoder_hidden_states=inputs.prompt,
                timestep=times, timestep_indices=indices,
                token_tags=layout.token_tags[keep], position_ids=layout.position_ids[keep],
                video_indices=video_ids, audio_indices=layout.audio_indices,
                text_indices=layout.text_indices, fused_prope=True,
                packed_sequence_parallel=inputs.sp_enabled, sgf_attention=control, return_dict=False)
            velocity = output[:, layout.num_condition_video_rows:]
            active_slots = torch.searchsorted(compute_ids, active_ids)
            x0 = rows[:, active_ids] + (1 - timestep) * velocity[:, active_slots].float()
            hidden = None
            if features:
                hidden = self._hidden
                if inputs.sp_enabled:
                    hidden = sequence_model_parallel_all_gather(hidden, dim=1)
                hidden = hidden[:, prefix:][:, active_slots].clone()
            return x0, hidden
        finally:
            self.capture, self._hidden = False, None

    @torch.no_grad()
    def commit(self, inputs, rows, chunk_index, cache):
        ids = torch.arange(rows.shape[1], device=rows.device)
        self.forward(inputs=inputs, rows=rows, timestep=rows.new_tensor(1.0),
                     chunk_index=chunk_index, cache=cache, active_ids=ids,
                     refresh_ids=ids[:0], workspace=PatchWorkspace(), commit=True)


class RouterHead(nn.Module):
    def __init__(self, dim=5376, hidden_dim=2048, initial_exit_bias=-2.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim, eps=1e-6), nn.Linear(dim, hidden_dim),
                                 nn.SiLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, initial_exit_bias)
        self.float()

    def forward(self, hidden):
        logits = self.net(hidden.detach().float()).squeeze(-1)
        return logits
