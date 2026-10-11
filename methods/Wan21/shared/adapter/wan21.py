"""Wan 2.1 implementation of the model-independent lifecycle protocol."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from methods.core.data import open_legacy_lmdb, validate_legacy_lmdb
from methods.core.errors import AdapterError
from methods.core.weights import checkpoint, wan_model


_MODEL_NAMES = {
    "wan21-1b3": "Wan2.1-T2V-1.3B",
    "wan21-14b": "Wan2.1-T2V-14B",
}

_PARAM_ALIASES = {
    "num_train_timesteps": "num_train_timestep",
    "denoising_steps": "denoising_step_list",
    "warp_denoising_steps": "warp_denoising_step",
    "discrete_cd_steps": "discrete_cd_N",
    "learning_rate": "lr",
    "generator_learning_rate": "lr",
    "critic_learning_rate": "lr_critic",
    "fake_score_updates_per_generator": "dfake_gen_update_ratio",
    "ema_decay": "ema_weight",
    "exit_steps": "adaptive_exit_steps",
    "exit_sampling": "adaptive_exit_sampling_mode",
    "exit_step_weights": "adaptive_exit_step_weights",
    "fallback_probabilities": "adaptive_fallback_prob",
}

_ROUTER_PARAM_ALIASES = {
    "nfe_weight": "router_nfe_weight",
    "exit_threshold": "router_exit_threshold",
    "initial_exit_bias": "router_initial_exit_bias",
    "train_chunk_min": "router_train_chunk_min",
    "train_chunk_max": "router_train_chunk_max",
    "train_step_choices": "router_train_step_choices",
    "head_dim": "router_head_dim",
    "reset_router": "router_reset_head",
}

_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_path(value: Any) -> Path | None:
    if value in (None, "", False):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (_repo_root() / path).resolve()


def _role_model_name(roles: dict[str, Any], role: str, fallback: str) -> str:
    model_id = str(roles.get(role, fallback))
    try:
        return _MODEL_NAMES[model_id]
    except KeyError as exc:
        raise AdapterError(f"Wan21 role {role!r} references unknown model id {model_id!r}") from exc


def to_legacy_config(config: DictConfig, *, require_inputs: bool = False, for_training: bool = True) -> DictConfig:
    """Translate the public structured recipe to the retained Wan trainer ABI."""

    stage = str(config.stage)
    params = dict(config.params)
    if stage == "router" and for_training:
        from methods.Wan21.shared.router_objective import validate_router_training_config

        validate_router_training_config(params)
    runtime = dict(config.runtime)
    roles = dict(config.roles)
    data = dict(config.data)
    artifacts = dict(config.artifacts)
    weights = dict(config.weights)

    translated: dict[str, Any] = {}
    for key, value in params.items():
        target = _PARAM_ALIASES.get(key, key)
        if stage == "router":
            target = _ROUTER_PARAM_ALIASES.get(key, target)
        translated[target] = value

    num_train_timesteps = int(
        params.get("num_train_timesteps", params.get("num_train_timestep", 1000))
    )
    if num_train_timesteps <= 0:
        raise AdapterError("params.num_train_timesteps must be positive")

    model_id = str(config.model_id)
    generator_name = _role_model_name(roles, "generator", model_id)
    mixed_precision_value = runtime.get("mixed_precision", "bf16")
    mixed_precision = mixed_precision_value is True or str(mixed_precision_value).lower() in {
        "true",
        "1",
        "bf16",
        "bfloat16",
    }
    fsdp = dict(runtime.get("fsdp", {}))
    backward_prefetch_aliases = {
        "pre": "backward_pre",
        "backward_pre": "backward_pre",
        "post": "backward_post",
        "backward_post": "backward_post",
        "none": "none",
    }
    requested_backward_prefetch = str(
        fsdp.get("backward_prefetch", "backward_pre")
    ).strip().lower()
    try:
        fsdp_backward_prefetch = backward_prefetch_aliases[
            requested_backward_prefetch
        ]
    except KeyError as exc:
        choices = ", ".join(sorted(backward_prefetch_aliases))
        raise AdapterError(
            "runtime.fsdp.backward_prefetch must be one of "
            f"{choices}, got {requested_backward_prefetch!r}"
        ) from exc
    model_load_dtype = runtime.get("model_load_dtype")
    data_path = _resolve_path(data.get("source"))
    resume_checkpoint = _resolve_path(artifacts.get("resume_checkpoint"))
    if require_inputs and resume_checkpoint is not None and not resume_checkpoint.exists():
        raise AdapterError(f"resume checkpoint does not exist: {resume_checkpoint}")
    output_dir = _resolve_path(artifacts.get("output_dir"))
    if output_dir is None:
        raise AdapterError("artifacts.output_dir must be configured")
    if require_inputs and data_path is None:
        raise AdapterError(
            f"{model_id}/{stage} requires data.source; pass --set data.source=<path>"
        )
    if require_inputs and data_path is not None and not data_path.exists():
        raise AdapterError(f"data source does not exist: {data_path}")

    # CD bundles restore generator/EMA only; the frozen teacher still needs
    # the original TF initialization when resuming.
    parent_checkpoint = (
        checkpoint(weights["init"], filename=("chunkwise/causal_forcing.pt", "Causal-Forcing/ForcingBooster.pt"))
        if for_training and (not resume_checkpoint or stage == "cd") and weights.get("init") else None
    )
    model_path = wan_model(weights.get("model"))
    real_model_path = model_path
    if for_training and stage in {"dmd", "adaptive_dmd"}:
        real_model_path = wan_model(weights.get("teacher") or weights["model"], assets=False)

    defaults: dict[str, Any] = {
        "model_id": model_id,
        "stage": stage,
        "config_name": f"{model_id}-{stage}",
        "logdir": str(output_dir),
        "data_path": None if data_path is None else str(data_path),
        "generator_ckpt": None if parent_checkpoint is None else str(parent_checkpoint),
        "resume_checkpoint": (
            None if resume_checkpoint is None else str(resume_checkpoint)
        ),
        "model_kwargs": {
            "model_name": generator_name,
            "model_path": str(model_path),
            "timestep_shift": float(params.get("timestep_shift", 5.0)),
            "causal_model_variant": "adaptive" if stage in {"adaptive_dmd", "router"} else "dense",
            "load_dtype": model_load_dtype,
            "num_train_timesteps": num_train_timesteps,
        },
        "num_train_timesteps": num_train_timesteps,
        # Retain the old trainer ABI while primary consumers migrate to the
        # public plural spelling.
        "num_train_timestep": num_train_timesteps,
        "model_load_dtype": model_load_dtype,
        "real_model_path": str(real_model_path),
        "fake_model_path": str(model_path),
        "real_name": _role_model_name(roles, "real_teacher", model_id),
        "fake_name": _role_model_name(roles, "fake_score", model_id),
        "causal": True,
        "is_causal": True,
        "i2v": False,
        "independent_first_frame": False,
        "same_step_across_blocks": True,
        "last_step_only": False,
        "context_noise": 0,
        "load_raw_video": False,
        "image_or_video_shape": [
            int(params.get("batch_size", 1)),
            int(params.get("num_training_frames", 21)),
            16,
            60,
            104,
        ],
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": bool(runtime.get("gradient_checkpointing", True)),
        "sharding_strategy": str(runtime.get("sharding_strategy", "full")),
        "generator_fsdp_wrap_strategy": str(fsdp.get("wrap_strategy", "size")),
        "real_score_fsdp_wrap_strategy": str(fsdp.get("wrap_strategy", "size")),
        "fake_score_fsdp_wrap_strategy": str(fsdp.get("wrap_strategy", "size")),
        "text_encoder_fsdp_wrap_strategy": str(fsdp.get("wrap_strategy", "size")),
        "fsdp_backward_prefetch": fsdp_backward_prefetch,
        "text_encoder_cpu_offload": bool(fsdp.get("cpu_offload", False)),
        "seed": int(runtime.get("seed", 0)),
        "weight_decay": float(params.get("weight_decay", 0.01)),
        "beta1": float(params.get("beta1", 0.0)),
        "beta2": float(params.get("beta2", 0.999)),
        "beta1_critic": float(params.get("beta1_critic", 0.0)),
        "beta2_critic": float(params.get("beta2_critic", 0.999)),
        "negative_prompt": str(params.get("negative_prompt", _NEGATIVE_PROMPT)),
        "guidance_scale": float(params.get("guidance_scale", 3.0)),
        "real_guidance_scale": float(params.get("real_guidance_scale", 3.0)),
        "fake_guidance_scale": float(params.get("fake_guidance_scale", 1.0)),
        "denoising_loss_type": str(params.get("denoising_loss_type", "flow")),
        "distribution_loss": str(params.get("distribution_loss", "dmd")),
        "ema_start_step": int(params.get("ema_start_step", 200)),
        "log_iters": int(params.get("log_iters", 100)),
        "progress_log_iters": int(params.get("progress_log_iters", 10)),
        "gc_interval": int(params.get("gc_interval", 100)),
        "no_save": bool(runtime.get("no_save", False)),
        "no_visualize": bool(runtime.get("no_visualize", True)),
        "disable_wandb": not bool(runtime.get("wandb", {}).get("enabled", False)),
        "wandb_host": str(runtime.get("wandb", {}).get("host", "https://api.wandb.ai")),
        "wandb_key": os.environ.get("WANDB_API_KEY", ""),
        "wandb_entity": os.environ.get("WANDB_ENTITY", ""),
        "wandb_project": os.environ.get("WANDB_PROJECT", "causal-forcing"),
        "wandb_save_dir": str(output_dir),
        "wandb_run_name": f"{model_id}-{stage}",
        "artifact_manifest": {
            "model_id": model_id,
            "stage": stage,
            "parent_checkpoint": None if parent_checkpoint is None else str(parent_checkpoint),
            "save_full_generator": bool(artifacts.get("save_full_generator", True)),
        },
        "resolved_config": OmegaConf.to_container(config, resolve=True),
    }
    defaults.update(translated)

    if stage == "adaptive_dmd":
        defaults["adaptive_seq_mode"] = "adaptive"
    else:
        defaults["adaptive_seq_mode"] = "none"
    if stage == "router":
        defaults["router_seq_mode"] = "router"
        defaults["require_router_checkpoint"] = True
        if "first_chunk_step" not in defaults:
            configured_steps = list(defaults["denoising_step_list"])
            defaults["first_chunk_step"] = len(configured_steps) - int(
                bool(configured_steps) and float(configured_steps[-1]) == 0.0
            )
        defaults.setdefault("router_nfe_weight", 0.03)
        if bool(defaults.get("router_reset_head", False)) and resume_checkpoint is not None:
            raise AdapterError("params.reset_router cannot be combined with resume_checkpoint")
        for name in ("max_optimizer_updates", "max_train_attempts"):
            if int(defaults.get(name, 0)) < 0:
                raise AdapterError(f"params.{name} must be non-negative")
        if int(defaults.get("max_optimizer_updates", 0)):
            defaults["max_train_steps"] = 0
            defaults.setdefault("max_train_attempts", 10000)
        defaults.setdefault("save_initial_checkpoint", True)
    else:
        defaults["router_seq_mode"] = "none"
    return OmegaConf.create(defaults)


class Wan21Adapter:
    """Execute the five public lifecycle stages using the Wan 2.1 family code."""

    def __init__(self, manifest: Any = None) -> None:
        self.manifest = manifest

    def prepare(self, config: DictConfig) -> dict[str, Any]:
        data = dict(config.data)
        source = _resolve_path(data.get("source"))
        if source is None:
            return {
                "status": "requires_data_source",
                "model_id": str(config.model_id),
                "stage": str(config.stage),
                "override": "--set data.source=<path>",
            }
        if not source.exists():
            raise AdapterError(f"data source does not exist: {source}")
        if str(data.get("format", "")).lower() in {"latent_lmdb", "legacy_latent_lmdb"}:
            env = open_legacy_lmdb(source)
            try:
                summary = validate_legacy_lmdb(env)
            finally:
                env.close()
            return {
                "status": "ready",
                "format": summary.format,
                "length": summary.length,
                "latents_shape": summary.latents_shape,
            }
        if not source.is_file():
            raise AdapterError(f"prompt data source must be a file: {source}")
        return {"status": "ready", "format": "prompt_file", "path": str(source)}

    def train(self, config: DictConfig) -> None:
        legacy = to_legacy_config(config, require_inputs=True)
        stage = str(config.stage)
        import torch.distributed as dist

        from methods.Wan21.shared.trainer import (
            ConsistencyDistillationTrainer,
            DiffusionTrainer,
            RouterTrainer,
            ScoreDistillationTrainer,
        )

        trainer_cls = {
            "tf": DiffusionTrainer,
            "cd": ConsistencyDistillationTrainer,
            "dmd": ScoreDistillationTrainer,
            "adaptive_dmd": ScoreDistillationTrainer,
            "router": RouterTrainer,
        }[stage]
        try:
            trainer_cls(legacy).train()
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()

    def _inference_command(self, config: DictConfig) -> tuple[list[str], Path]:
        data = dict(config.data)
        artifacts = dict(config.artifacts)
        runtime = dict(config.runtime)
        prompt_path = _resolve_path(data.get("validation_source") or data.get("source"))
        if prompt_path is None or not prompt_path.is_file():
            raise AdapterError(
                "inference requires a prompt file via data.validation_source or data.source"
            )
        if str(config.stage) == "router" and not config.weights.get("checkpoint"):
            raise AdapterError("Router inference requires weights.checkpoint")
        legacy = to_legacy_config(config, require_inputs=False, for_training=False)
        checkpoint_path = checkpoint(
            config.weights.get("checkpoint") or (Path(str(artifacts["output_dir"])) / "latest"),
            filename=("Causal-Forcing/ForcingBooster.pt", "chunkwise/causal_forcing.pt"),
        )
        output_dir = _resolve_path(
            artifacts.get("validation_output_dir")
            or (Path(str(artifacts["output_dir"])) / "validation")
        )
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="wan21-resolved-", delete=False
        )
        try:
            temporary.write(OmegaConf.to_yaml(legacy, resolve=True))
            temporary.close()
        except Exception:
            temporary.close()
            Path(temporary.name).unlink(missing_ok=True)
            raise
        command = [
            sys.executable,
            "-m",
            "methods.Wan21.shared.adapter.inference",
            "--config_path",
            temporary.name,
            "--checkpoint_path",
            str(checkpoint_path),
            "--data_path",
            str(prompt_path),
            "--output_folder",
            str(output_dir),
            "--num_output_frames",
            str(int(dict(config.params).get("inference_num_frames", 21))),
            "--seed",
            str(int(runtime.get("seed", 0))),
        ]
        if bool(runtime.get("inference_cpu_offload", False)):
            command.append("--low_memory")
        return command, output_dir

    def infer(self, config: DictConfig) -> dict[str, Any]:
        command, output_dir = self._inference_command(config)
        config_path = Path(command[command.index("--config_path") + 1])
        try:
            subprocess.run(command, cwd=_repo_root(), check=True)
        finally:
            config_path.unlink(missing_ok=True)
        return {"status": "complete", "output_dir": str(output_dir)}

__all__ = ["Wan21Adapter", "to_legacy_config"]
