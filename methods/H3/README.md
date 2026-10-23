# MiniMax-H3

Patch router inference and training for [SolarWM](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B) (MiniMax-H3, 33B). Variant details are in [SolarWM](SolarWM/README.md).

## Setup

```bash
pip install -r methods/H3/SolarWM/requirements.txt
hf auth login
```

H3 uses the torch, CUDA and flash-attn of the training image; set `PYTHON` to that interpreter for the scripts. The weights repository is gated: accept its terms before `hf auth login`.

## Inference

The router defaults to `MiniMax-H3/model.pt` from [ForcingBooster](https://huggingface.co/EXCAI/ForcingBooster). `prepare` downloads every weight file (base, Stage2 LoRA, taeh3 decoder, router) and encodes the LingBot examples in `assets/lingbot-world-v2/examples` on one GPU; `infer` then reads them offline on eight GPUs. Pass the same `--set` overrides to both.

```bash
bash methods/H3/SolarWM/scripts/prepare.sh
bash methods/H3/SolarWM/scripts/infer.sh
```

Each sample writes `video.mp4` (158 frames, 768 × 1344, 24 FPS), `latent.pt` and `exit_steps.pt` to `logs/h3/router/<sample>/`. To use your own trained router, append `--set weights.checkpoint=logs/h3/train/latest` to both commands; a `model.pt` or a checkpoint directory also works. Append `--set mode=native` for the original four-step model, `--set forced_exit_step=K` to stop every patch after step K, or `--set decoder=vae` to decode with the official H3 VAE instead of taeh3.

## Training

Router training uses DL3DV-10K windows of 158 frames, read from `dataset/dl3dv10k` (override with `--root` for `build` and `--set dl3dv_root=...` for `prepare`). Build a manifest on CPU, encode its conditions on one GPU, then train on eight GPUs:

```bash
python -m methods.H3.SolarWM.dl3dv build \
  --train 64 --validation 8 --out logs/h3/prepared/dl3dv.json

CUDA_VISIBLE_DEVICES=0 python -m methods.H3.cli prepare \
  --config methods/H3/SolarWM/config/train.yaml \
  --set manifest=logs/h3/prepared/dl3dv.json

bash methods/H3/SolarWM/scripts/train.sh \
  --set train.manifest=logs/h3/prepared/dl3dv.json
```

Conditions are written to `logs/h3/prepared/conditions/`. For large manifests, run several single-GPU `prepare` processes with `--set prepare_shard=i/n`. Training freezes SolarWM and updates the FP32 router head on one sample per update across the eight-GPU sequence-parallel group. The training configuration controls learning rate, duration, and denoising cost through `train.learning_rate` (1e-4), `train.max_updates` (500), and `train.nfe_weight` (0.1). Checkpoints are written every `train.save_every` (100) updates to `logs/h3/train/checkpoint_step_XXXXXX/` with a `latest` link. `train.resume` continues a run; `train.init_head` only initializes the router head.
