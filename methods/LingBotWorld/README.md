# LingBot-World v2

Patch router inference and training for LingBot-World v2 causal-fast. `LingBotV2_14B` supports router inference and training on eight GPUs; `LingBotV2_1B3` supports four-step inference on one GPU. Variant details are in [LingBotV2_14B](LingBotV2_14B/README.md) and [LingBotV2_1B3](LingBotV2_1B3/README.md).

## Inference

The 14B router defaults to `LingBot-World/model.pt` from [ForcingBooster](https://huggingface.co/EXCAI/ForcingBooster) and the upstream [lingbot-world-v2-14b-causal-fast](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast) weights.

```bash
torchrun --standalone --nproc-per-node=8 -m methods.LingBotWorld.cli infer \
  --config methods/LingBotWorld/LingBotV2_14B/config/router.yaml
```

This runs on eight GPUs over the examples in `assets/lingbot-world-v2/examples` (image, prompt, camera poses and intrinsics) and writes `<id>.mp4` files to `logs/lingbot-world-v2-14b/router/inference/`. Each video has 77 frames at 480 × 832. To select examples, append `--set 'params.inference_sample_ids=["00"]'`. To use a complete local download of the upstream repository, append `--set weights.model=ckpts/my-lingbot-download`. To use your own trained router, append `--set weights.checkpoint=logs/lingbot-world-v2-14b/router/latest`.

Fixed four-step baselines without a router:

```bash
bash methods/LingBotWorld/LingBotV2_14B/scripts/infer.sh   # 14B, eight GPUs
bash methods/LingBotWorld/LingBotV2_1B3/scripts/infer.sh   # 1.3B, one GPU with CPU offload
```

They write to `logs/lingbot-world-v2-14b/baseline/inference/` and `logs/lingbot-world-v2-1.3b/baseline/inference/`. The 1.3B transformer comes from `robbyant/lingbot-world-v2-1.3b-causal-fast`; its T5, VAE and tokenizer come from the 14B repository (`weights.assets`).

## Training

Router training uses eight GPUs and DL3DV-10K, expected at `dataset/dl3dv10k` with `images+poses/` and `caption_json_files/` (override with `--set data.source=...`). Scenes need at least 115 source frames; every 20th valid scene is held out for validation.

```bash
bash methods/LingBotWorld/LingBotV2_14B/scripts/prepare.sh   # optional: count valid scenes, writes nothing
bash methods/LingBotWorld/LingBotV2_14B/scripts/train.sh
```

Training starts from the upstream weights, freezes the generator and updates the router head with one sample per GPU (global batch size 8). The router configuration controls learning rate, duration, and denoising cost through `params.learning_rate` (1e-4), `params.max_train_steps` (1000), and `params.nfe_weight` (0.03). Only the final checkpoint is written, to `logs/lingbot-world-v2-14b/router/checkpoint_step_001000/` with a `latest` link. `weights.init` loads initial generator/router weights; `artifacts.resume_checkpoint` resumes a run. The two are mutually exclusive.
