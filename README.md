# ForcingBooster: Adaptive Patch-wise Denoising for Fast Autoregressive Video Diffusion

We introduce ForcingBooster, a general framework that dynamically allocates denoising steps to individual patches: easy patches stop early to improve efficiency, while difficult patches receive more refinement steps. This design accelerates generation with negligible quality degradation. Our method is compatible with most existing few-step autoregressive diffusion models, requiring only a lightweight routing module with modest training.

[Project Page](https://igl-hkust.github.io/ForcingBooster)  |  [Checkpoints](https://huggingface.co/EXCAI/ForcingBooster)

## Setup

```bash
conda create -n forcingbooster python=3.10 -y
conda activate forcingbooster
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

Ten example inference prompts are included in [`assets/inference.txt`](assets/inference.txt), one prompt per line. For training, use [`vidprom_filtered_extended.txt` from Self-Forcing](https://huggingface.co/gdhe17/Self-Forcing/blob/main/vidprom_filtered_extended.txt), downloaded as described below. We evaluated our method on a Linux server equipped with 1TB of RAM and 8 NVIDIA A800 (80GB) GPUs.
## Inference

We use Causal-Forcing checkpoint as example. The Wan 1.3B router configuration defaults to the released [ForcingBooster](https://huggingface.co/EXCAI/ForcingBooster) checkpoint and [Wan base components](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B). Other checkpoints will be released to our [HuggingFace](https://huggingface.co/EXCAI/ForcingBooster) repo.

```bash
bash methods/Wan21/Wan1B3/scripts/infer.sh \
  --set data.source=assets/inference.txt \
  --set artifacts.validation_output_dir=outputs/inference \
  --set runtime.seed=42
```

This runs on one GPU and writes MP4 files to `outputs/inference/`. The default output is 81 video frames at 480 × 832 and 16 FPS. To use a complete local download of ForcingBooster, append `--set weights.checkpoint=ckpts/my-forcingbooster-download`. To use your own trained model, append `--set weights.checkpoint=outputs/wan21-1b3/router/latest`.

Other methods use the same source fields:

| Method | Base or official source (`weights.model`) | Checkpoints |
|---|---|---|
| Wan 1.3B / 14B | `Wan-AI/Wan2.1-T2V-1.3B` | [ForcingBooster-Wan](https://huggingface.co/EXCAI/ForcingBooster/tree/main/Causal-Forcing) |
| LingBot  14B |  `robbyant/lingbot-world-v2-14b-causal-fast` | [ForcingBooster-LingBot](https://huggingface.co/EXCAI/ForcingBooster) |
| LongLive | `Efficient-Large-Model/LongLive-1.3B` | To be released |
| H3 | `junchaoh-cs/SolarWM-H3-33B` | To be released |

LingBot 14B router inference defaults to `LingBot-World/model.pt` from ForcingBooster. LongLive and H3 routers also require explicit checkpoints.
```bash
bash methods/LingBotWorld/LingBotV2_14B/scripts/infer.sh \
  --set weights.model=ckpts/my-lingbot-download
```

## Training

The two Wan stages below use eight GPUs and text prompts only. Their configurations are in `methods/Wan21/Wan1B3/config/`; use `--set` to override training parameters. `weights.init` selects initial generator weights, `weights.teacher` selects the DMD teacher.

Download the training prompts before launching either stage:

```bash
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir dataset
```

### Adaptive DMD (optional)

To achieve better performance, patch-wise adaptive DMD fine-tunes the generator with random exit steps before router training. It defaults to [Causal Forcing](https://huggingface.co/zhuhz22/Causal-Forcing) initialization and 50 training iterations. 

```bash
torchrun --standalone --nproc-per-node=8 -m methods.cli train \
  --model wan21-1b3 --stage adaptive_dmd \
  --config methods/Wan21/Wan1B3/config/adaptive_dmd.yaml \
  --set data.source=dataset/vidprom_filtered_extended.txt \
  --set artifacts.output_dir=outputs/wan21-1b3/adaptive_dmd
```

### Router training

Router training freezes the generator and updates the routing head. The default initialization is Causal Forcing:

```bash
torchrun --standalone --nproc-per-node=8 -m methods.cli train \
  --model wan21-1b3 --stage router \
  --config methods/Wan21/Wan1B3/config/router.yaml \
  --set data.source=dataset/vidprom_filtered_extended.txt \
  --set artifacts.output_dir=outputs/wan21-1b3/router
```

After adaptive DMD, append `--set weights.init=outputs/wan21-1b3/adaptive_dmd/latest`. The router configuration controls learning rate, duration, and denoising cost through `params.learning_rate`, `params.max_train_steps`, and `params.nfe_weight`.
