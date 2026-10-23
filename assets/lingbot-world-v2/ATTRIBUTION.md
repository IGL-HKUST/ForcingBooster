# LingBot-World v2 official test inputs

The files under `examples/00` through `examples/05` are unmodified copies of
the official LingBot-World v2 examples at Git commit
`2648877f763a06cc743bcd919936da4d25f12e7b`:

https://github.com/robbyant/lingbot-world-v2/tree/2648877f763a06cc743bcd919936da4d25f12e7b/examples

Copyright and attribution remain with Robbyant and the upstream contributors.
These assets are provided under CC BY-NC-SA 4.0; see `LICENSE.txt`. Their use in
this Apache-2.0 repository does not relicense them, and they are intended for
non-commercial research and reproducibility testing.

The upstream repository does not publish paired generated outputs. Therefore
this directory contains inputs only. Outputs generated from them must not be
described as official upstream outputs.

Examples `03` and `04` do not contain `prompt.txt` in the upstream tree. Their
`prompt.txt` files here hold the prompts verbatim from the fixed revision's
`run_fast.sh` and `generate.py`, respectively. The copied action arrays are kept
for provenance, but the released causal-fast Python path currently reads only
`image.jpg`, `poses.npy`, `intrinsics.npy`, and the prompt.
