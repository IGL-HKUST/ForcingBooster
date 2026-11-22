# ForcingBooster project page

[arXiv](https://arxiv.org/abs/2610.12625)

Static project page on the `page` branch. This branch contains the website only; model and training code remain on `main`.

## Preview

Open `index.html`, or serve this directory with any static HTTP server:

```sh
python -m http.server 8000
```

No build step, package installation, CDN, or external font request is required.

CSS, JavaScript, and hero media URLs in `index.html` include the first 12 characters of each file’s SHA-256 hash as a `?v=` parameter. When changing an asset, update its URL hash in the same commit so returning browsers load the matching version instead of a cached earlier layout or script.

## GitHub Pages

Publish the `page` branch of [IGL-HKUST/ForcingBooster](https://github.com/IGL-HKUST/ForcingBooster). In **Settings → Pages**, select **Deploy from a branch → page → /(root)**. The public site is [igl-hkust.github.io/ForcingBooster/](https://igl-hkust.github.io/ForcingBooster/). All URLs are relative and work under this repository subpath.

## Files

- `index.html`: project page.
- `assets/style.css`: visual design (system fonts, light and dark sections, no external requests).
- `assets/hero.js`: hero overview card (video clock, prompt / camera panels, captions, and the centred arrow).
- `assets/app.js`: navigation, scroll reveal, and the two example players.
- `assets/config/data.js`: metadata, prompts, and chunk-wise average denoising steps (`chunkSteps`, `chunkFrames`, `chunkStart`, `chunkTotal`, `chunkSource`) for all 20 gallery examples; `sources.json` is the same list for reference.
- `assets/config/camera-data.js`: per-frame WASD / joystick traces for the 10 world-model examples.
- `assets/config/hero-world-motion.js`, `hero-source.json`: hero camera controls and provenance.
- `assets/videos/`: generated videos and posters, the hero video, and the four How it works clips (`how-it-works-1..4`).
- `assets/config/how-it-works-source.json`: where each How it works clip, its render script, and its source material live.
- `assets/exitmap/`: step-map overlays and posters.
- `.nojekyll`: serve the files directly with GitHub Pages.

The existing `.gitignore` is preserved. Ignored local training code, models, logs, and experiment files are not website assets and must not be removed when maintaining this branch.

## Page structure

Section order: hero, How it works, Demo (two galleries), abstract with the two highlights, citation. The top bar links to Overview, How it works, and Demo.

1. Hero: a two-column layout in the style of Apple product pages. Left: title, authors, affiliations, and links (Explore the results, How it works, Code, Checkpoints, arXiv). Right: the overview card, stacked vertically from bottom to top: the input panel (text prompt / camera controls; when the example switches, the outgoing panel slides out to the left and the next one enters from the right), Existing Method, the Router, and the patch-wise denoising video, joined by one centred striped arrow that runs up through the gaps and points to the video. The green/red explanation captions appear in the row above the video (replacing its label during the holds), so they never cover the step colors. Below 960 px the text sits centred above the card. The animation alternates the original canyon climber (frame 9, the first frame of its second chunk) and the first world-model demo, Sunlit studio (`lw-studio-back-pan-right`, frame 55, played at 16 frames/s with its S + turn-right camera controls). The studio clip is scaled to the full 360 px height and centre-cropped in width, so no white bars appear above or below it; its exit map is scaled and cropped the same way. Rendered by `artifacts/page-tooling/render_hero_v2.py` with the same 28 s timeline, 2×2 max-exit grouping, and 3 s green/red holds as before; `hero-world-motion.js` carries the start frame and playback rate that `hero.js` reads, and `hero-source.json` records the sources.
2. How it works: four steps in the format of the SMD project page method overview (https://zjwfufu.github.io/SMD-Page/): a 4:3 clip on a light-grey panel on the left, a Georgia heading and a short Arial paragraph on the right (complex patches in red, simple patches in green, the router and its inputs in indigo), and a row of step tabs below whose black underline fills with playback progress. A clip plays only while at least 45% of it is on screen; when it ends the next step starts, and after the fourth step the overview returns to the first. Each clip has pause/play and fullscreen buttons; tabs, arrow keys, Home and End switch steps. The clips are the Codex renders in `artifacts/world_model_explainer/` (`render_section1_v41.py`, `render_no_booster_block.py` → v36, `render_section3_v39.py`, `render_section4_v40.py`), remuxed without re-encoding; see `how-it-works-source.json`. The step 4 clip is `forcingbooster_section4_efficiency_v41.mp4` (2026-11-22, replacing v40); its numbers (about 50%, 1.68×, 1.31×) match the step 4 text.
3. Demo: one section title with two subsections (`#interactive`, `#world`). The title, the subsection headers, and the players share one left edge (the player width). Each subsection opens with a thin rule, its heading, and its example count on the right (filled in by `app.js` from `data.js`). Interactive Video Generation: 10 examples; `h3-racing-car` is shown first. Video World Model: 10 LingBot-World examples, one camera action per scene; `lw-studio-back-pan-right` is shown first.
4. Abstract with the two highlights (50% fewer denoising steps; 1.68× generator speedup on Causal Forcing 1.3B, from the main results table in `latex/sec/exp.tex`: generator forward time only, excluding the first chunk and VAE decoding. End-to-end time in the same table is 11.15 s → 8.56 s, about 1.30×).
5. Citation.

Both galleries use the same two-column layout: the current example on the left (generated video left of a draggable divider, its step map on the right with the 1–4 step legend in its label, one shared transport, then the prompt and, for world models, the camera input), and a two-column grid of candidates on the right. The left column stays in view while the candidate list scrolls; below 900 px the player sits above the grid. A player starts muted when it scrolls into view and pauses when it leaves; only one player runs at a time.

The transport ends with a readout for the chunk on screen, shown in the button blue: its mean denoising steps and its speedup, defined as 4 ÷ mean steps (the four-step baseline's denoising steps over this chunk's average; denoising steps only, not wall-clock time). Means are taken over all patches of the chunk's latent frames. Chunks are 3 latent frames for the Causal Forcing / Wan examples and 4 for LingBot-World, and pixel frame `f` belongs to latent frame `(f + 3) // 4`.

The interactive-generation demos start at their second chunk: the first chunk (source frames 0–8) always uses four steps, so it is cut from both the generated video and the step map (frame-exact cut, re-encoded with libx264 CRF 16 / 18; recorded in each example's `source.trim`). World-model demos keep their first chunk. The hero's climber starts at frame 9, the first frame of its second chunk.

Sources: 13 examples use their original exit maps (path in `chunkSource.file`); each was checked against the page's step-map overlay and agrees on 100% of patches outside the debug label box. The exit maps of `dragon-flight`, `classroom`, and `decor-store` no longer exist (their links point to removed storage), so their steps are decoded from the step-map overlay with a linear blend model (`overlay = a · preview + b[step]`) fitted on `great-wall` and `toy-display`, which share the same overlay renderer. Cross-validation between those two gives 100% patch accuracy; the 80 patches under the debug label (about 5%) cannot be read and are left out, which shifts chunk means by at most 0.10 in that test. The decoded first chunks come out at exactly 4.0.

## World-model examples

One camera action is kept per scene:

- `great-wall` (official_04), `dragon-flight` (official_00): recorded WASD paths. `classroom`, `decor-store`, `toy-display` (DL3DV): camera paths from the dataset. Prompts are the validation inputs of `logs/0809_lingbot14B_patch-router`, verified against the sha256 in its `inputs/manifest.json`.
- `lw-mine-left`, `lw-cabin-right`, `lw-forest-cave-up`: `outputs/teaservid/1015_lingbot_camera30_pause_resume/`, router checkpoint `logs/0809_lingbot14B_patch-router/router/checkpoint_step_001000`, 832×480, 16 FPS, 96 frames. The camera holds for 1 s, turns 15° over 2 s, pauses 1 s, then turns another 15°. Camera widgets show the rotation direction only (no translation).
- `lw-palace-back-pan-left`, `lw-studio-back-pan-right`: `logs/1107_solarWM_teaser/results/lingbot_router`, 832×464, 24 FPS, 158 frames, seed 42. 45° yaw with a backward dolly, cosine ease-in-out. The joystick follows the per-frame yaw rate from each `trajectory.json`; S is lit while the dolly moves.

Interactive-generation prompts come from `prompts/demos.txt` (matched by the 100-character prefix in each source filename) and, for `running-dog`, the VBench manifest of `logs/0917_cf1b3_adaptive0729_vbench50` (sample `a001`).

Generated videos are remuxed without re-encoding (video stream only, fast-start). Step maps are rendered from the saved exit steps (`exit_maps/*.npz`, `exit_steps.pt`) with the legend palette at 48% over the video (`clean × 0.52 + color × 0.48`), one color per 16×16 patch, pixel frame `f` using latent frame `(f + 3) // 4`.

## H3 interactive examples

`h3-racing-car`, `h3-anime-beach`, `h3-minimal-rocket`, `h3-mars-megastructure` come from `artifacts/h3-videos` (the first four of its five prompts; `prompts.json` holds the original Chinese request and the English prompt). The base videos were generated by MiniMax-H3 with its official sampling (5 s, 16 FPS, 480p). The Wan Causal Forcing router (`Causal-Forcing/logs/0729_dmd_vidprom_adaptive/patch_router_ckpt_001000`) then re-generated every chunk with the clean H3 chunks as history (`run_wan0729.py`, protocol `pure_noise_router_with_t0_GT_history`, first chunk forced to four steps). The page shows `wan/<id>.mp4` (that run's last-step output) from frame 9 on, and a step map rendered from `exitmaps/<id>.pt` (the router's exit steps, 7 chunks × 3 latent frames). The per-chunk means of chunks 2–7 match `later_mean_steps` in `exitmaps/summary.json`.
