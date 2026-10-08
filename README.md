# ForcingBooster project page

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
- `assets/style.css`, `assets/app.js`, `assets/hero.js`, `assets/favicon.svg`: presentation and playback.
- `assets/config/`: gallery metadata, camera controls, and source configuration.
- `assets/videos/`: generated videos and matching posters.
- `assets/exitmap/`: denoising schedule videos and matching posters.
- `.nojekyll`: serve the files directly with GitHub Pages.

The existing `.gitignore` is preserved. Ignored local training code, models, logs, and experiment files are not website assets and must not be removed when maintaining this branch.

## Content

- Two example cards per row on desktop, with a single-column layout on smaller screens.
- 6 Interactive Video Generation examples.
- 4 Video World Model examples.
- Every example pairs a generated video with its original denoising schedule overlay, with shared play/pause, seeking, restart, and a step-color legend to the right of each Denoising Schedule heading.
- Playback starts on demand. Only one example plays at a time; hidden examples pause.
- `assets/config/data.js` supplies gallery metadata. `assets/config/sources.json` records source project and log paths, not runtime dependencies.
- `assets/videos/` contains the 10 gallery previews and the hero animation, each with a poster. `assets/exitmap/` contains the 10 denoising schedule videos and their posters. Preview videos are losslessly remuxed with fast-start metadata. The two added overlays are rendered from their matching exit maps using the legend palette and the causal VAE mapping `(frame + 3) // 4`.

The title and method text are based on the local manuscript. The author list and affiliations follow the supplied reference, with VAST added for Zekai Gu and Peng Wang. Publication links and citation metadata are omitted until provided. The layout is inspired by https://igl-hkust.github.io/GO-Renderer/; no reference-site code or media is copied.

World Model cards include SVG WASD/joystick indicators synchronized to the video's current time, including pause, seeking, and restart. `assets/config/camera-data.js` contains sampled motion data and provenance. The Great Wall example uses the recorded WASD inputs; the three DL3DV examples have no keyboard recording, so direction indicators are derived from local camera translation. Their OpenGL poses are converted to OpenCV and resampled from 24 to 16 fps using linear translation and SLERP rotation. Joystick direction follows relative camera yaw/pitch with three-frame smoothing.

The hero overview is a looping alternating text prompt / camera controls → Existing Method + ForcingBooster Router → patch-wise denoising card. `assets/hero.js` synchronizes typing to the 22-second video clock and automatically loops while visible, with no play button; offscreen playback pauses to save resources. A single black-and-green striped line connects prompt and video. `assets/config/hero-source.json` records the real exit-map source; the noise transitions are illustrative composites, not saved intermediate model outputs. Visualization groups each 2×2 block of real patches and uses its maximum exit step. Both initial denoising and subsequent chunks use a uniform opacity fade within each visual patch; slower groups finish later.

The hero alternates the original female-climber example with the official_02 city walkthrough. After the first clip, the prompt fades upward; recorded WASD/trajectory controls slide in and follow the world video. The loop then transitions back to text. `assets/config/hero-world-motion.js` contains the original 157-frame camera controls. The two method modules use separate rounded rectangles without a puzzle tab.
