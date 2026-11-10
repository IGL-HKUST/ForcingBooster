window.EXAMPLES = [
  {
    "id": "h3-racing-car",
    "title": "Racing car",
    "description": "一辆赛车在无人的高速路上疾驰，写实风格",
    "category": "interactive",
    "prompt": "Photorealistic cinematic footage of a single sleek silver racing car speeding along a completely empty, immaculate highway through an open arid landscape. A low three-quarter front tracking shot moves smoothly alongside the car, keeping its sculpted body crisp as the road markings and background sweep past with natural motion blur. Rapidly spinning wheels, believable tire contact, subtle body vibration, controlled reflections across polished metal. The highway stretches to a clean distant vanishing point beneath a clear pale blue sky. Warm late-afternoon sunlight, restrained silver and sand color palette, high-end automotive commercial photography, uncluttered composition. Continuous forward motion in one seamless shot, no other vehicles, no pedestrians, no branding or text.",
    "source": {
      "preview": "artifacts/h3-videos/wan/02_racing_car.mp4",
      "exitMap": "artifacts/h3-videos/exitmaps/02_racing_car.pt",
      "baseVideo": "artifacts/h3-videos/02_racing_car.mp4",
      "note": "Base video from MiniMax-H3 (official sampling). The Wan Causal Forcing router (Causal-Forcing 0729 patch_router_ckpt_001000) re-generates each chunk with the clean H3 chunks as history; the preview is its last-step output (wan/), the exit map comes from the router run (exitmaps/).",
      "seed": 1042,
      "trim": "first chunk removed: source frames 0-8 (always four steps); preview re-encoded from frame 9 (libx264 CRF 16), step map rendered with the page legend at 48%"
    },
    "preview": "assets/videos/h3-racing-car-preview.mp4",
    "previewPoster": "assets/videos/h3-racing-car-preview.jpg",
    "overlay": "assets/exitmap/h3-racing-car-overlay.mp4",
    "overlayPoster": "assets/exitmap/h3-racing-car-overlay.jpg",
    "chunkSteps": [
      1.796,
      1.838,
      1.932,
      1.835,
      1.807,
      1.733
    ],
    "chunkFrames": [
      [
        0,
        11
      ],
      [
        12,
        23
      ],
      [
        24,
        35
      ],
      [
        36,
        47
      ],
      [
        48,
        59
      ],
      [
        60,
        70
      ]
    ],
    "chunkLatents": 3,
    "chunkStart": 1,
    "chunkTotal": 7,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/artifacts/h3-videos/exitmaps/02_racing_car.pt"
    }
  },
  {
    "id": "h3-anime-beach",
    "title": "Anime beach",
    "description": "一个动漫女生站在沙滩上中朝我们挥手，背景是干净的天空，日本动漫风格。",
    "category": "interactive",
    "prompt": "A beautiful Japanese anime film shot of a young adult woman standing on a pristine sunlit beach, facing the camera and gently waving hello with one hand, a warm natural smile on her face. She wears a simple white summer dress with pale blue accents; her dark hair and the hem of her dress sway softly in the sea breeze. Medium-wide composition, clearly readable hand gesture, relaxed natural anatomy. Fine pale sand, a thin turquoise ocean horizon, and an expansive clean blue sky with just a few delicate white clouds. Refined hand-drawn linework, elegant cel shading, luminous pastel colors, soft summer sunlight, restrained detail and generous negative space. A steady camera with a very subtle slow push-in, one continuous shot, serene and welcoming, no text or logos.",
    "source": {
      "preview": "artifacts/h3-videos/wan/01_anime_beach.mp4",
      "exitMap": "artifacts/h3-videos/exitmaps/01_anime_beach.pt",
      "baseVideo": "artifacts/h3-videos/01_anime_beach.mp4",
      "note": "Base video from MiniMax-H3 (official sampling). The Wan Causal Forcing router (Causal-Forcing 0729 patch_router_ckpt_001000) re-generates each chunk with the clean H3 chunks as history; the preview is its last-step output (wan/), the exit map comes from the router run (exitmaps/).",
      "seed": 42,
      "trim": "first chunk removed: source frames 0-8 (always four steps); preview re-encoded from frame 9 (libx264 CRF 16), step map rendered with the page legend at 48%"
    },
    "preview": "assets/videos/h3-anime-beach-preview.mp4",
    "previewPoster": "assets/videos/h3-anime-beach-preview.jpg",
    "overlay": "assets/exitmap/h3-anime-beach-overlay.mp4",
    "overlayPoster": "assets/exitmap/h3-anime-beach-overlay.jpg",
    "chunkSteps": [
      1.176,
      1.193,
      1.199,
      1.24,
      1.184,
      1.192
    ],
    "chunkFrames": [
      [
        0,
        11
      ],
      [
        12,
        23
      ],
      [
        24,
        35
      ],
      [
        36,
        47
      ],
      [
        48,
        59
      ],
      [
        60,
        70
      ]
    ],
    "chunkLatents": 3,
    "chunkStart": 1,
    "chunkTotal": 7,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/artifacts/h3-videos/exitmaps/01_anime_beach.pt"
    }
  },
  {
    "id": "h3-minimal-rocket",
    "title": "Minimal rocket",
    "description": "简约的火箭在太空中飞行，背景是无垠的太空，下方是地球",
    "category": "interactive",
    "prompt": "An elegant minimalist 3D animation of a small clean white rocket gliding steadily through the vastness of outer space. The rocket has a simple rounded silhouette, a single dark circular window and understated red fins, with a narrow soft blue engine glow. It travels diagonally upward and to the right while the camera gently tracks beside it. Far below, the graceful curved edge of Earth fills the lower part of the frame, with deep blue oceans, soft white clouds and a thin luminous atmospheric rim. Expansive nearly black space above, a few tiny distant stars, abundant negative space. Refined matte materials, soft cinematic rim lighting, a restrained white, blue and black palette, calm smooth motion, one continuous shot, no lettering or logos.",
    "source": {
      "preview": "artifacts/h3-videos/wan/03_minimal_rocket.mp4",
      "exitMap": "artifacts/h3-videos/exitmaps/03_minimal_rocket.pt",
      "baseVideo": "artifacts/h3-videos/03_minimal_rocket.mp4",
      "note": "Base video from MiniMax-H3 (official sampling). The Wan Causal Forcing router (Causal-Forcing 0729 patch_router_ckpt_001000) re-generates each chunk with the clean H3 chunks as history; the preview is its last-step output (wan/), the exit map comes from the router run (exitmaps/).",
      "seed": 2042,
      "trim": "first chunk removed: source frames 0-8 (always four steps); preview re-encoded from frame 9 (libx264 CRF 16), step map rendered with the page legend at 48%"
    },
    "preview": "assets/videos/h3-minimal-rocket-preview.mp4",
    "previewPoster": "assets/videos/h3-minimal-rocket-preview.jpg",
    "overlay": "assets/exitmap/h3-minimal-rocket-overlay.mp4",
    "overlayPoster": "assets/exitmap/h3-minimal-rocket-overlay.jpg",
    "chunkSteps": [
      1.332,
      1.307,
      1.263,
      1.415,
      1.525,
      1.907
    ],
    "chunkFrames": [
      [
        0,
        11
      ],
      [
        12,
        23
      ],
      [
        24,
        35
      ],
      [
        36,
        47
      ],
      [
        48,
        59
      ],
      [
        60,
        70
      ]
    ],
    "chunkLatents": 3,
    "chunkStart": 1,
    "chunkTotal": 7,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/artifacts/h3-videos/exitmaps/03_minimal_rocket.pt"
    }
  },
  {
    "id": "h3-mars-megastructure",
    "title": "Mars megastructure",
    "description": "宏伟的巨型建筑矗立在火星上，一个宇航员正从远处走来。",
    "category": "interactive",
    "prompt": "A majestic cinematic wide shot on Mars: an immense monumental building rises from a pristine rust-red plain, its towering geometric buttresses and elegant monolithic arches dwarfing the surrounding landscape. One lone astronaut in a white spacesuit walks slowly toward the camera from the middle distance, with the colossal building behind them, establishing an unmistakable sense of scale. Measured footsteps lift tiny puffs of fine red dust in low gravity. Clean architectural silhouette, a sparse rocky foreground, distant ochre mountains and a pale dusty peach sky. Low sunlight grazes the structure, creating long sculptural shadows and subtle atmospheric depth. Photorealistic science-fiction cinematography, restrained colors, monumental yet uncluttered composition, slow gentle camera dolly backward, one continuous shot, no text or logos.",
    "source": {
      "preview": "artifacts/h3-videos/wan/04_mars_megastructure.mp4",
      "exitMap": "artifacts/h3-videos/exitmaps/04_mars_megastructure.pt",
      "baseVideo": "artifacts/h3-videos/04_mars_megastructure.mp4",
      "note": "Base video from MiniMax-H3 (official sampling). The Wan Causal Forcing router (Causal-Forcing 0729 patch_router_ckpt_001000) re-generates each chunk with the clean H3 chunks as history; the preview is its last-step output (wan/), the exit map comes from the router run (exitmaps/).",
      "seed": 3042,
      "trim": "first chunk removed: source frames 0-8 (always four steps); preview re-encoded from frame 9 (libx264 CRF 16), step map rendered with the page legend at 48%"
    },
    "preview": "assets/videos/h3-mars-megastructure-preview.mp4",
    "previewPoster": "assets/videos/h3-mars-megastructure-preview.jpg",
    "overlay": "assets/exitmap/h3-mars-megastructure-overlay.mp4",
    "overlayPoster": "assets/exitmap/h3-mars-megastructure-overlay.jpg",
    "chunkSteps": [
      1.033,
      1.03,
      1.024,
      1.032,
      1.029,
      1.03
    ],
    "chunkFrames": [
      [
        0,
        11
      ],
      [
        12,
        23
      ],
      [
        24,
        35
      ],
      [
        36,
        47
      ],
      [
        48,
        59
      ],
      [
        60,
        70
      ]
    ],
    "chunkLatents": 3,
    "chunkStart": 1,
    "chunkTotal": 7,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/artifacts/h3-videos/exitmaps/04_mars_megastructure.pt"
    }
  },
  {
    "id": "snowfield",
    "title": "Across a snowfield",
    "description": "A skier in a crimson jacket crosses a snowfield under a pink dawn.",
    "category": "interactive",
    "source": {
      "preview": "ForcingBooster/logs/0807_dmd_vidprom_adaptive/checkpoint_eval_train2/outputs/reference_seed10000/Across a snowfield under pink dawn, a skier in a crimson down jacket, fur-lined hood, and mirrored g.mp4",
      "exit_map": "ForcingBooster/logs/0807_dmd_vidprom_adaptive/checkpoint_eval_train2/outputs/reference_seed10000/exit_maps/Across a snowfield under pink dawn, a skier in a crimson down jacket, fur-lined hood, and mirrored g.pt",
      "overlay": "Rendered from the matching exit map; RGB palette at 48% opacity; latent index (frame + 3) // 4.",
      "prompt": "prompts/demos.txt",
      "trim": "first chunk removed: source frames 0-8 (always four steps); videos re-encoded from frame 9 (libx264, CRF 16 preview / 18 step map)"
    },
    "preview": "assets/videos/snowfield-preview.mp4",
    "previewPoster": "assets/videos/snowfield-preview.jpg",
    "overlay": "assets/exitmap/snowfield-overlay.mp4",
    "overlayPoster": "assets/exitmap/snowfield-overlay.jpg",
    "motion": "Text prompt",
    "prompt": "Across a snowfield under pink dawn, a skier in a crimson down jacket, fur-lined hood, and mirrored goggles dominates the frame in HDR 4K with sparkling ice detail. First, she plants her poles and adjusts the strap of a small camera on her chest, breath puffing in the cold. Then she pushes off, carves a wide S-curve through powder, and throws a spray of snow toward the lens as the camera follows low and fast, mountains glowing and the sky vast and stunning.",
    "chunkSteps": [
      2.172,
      2.604,
      2.803,
      2.753,
      3.196,
      3.592
    ],
    "chunkFrames": [
      [
        0,
        11
      ],
      [
        12,
        23
      ],
      [
        24,
        35
      ],
      [
        36,
        47
      ],
      [
        48,
        59
      ],
      [
        60,
        71
      ]
    ],
    "chunkLatents": 3,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/logs/0807_dmd_vidprom_adaptive/checkpoint_eval_train2/outputs/reference_seed10000/exit_maps/Across a snowfield under pink dawn, a skier in a crimson down jacket, fur-lined hood, and mirrored g.pt",
      "overlayAgreement": 1.0
    },
    "chunkStart": 1,
    "chunkTotal": 7
  },
  {
    "id": "canyon-front",
    "title": "Canyon ascent",
    "description": "A climber moves through a sunlit canyon above turquoise water.",
    "category": "interactive",
    "source": {
      "preview": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands domina.mp4",
      "overlay": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/generated_exit_overlay/At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands domina.mp4",
      "prompt": "prompts/demos.txt",
      "trim": "first chunk removed: source frames 0-8 (always four steps); videos re-encoded from frame 9 (libx264, CRF 16 preview / 18 step map)"
    },
    "preview": "assets/videos/canyon-front-preview.mp4",
    "previewPoster": "assets/videos/canyon-front-preview.jpg",
    "overlay": "assets/exitmap/canyon-front-overlay.mp4",
    "overlayPoster": "assets/exitmap/canyon-front-overlay.jpg",
    "motion": "Text prompt",
    "prompt": "At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands dominates the frame in HDR UHD. He grips a hold, pulls up with visible effort, then pauses on a ledge to look down at emerald water as the camera tilts to reveal sunlight slicing through rock. Mineral streaks, wet moss, and fabric seams read sharply. True tones and subtle film grain create a thrilling sense of scale.",
    "chunkSteps": [
      2.232,
      2.177,
      1.807,
      1.522,
      1.296,
      1.236
    ],
    "chunkFrames": [
      [
        0,
        11
      ],
      [
        12,
        23
      ],
      [
        24,
        35
      ],
      [
        36,
        47
      ],
      [
        48,
        59
      ],
      [
        60,
        71
      ]
    ],
    "chunkLatents": 3,
    "chunkSource": {
      "method": "exit map",
      "file": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/exit_maps/At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands domina.pt",
      "overlayAgreement": 1.0
    },
    "chunkStart": 1,
    "chunkTotal": 7
  },
  {
    "id": "violin",
    "title": "A rooftop performance",
    "description": "A violinist performs against an illuminated city skyline.",
    "category": "interactive",
    "source": {
      "preview": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/At a rooftop garden above Seoul at night, a violinist in a black velvet coat and sparkling hairpin s.mp4",
      "overlay": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/generated_exit_overlay/At a rooftop garden above Seoul at night, a violinist in a black velvet coat and sparkling hairpin s.mp4",
      "prompt": "prompts/demos.txt",
      "trim": "first chunk removed: source frames 0-8 (always four steps); videos re-encoded from frame 9 (libx264, CRF 16 preview / 18 step map)"
    },
    "preview": "assets/videos/violin-preview.mp4",
    "previewPoster": "assets/videos/violin-preview.jpg",
    "overlay": "assets/exitmap/violin-overlay.mp4",
    "overlayPoster": "assets/exitmap/violin-overlay.jpg",
    "motion": "Text prompt",
    "prompt": "At a rooftop garden above Seoul at night, a violinist in a black velvet coat and sparkling hairpin stands prominent in ultra-realistic 4K with city lights bokeh. First, she tightens one string and holds still as a breeze lifts her coat hem, distant traffic humming. Then she draws the bow into a bright phrase and takes two steps forward, the camera drifting closer as skyscrapers glitter behind her and lanterns glow, textures rich and the mood intensely alive. while the skyline shimmers softly.",
    "chunkSteps": [
      1.612,
      1.859,
      1.804,
      1.948,
      1.916,
      1.844
    ],
    "chunkFrames": [
      [
        0,
        11
      ],
      [
        12,
        23
      ],
      [
        24,
        35
      ],
      [
        36,
        47
      ],
      [
        48,
        59
      ],
      [
        60,
        71
      ]
    ],
    "chunkLatents": 3,
    "chunkSource": {
      "method": "exit map",
      "file": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/exit_maps/At a rooftop garden above Seoul at night, a violinist in a black velvet coat and sparkling hairpin s.pt",
      "overlayAgreement": 1.0
    },
    "chunkStart": 1,
    "chunkTotal": 7
  },
  {
    "id": "canyon-side",
    "title": "On the rock face",
    "description": "A closer view of the climber, with fine rock textures and moving hands.",
    "category": "interactive",
    "source": {
      "preview": "ForcingBooster/logs/0807_causvid14b_patch-router_4steps/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands domina.mp4",
      "overlay": "ForcingBooster/logs/0807_causvid14b_patch-router_4steps/validation/checkpoint_model_001000/generated_exit_overlay/At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands domina.mp4",
      "prompt": "prompts/demos.txt",
      "trim": "first chunk removed: source frames 0-8 (always four steps); videos re-encoded from frame 9 (libx264, CRF 16 preview / 18 step map)"
    },
    "preview": "assets/videos/canyon-side-preview.mp4",
    "previewPoster": "assets/videos/canyon-side-preview.jpg",
    "overlay": "assets/exitmap/canyon-side-overlay.mp4",
    "overlayPoster": "assets/exitmap/canyon-side-overlay.jpg",
    "motion": "Text prompt",
    "prompt": "At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands dominates the frame in HDR UHD. He grips a hold, pulls up with visible effort, then pauses on a ledge to look down at emerald water as the camera tilts to reveal sunlight slicing through rock. Mineral streaks, wet moss, and fabric seams read sharply. True tones and subtle film grain create a thrilling sense of scale.",
    "chunkSteps": [
      3.365,
      2.716,
      2.569,
      2.404,
      2.167,
      2.171
    ],
    "chunkFrames": [
      [
        0,
        11
      ],
      [
        12,
        23
      ],
      [
        24,
        35
      ],
      [
        36,
        47
      ],
      [
        48,
        59
      ],
      [
        60,
        71
      ]
    ],
    "chunkLatents": 3,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/logs/0807_causvid14b_patch-router_4steps/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/exit_maps/At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands domina.pt",
      "overlayAgreement": 1.0
    },
    "chunkStart": 1,
    "chunkTotal": 7
  },
  {
    "id": "city-night",
    "title": "After the rain",
    "description": "A woman in a silver jacket walks through a rain-soaked city street.",
    "category": "interactive",
    "source": {
      "preview": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/At midsummer in a city street after rain, a woman in a reflective silver jacket and black trousers d.mp4",
      "overlay": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/generated_exit_overlay/At midsummer in a city street after rain, a woman in a reflective silver jacket and black trousers d.mp4",
      "prompt": "prompts/demos.txt",
      "trim": "first chunk removed: source frames 0-8 (always four steps); videos re-encoded from frame 9 (libx264, CRF 16 preview / 18 step map)"
    },
    "preview": "assets/videos/city-night-preview.mp4",
    "previewPoster": "assets/videos/city-night-preview.jpg",
    "overlay": "assets/exitmap/city-night-overlay.mp4",
    "overlayPoster": "assets/exitmap/city-night-overlay.jpg",
    "motion": "Text prompt",
    "prompt": "At midsummer in a city street after rain, a woman in a reflective silver jacket and black trousers dominates the frame in HDR UHD. She walks with confident stride, then pauses to look over her shoulder as the camera pushes close and neon reflections ripple across puddles. Behind her, traffic blurs and steam rises from vents in humid air. Natural skin tones, crisp textures, and subtle film grain make the modern night feel cinematic and real.",
    "chunkSteps": [
      2.0,
      2.202,
      2.28,
      2.46,
      2.161,
      2.447
    ],
    "chunkFrames": [
      [
        0,
        11
      ],
      [
        12,
        23
      ],
      [
        24,
        35
      ],
      [
        36,
        47
      ],
      [
        48,
        59
      ],
      [
        60,
        71
      ]
    ],
    "chunkLatents": 3,
    "chunkSource": {
      "method": "exit map",
      "file": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/exit_maps/At midsummer in a city street after rain, a woman in a reflective silver jacket and black trousers d.pt",
      "overlayAgreement": 1.0
    },
    "chunkStart": 1,
    "chunkTotal": 7
  },
  {
    "id": "running-dog",
    "title": "A dog running happily",
    "description": "A dog runs happily along a dirt path across a meadow.",
    "category": "interactive",
    "source": {
      "preview": "ForcingBooster/logs/0917_cf1b3_adaptive0729_vbench50/videos/router_t05/a001.mp4",
      "exit_map": "ForcingBooster/logs/0917_cf1b3_adaptive0729_vbench50/exit_maps/router_t05/a001.npy",
      "overlay": "Rendered from the matching exit map; RGB palette at 48% opacity; latent index (frame + 3) // 4.",
      "prompt": "logs/0917_cf1b3_adaptive0729_vbench50/manifest.json (a001)",
      "trim": "first chunk removed: source frames 0-8 (always four steps); videos re-encoded from frame 9 (libx264, CRF 16 preview / 18 step map)"
    },
    "preview": "assets/videos/running-dog-preview.mp4",
    "previewPoster": "assets/videos/running-dog-preview.jpg",
    "overlay": "assets/exitmap/running-dog-overlay.mp4",
    "overlayPoster": "assets/exitmap/running-dog-overlay.jpg",
    "motion": "Text prompt",
    "prompt": "a dog running happily",
    "chunkSteps": [
      1.919,
      2.054,
      2.189,
      2.182,
      2.109,
      2.179
    ],
    "chunkFrames": [
      [
        0,
        11
      ],
      [
        12,
        23
      ],
      [
        24,
        35
      ],
      [
        36,
        47
      ],
      [
        48,
        59
      ],
      [
        60,
        71
      ]
    ],
    "chunkLatents": 3,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/logs/0917_cf1b3_adaptive0729_vbench50/exit_maps/router_t05/a001.npy",
      "overlayAgreement": 1.0
    },
    "chunkStart": 1,
    "chunkTotal": 7
  },
  {
    "id": "lw-studio-back-pan-right",
    "title": "Sunlit studio",
    "description": "A painterly studio interior with a large window, a workbench, shelves and hanging plants.",
    "category": "world",
    "motion": "Dolly back + pan right 45°",
    "motionDetail": "Backward dolly and yaw together, cosine ease-in-out",
    "prompt": "A softly painted anime-style interior of a peaceful sunlit studio. A large window reveals a vivid blue sky and billowing white clouds. Warm afternoon sunlight falls across a wooden workbench on the left, a hanging green vine plant, wall-mounted shelves, a pegboard and drawers. Pale shelving stands on the right, and the wooden floor and cream walls carry crisp warm window shadows. Preserve the delicate painterly texture, pastel colors, room layout and straight architectural lines. The room is empty and still, with steady lighting. The camera slowly dollies backward a short distance while smoothly panning to the right by 45 degrees, revealing a slightly wider view and more of the scene on the right. The backward movement follows a straight line opposite the initial viewing direction. A continuous single shot, level horizon, no zoom, no cuts.",
    "frames": 158,
    "fps": 24.0,
    "source": {
      "preview": "logs/1107_solarWM_teaser/results/lingbot_router/studio_back_pan_right/video.mp4",
      "exitMap": "logs/1107_solarWM_teaser/results/lingbot_router/studio_back_pan_right/exit_steps.pt",
      "checkpoint": "LingBot-World 14B router, checkpoint step 1000 (router_nfe_weight 0.03), seed 42"
    },
    "preview": "assets/videos/lw-studio-back-pan-right-preview.mp4",
    "previewPoster": "assets/videos/lw-studio-back-pan-right-preview.jpg",
    "overlay": "assets/exitmap/lw-studio-back-pan-right-overlay.mp4",
    "overlayPoster": "assets/exitmap/lw-studio-back-pan-right-overlay.jpg",
    "chunkSteps": [
      4.0,
      1.303,
      1.329,
      1.366,
      1.488,
      1.498,
      1.527,
      1.385,
      1.24,
      1.115,
      1.039
    ],
    "chunkFrames": [
      [
        0,
        12
      ],
      [
        13,
        28
      ],
      [
        29,
        44
      ],
      [
        45,
        60
      ],
      [
        61,
        76
      ],
      [
        77,
        92
      ],
      [
        93,
        108
      ],
      [
        109,
        124
      ],
      [
        125,
        140
      ],
      [
        141,
        156
      ],
      [
        157,
        157
      ]
    ],
    "chunkLatents": 4,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/logs/1107_solarWM_teaser/results/lingbot_router/studio_back_pan_right/exit_steps.pt",
      "overlayAgreement": 1.0
    },
    "chunkStart": 0,
    "chunkTotal": 11
  },
  {
    "id": "great-wall",
    "title": "Along the Great Wall",
    "description": "A walk along the Great Wall through autumn hills, driven by recorded keyboard input.",
    "category": "world",
    "source": {
      "preview": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/preview_merged/router/official_04.mp4",
      "overlay": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/preview_merged/overlay/official_04.mp4",
      "camera": {
        "kind": "Recorded WASD; joystick derived from camera trajectory",
        "poseSource": "assets/lingbot-world-v2/examples/04/poses.npy",
        "keySource": "assets/lingbot-world-v2/examples/04/wasd_action.npy"
      },
      "prompt": "logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/inputs_new/official_04/prompt.txt"
    },
    "preview": "assets/videos/great-wall-preview.mp4",
    "previewPoster": "assets/videos/great-wall-preview.jpg",
    "overlay": "assets/exitmap/great-wall-overlay.mp4",
    "overlayPoster": "assets/exitmap/great-wall-overlay.jpg",
    "sourceId": "official_04",
    "motion": "Recorded WASD path",
    "prompt": "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder.",
    "chunkSteps": [
      4.0,
      2.304,
      2.251,
      2.519,
      2.384,
      3.186,
      3.513,
      3.822,
      3.15,
      2.852
    ],
    "chunkFrames": [
      [
        0,
        12
      ],
      [
        13,
        28
      ],
      [
        29,
        44
      ],
      [
        45,
        60
      ],
      [
        61,
        76
      ],
      [
        77,
        92
      ],
      [
        93,
        108
      ],
      [
        109,
        124
      ],
      [
        125,
        140
      ],
      [
        141,
        156
      ]
    ],
    "chunkLatents": 4,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/router/exit_maps/official_04.pt",
      "overlayAgreement": 1.0
    },
    "chunkStart": 0,
    "chunkTotal": 10
  },
  {
    "id": "lw-mine-left",
    "title": "Underground mine",
    "description": "A mine cavern with rugged rock walls, work lights, mining trucks and an excavator.",
    "category": "world",
    "motion": "Pan left 30°",
    "motionDetail": "Hold, turn for 2 s, pause for 1 s, turn again",
    "prompt": "A large underground mine cavern with rugged rock walls, work lights, heavy mining trucks, an excavator and a damp gravel road. The camera pans to the left after holding still for one second. It rotates for two seconds, pauses completely for one second, then continues rotating in the same direction for two more seconds. The camera stays at the same position. Continuous realistic footage.",
    "frames": 96,
    "fps": 16.0,
    "source": {
      "preview": "outputs/teaservid/1015_lingbot_camera30_pause_resume/07.mp4",
      "exitMap": "outputs/teaservid/1015_lingbot_camera30_pause_resume/exit_maps/07.npz",
      "checkpoint": "logs/0809_lingbot14B_patch-router/router/checkpoint_step_001000/model.pt"
    },
    "preview": "assets/videos/lw-mine-left-preview.mp4",
    "previewPoster": "assets/videos/lw-mine-left-preview.jpg",
    "overlay": "assets/exitmap/lw-mine-left-overlay.mp4",
    "overlayPoster": "assets/exitmap/lw-mine-left-overlay.jpg",
    "chunkSteps": [
      4.0,
      1.45,
      1.342,
      1.424,
      1.252,
      1.327,
      1.323
    ],
    "chunkFrames": [
      [
        0,
        12
      ],
      [
        13,
        28
      ],
      [
        29,
        44
      ],
      [
        45,
        60
      ],
      [
        61,
        76
      ],
      [
        77,
        92
      ],
      [
        93,
        95
      ]
    ],
    "chunkLatents": 4,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/outputs/teaservid/1015_lingbot_camera30_pause_resume/exit_maps/07.npz",
      "overlayAgreement": 1.0
    },
    "chunkStart": 0,
    "chunkTotal": 7
  },
  {
    "id": "lw-palace-back-pan-left",
    "title": "Palace at sunset",
    "description": "A white neoclassical palace with a gold-tipped dome, reflected in a polished courtyard.",
    "category": "world",
    "motion": "Dolly back + pan left 45°",
    "motionDetail": "Backward dolly and yaw together, cosine ease-in-out",
    "prompt": "A photorealistic wide view of an elegant white neoclassical palace at sunset. A large central dome with a gold finial rises above an illuminated arched glass entrance, flanked by symmetrical colonnades, tall arched windows, and two smaller domed towers. Warm golden interior lights glow against the pale stone facade. The polished stone courtyard reflects the building and the peach-pink sky. Low hedges and distant mountains frame the scene. The architecture remains rigid and stable; the evening light and reflections remain consistent. No people or moving objects. The camera slowly dollies backward a short distance while smoothly panning to the left by 45 degrees, revealing a slightly wider view and more of the scene on the left. The backward movement follows a straight line opposite the initial viewing direction. A continuous single shot, level horizon, no zoom, no cuts.",
    "frames": 158,
    "fps": 24.0,
    "source": {
      "preview": "logs/1107_solarWM_teaser/results/lingbot_router/palace_back_pan_left/video.mp4",
      "exitMap": "logs/1107_solarWM_teaser/results/lingbot_router/palace_back_pan_left/exit_steps.pt",
      "checkpoint": "LingBot-World 14B router, checkpoint step 1000 (router_nfe_weight 0.03), seed 42"
    },
    "preview": "assets/videos/lw-palace-back-pan-left-preview.mp4",
    "previewPoster": "assets/videos/lw-palace-back-pan-left-preview.jpg",
    "overlay": "assets/exitmap/lw-palace-back-pan-left-overlay.mp4",
    "overlayPoster": "assets/exitmap/lw-palace-back-pan-left-overlay.jpg",
    "chunkSteps": [
      4.0,
      1.133,
      1.114,
      1.165,
      1.175,
      1.252,
      1.262,
      1.2,
      1.191,
      1.122,
      1.035
    ],
    "chunkFrames": [
      [
        0,
        12
      ],
      [
        13,
        28
      ],
      [
        29,
        44
      ],
      [
        45,
        60
      ],
      [
        61,
        76
      ],
      [
        77,
        92
      ],
      [
        93,
        108
      ],
      [
        109,
        124
      ],
      [
        125,
        140
      ],
      [
        141,
        156
      ],
      [
        157,
        157
      ]
    ],
    "chunkLatents": 4,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/logs/1107_solarWM_teaser/results/lingbot_router/palace_back_pan_left/exit_steps.pt",
      "overlayAgreement": 1.0
    },
    "chunkStart": 0,
    "chunkTotal": 11
  },
  {
    "id": "dragon-flight",
    "title": "Through the jungle",
    "description": "Riding a dragon toward a castle above the jungle, driven by recorded keyboard input.",
    "category": "world",
    "source": {
      "preview": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/router/official_00.mp4",
      "overlay": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/overlays/official_00.mp4",
      "exit_map": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/router/exit_maps/official_00.pt",
      "camera": {
        "kind": "Recorded WASD; joystick derived from camera trajectory",
        "poseSource": "assets/lingbot-world-v2/examples/00/poses.npy",
        "keySource": "assets/lingbot-world-v2/examples/00/wasd_action.npy"
      },
      "prompt": "logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/inputs_new/official_00/prompt.txt"
    },
    "preview": "assets/videos/dragon-flight-preview.mp4",
    "previewPoster": "assets/videos/dragon-flight-preview.jpg",
    "overlay": "assets/exitmap/dragon-flight-overlay.mp4",
    "overlayPoster": "assets/exitmap/dragon-flight-overlay.jpg",
    "sourceId": "official_00",
    "motion": "Recorded WASD path",
    "prompt": "This is a first-person-view video from the perspective of a rider gripping the reins of a large, green-scaled dragon flying through a dense jungle. The dragon's neck and the rider's leather-gloved hands remain at the exact centre of the frame at constant size and distance. Neither the dragon nor the camera moves on its own; arrow-key look-input is the only source of camera motion, orbiting around the stable composite subject only while held. With no event key pressed, the rider holds the reins steady and the dragon maintains a hovering, interaction-ready posture",
    "chunkSteps": [
      4.0,
      2.69,
      2.878,
      3.236,
      3.234,
      3.166,
      3.103,
      3.161,
      3.422,
      3.408
    ],
    "chunkFrames": [
      [
        0,
        12
      ],
      [
        13,
        28
      ],
      [
        29,
        44
      ],
      [
        45,
        60
      ],
      [
        61,
        76
      ],
      [
        77,
        92
      ],
      [
        93,
        108
      ],
      [
        109,
        124
      ],
      [
        125,
        140
      ],
      [
        141,
        156
      ]
    ],
    "chunkLatents": 4,
    "chunkSource": {
      "method": "decoded from step-map overlay (label box excluded)",
      "file": "assets/exitmap/dragon-flight-overlay.mp4"
    },
    "chunkStart": 0,
    "chunkTotal": 10
  },
  {
    "id": "lw-cabin-right",
    "title": "Lakeside cabin",
    "description": "A wooden cabin by a mountain lake at dusk, with a porch light and a garden campfire.",
    "category": "world",
    "motion": "Pan right 30°",
    "motionDetail": "Hold, turn for 2 s, pause for 1 s, turn again",
    "prompt": "A cozy wooden cabin beside a mountain lake at dusk, a warm porch light, a glowing campfire in the garden, trees and distant mountains. The camera pans to the right after holding still for one second. It rotates for two seconds, pauses completely for one second, then continues rotating in the same direction for two more seconds. The camera stays at the same position. Continuous realistic footage.",
    "frames": 96,
    "fps": 16.0,
    "source": {
      "preview": "outputs/teaservid/1015_lingbot_camera30_pause_resume/02.mp4",
      "exitMap": "outputs/teaservid/1015_lingbot_camera30_pause_resume/exit_maps/02.npz",
      "checkpoint": "logs/0809_lingbot14B_patch-router/router/checkpoint_step_001000/model.pt"
    },
    "preview": "assets/videos/lw-cabin-right-preview.mp4",
    "previewPoster": "assets/videos/lw-cabin-right-preview.jpg",
    "overlay": "assets/exitmap/lw-cabin-right-overlay.mp4",
    "overlayPoster": "assets/exitmap/lw-cabin-right-overlay.jpg",
    "chunkSteps": [
      4.0,
      1.443,
      1.348,
      1.508,
      1.179,
      1.278,
      1.296
    ],
    "chunkFrames": [
      [
        0,
        12
      ],
      [
        13,
        28
      ],
      [
        29,
        44
      ],
      [
        45,
        60
      ],
      [
        61,
        76
      ],
      [
        77,
        92
      ],
      [
        93,
        95
      ]
    ],
    "chunkLatents": 4,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/outputs/teaservid/1015_lingbot_camera30_pause_resume/exit_maps/02.npz",
      "overlayAgreement": 1.0
    },
    "chunkStart": 0,
    "chunkTotal": 7
  },
  {
    "id": "lw-forest-cave-up",
    "title": "Forest cave",
    "description": "A dark cave entrance in dense forest, with mossy rocks, ferns and vines.",
    "category": "world",
    "motion": "Tilt up 30°",
    "motionDetail": "Hold, turn for 2 s, pause for 1 s, turn again",
    "prompt": "A dark cave entrance surrounded by dense green forest, moss-covered rocks, ferns, vines and a narrow earthy path. The camera tilts upward after holding still for one second. It rotates for two seconds, pauses completely for one second, then continues rotating in the same direction for two more seconds. The camera stays at the same position. Continuous realistic footage.",
    "frames": 96,
    "fps": 16.0,
    "source": {
      "preview": "outputs/teaservid/1015_lingbot_camera30_pause_resume/12.mp4",
      "exitMap": "outputs/teaservid/1015_lingbot_camera30_pause_resume/exit_maps/12.npz",
      "checkpoint": "logs/0809_lingbot14B_patch-router/router/checkpoint_step_001000/model.pt"
    },
    "preview": "assets/videos/lw-forest-cave-up-preview.mp4",
    "previewPoster": "assets/videos/lw-forest-cave-up-preview.jpg",
    "overlay": "assets/exitmap/lw-forest-cave-up-overlay.mp4",
    "overlayPoster": "assets/exitmap/lw-forest-cave-up-overlay.jpg",
    "chunkSteps": [
      4.0,
      1.916,
      1.633,
      1.972,
      1.508,
      1.606,
      1.673
    ],
    "chunkFrames": [
      [
        0,
        12
      ],
      [
        13,
        28
      ],
      [
        29,
        44
      ],
      [
        45,
        60
      ],
      [
        61,
        76
      ],
      [
        77,
        92
      ],
      [
        93,
        95
      ]
    ],
    "chunkLatents": 4,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/outputs/teaservid/1015_lingbot_camera30_pause_resume/exit_maps/12.npz",
      "overlayAgreement": 1.0
    },
    "chunkStart": 0,
    "chunkTotal": 7
  },
  {
    "id": "classroom",
    "title": "Through the classroom",
    "description": "The camera moves between rows of desks in an empty classroom.",
    "category": "world",
    "source": {
      "preview": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/preview_merged/router/dl3dv_00039.mp4",
      "overlay": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/preview_merged/overlay/dl3dv_00039.mp4",
      "camera": {
        "kind": "Derived direction indicators from camera trajectory; no recorded keyboard input",
        "sampleMetadata": "logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/inputs/dl3dv_00039/metadata.json",
        "sourceFrames": [
          107,
          221
        ],
        "sourceFps": 24,
        "interpolation": "OpenGL to OpenCV; linear translation and SLERP rotation at 16 fps"
      },
      "prompt": "/mnt/pfs/data/zhengzeyi/dl3dv10k/caption_json_files/10K/9a16b921d4412a0278176be96cb526532734a469e5eb627cd33c51dcd26c3df8/caption.json"
    },
    "preview": "assets/videos/classroom-preview.mp4",
    "previewPoster": "assets/videos/classroom-preview.jpg",
    "overlay": "assets/exitmap/classroom-overlay.mp4",
    "overlayPoster": "assets/exitmap/classroom-overlay.jpg",
    "sourceId": "dl3dv_00039",
    "motion": "DL3DV camera path",
    "prompt": "The video showcases a spacious and well-lit classroom with rows of green desks and chairs, each neatly arranged. The room features a large blackboard adorned with colorful decorations and a red banner with Chinese characters above it. The walls are decorated with educational posters and a window that allows natural light to fill the space. The ceiling is equipped with multiple ceiling fans, and the floor is covered with a light-colored tile. The classroom appears to be empty, suggesting it is either a quiet moment during a break or a preparation for a new lesson.",
    "chunkSteps": [
      4.0,
      3.17,
      2.723,
      2.803,
      3.284
    ],
    "chunkFrames": [
      [
        0,
        12
      ],
      [
        13,
        28
      ],
      [
        29,
        44
      ],
      [
        45,
        60
      ],
      [
        61,
        76
      ]
    ],
    "chunkLatents": 4,
    "chunkSource": {
      "method": "decoded from step-map overlay (label box excluded)",
      "file": "assets/exitmap/classroom-overlay.mp4"
    },
    "chunkStart": 0,
    "chunkTotal": 5
  },
  {
    "id": "decor-store",
    "title": "Aisles of decorations",
    "description": "The camera weaves between decoration displays in a shop.",
    "category": "world",
    "source": {
      "preview": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/preview_merged/router/dl3dv_00019.mp4",
      "overlay": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/preview_merged/overlay/dl3dv_00019.mp4",
      "camera": {
        "kind": "Derived direction indicators from camera trajectory; no recorded keyboard input",
        "sampleMetadata": "logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/inputs/dl3dv_00019/metadata.json",
        "sourceFrames": [
          136,
          250
        ],
        "sourceFps": 24,
        "interpolation": "OpenGL to OpenCV; linear translation and SLERP rotation at 16 fps"
      },
      "prompt": "/mnt/pfs/data/zhengzeyi/dl3dv10k/caption_json_files/10K/9878106ffd83398c264a308b8346652a99397cc9c2c59e48899b430088144622/caption.json"
    },
    "preview": "assets/videos/decor-store-preview.mp4",
    "previewPoster": "assets/videos/decor-store-preview.jpg",
    "overlay": "assets/exitmap/decor-store-overlay.mp4",
    "overlayPoster": "assets/exitmap/decor-store-overlay.jpg",
    "sourceId": "dl3dv_00019",
    "motion": "DL3DV camera path",
    "prompt": "The video begins with a wide shot of a Halloween-themed store, showcasing various decorations and products. The camera pans across the store, highlighting a display of Halloween-themed items such as spider webs, pumpkins, and witch hats. The store is filled with festive decorations, including a large black and gold garland and a Christmas tree adorned with Halloween ornaments. The camera then focuses on a table with a collection of Halloween decorations, including a sign that reads 'Of course the cobwebs are part of the decor.' The camera continues to move, revealing more of the store's interior, including a display of Halloween-themed merchandise and a small Halloween-themed gift shop.",
    "chunkSteps": [
      4.0,
      3.526,
      3.916,
      3.87,
      3.566
    ],
    "chunkFrames": [
      [
        0,
        12
      ],
      [
        13,
        28
      ],
      [
        29,
        44
      ],
      [
        45,
        60
      ],
      [
        61,
        76
      ]
    ],
    "chunkLatents": 4,
    "chunkSource": {
      "method": "decoded from step-map overlay (label box excluded)",
      "file": "assets/exitmap/decor-store-overlay.mp4"
    },
    "chunkStart": 0,
    "chunkTotal": 5
  },
  {
    "id": "toy-display",
    "title": "Around the toy display",
    "description": "The camera circles a red toy display in a store aisle.",
    "category": "world",
    "source": {
      "preview": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/preview_merged/router/dl3dv_00079.mp4",
      "overlay": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/preview_merged/overlay/dl3dv_00079.mp4",
      "camera": {
        "kind": "Derived direction indicators from camera trajectory; no recorded keyboard input",
        "sampleMetadata": "logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/inputs/dl3dv_00079/metadata.json",
        "sourceFrames": [
          102,
          216
        ],
        "sourceFps": 24,
        "interpolation": "OpenGL to OpenCV; linear translation and SLERP rotation at 16 fps"
      },
      "prompt": "/mnt/pfs/data/zhengzeyi/dl3dv10k/caption_json_files/10K/9e7a4d73f72909ea688ddc71cfbb8232f18a341544d6aa595dc85df85f5ecc51/caption.json"
    },
    "preview": "assets/videos/toy-display-preview.mp4",
    "previewPoster": "assets/videos/toy-display-preview.jpg",
    "overlay": "assets/exitmap/toy-display-overlay.mp4",
    "overlayPoster": "assets/exitmap/toy-display-overlay.jpg",
    "sourceId": "dl3dv_00079",
    "motion": "DL3DV camera path",
    "prompt": "The video begins with a view of a colorful toy display, featuring a variety of Schleich dinosaur figurines arranged on a multi-tiered shelf. The camera pans around the display, revealing the intricate details of the toys, including their different colors and sizes. The surrounding area is filled with other toys and merchandise, creating a vibrant and engaging environment. The camera then moves to show the floor, which is covered in a patterned carpet, and the walls, which are painted in bright colors. The video captures the lively atmosphere of the toy store, with the focus on the Schleich dinosaur collection.",
    "chunkSteps": [
      4.0,
      3.337,
      3.682,
      3.585,
      3.587
    ],
    "chunkFrames": [
      [
        0,
        12
      ],
      [
        13,
        28
      ],
      [
        29,
        44
      ],
      [
        45,
        60
      ],
      [
        61,
        76
      ]
    ],
    "chunkLatents": 4,
    "chunkSource": {
      "method": "exit map",
      "file": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/router/exit_maps/dl3dv_00079.pt",
      "overlayAgreement": 1.0
    },
    "chunkStart": 0,
    "chunkTotal": 5
  }
];
