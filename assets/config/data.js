window.EXAMPLES = [
  {
    "id": "canyon-front",
    "title": "Canyon ascent",
    "description": "A climber moves through a sunlit canyon above turquoise water.",
    "category": "interactive",
    "sample": null,
    "source": {
      "preview": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands domina.mp4",
      "overlay": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/generated_exit_overlay/At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands domina.mp4"
    },
    "preview": "assets/videos/canyon-front-preview.mp4",
    "previewPoster": "assets/videos/canyon-front-preview.jpg",
    "overlay": "assets/exitmap/canyon-front-overlay.mp4",
    "overlayPoster": "assets/exitmap/canyon-front-overlay.jpg"
  },
  {
    "id": "violin",
    "title": "A rooftop performance",
    "description": "A violinist performs against an illuminated city skyline.",
    "category": "interactive",
    "sample": null,
    "source": {
      "preview": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/At a rooftop garden above Seoul at night, a violinist in a black velvet coat and sparkling hairpin s.mp4",
      "overlay": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/generated_exit_overlay/At a rooftop garden above Seoul at night, a violinist in a black velvet coat and sparkling hairpin s.mp4"
    },
    "preview": "assets/videos/violin-preview.mp4",
    "previewPoster": "assets/videos/violin-preview.jpg",
    "overlay": "assets/exitmap/violin-overlay.mp4",
    "overlayPoster": "assets/exitmap/violin-overlay.jpg"
  },
  {
    "id": "canyon-side",
    "title": "On the rock face",
    "description": "A closer view of the climber, with fine rock textures and moving hands.",
    "category": "interactive",
    "sample": null,
    "source": {
      "preview": "ForcingBooster/logs/0807_causvid14b_patch-router_4steps/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands domina.mp4",
      "overlay": "ForcingBooster/logs/0807_causvid14b_patch-router_4steps/validation/checkpoint_model_001000/generated_exit_overlay/At midsummer in a canyon pool, a climber in an orange technical jacket and chalk dusted hands domina.mp4"
    },
    "preview": "assets/videos/canyon-side-preview.mp4",
    "previewPoster": "assets/videos/canyon-side-preview.jpg",
    "overlay": "assets/exitmap/canyon-side-overlay.mp4",
    "overlayPoster": "assets/exitmap/canyon-side-overlay.jpg"
  },
  {
    "id": "city-night",
    "title": "After the rain",
    "description": "A woman in a silver jacket walks through a rain-soaked city street.",
    "category": "interactive",
    "sample": null,
    "source": {
      "preview": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/validation_patch_router_noema_full4first/At midsummer in a city street after rain, a woman in a reflective silver jacket and black trousers d.mp4",
      "overlay": "Causal-Forcing/logs/0729_dmd_vidprom_adaptive/validation/checkpoint_model_001000/generated_exit_overlay/At midsummer in a city street after rain, a woman in a reflective silver jacket and black trousers d.mp4"
    },
    "preview": "assets/videos/city-night-preview.mp4",
    "previewPoster": "assets/videos/city-night-preview.jpg",
    "overlay": "assets/exitmap/city-night-overlay.mp4",
    "overlayPoster": "assets/exitmap/city-night-overlay.jpg"
  },
  {
    "id": "snowfield",
    "title": "Across a snowfield",
    "description": "",
    "category": "interactive",
    "sample": null,
    "source": {
      "preview": "ForcingBooster/logs/0807_dmd_vidprom_adaptive/checkpoint_eval_train2/outputs/reference_seed10000/Across a snowfield under pink dawn, a skier in a crimson down jacket, fur-lined hood, and mirrored g.mp4",
      "exit_map": "ForcingBooster/logs/0807_dmd_vidprom_adaptive/checkpoint_eval_train2/outputs/reference_seed10000/exit_maps/Across a snowfield under pink dawn, a skier in a crimson down jacket, fur-lined hood, and mirrored g.pt",
      "overlay": "Rendered from the matching exit map; RGB palette at 48% opacity; latent index (frame + 3) // 4."
    },
    "preview": "assets/videos/snowfield-preview.mp4",
    "previewPoster": "assets/videos/snowfield-preview.jpg",
    "overlay": "assets/exitmap/snowfield-overlay.mp4",
    "overlayPoster": "assets/exitmap/snowfield-overlay.jpg"
  },
  {
    "id": "running-dog",
    "title": "A dog running happily",
    "description": "",
    "category": "interactive",
    "sample": null,
    "source": {
      "preview": "ForcingBooster/logs/0917_cf1b3_adaptive0729_vbench50/videos/router_t05/a001.mp4",
      "exit_map": "ForcingBooster/logs/0917_cf1b3_adaptive0729_vbench50/exit_maps/router_t05/a001.npy",
      "overlay": "Rendered from the matching exit map; RGB palette at 48% opacity; latent index (frame + 3) // 4."
    },
    "preview": "assets/videos/running-dog-preview.mp4",
    "previewPoster": "assets/videos/running-dog-preview.jpg",
    "overlay": "assets/exitmap/running-dog-overlay.mp4",
    "overlayPoster": "assets/exitmap/running-dog-overlay.jpg"
  },
  {
    "id": "great-wall",
    "title": "Along the Great Wall",
    "description": "official_04",
    "category": "world",
    "sample": "official_04",
    "source": {
      "preview": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/preview_merged/router/official_04.mp4",
      "overlay": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/preview_merged/overlay/official_04.mp4",
      "camera": {
        "kind": "Recorded WASD; joystick derived from camera trajectory",
        "poseSource": "assets/lingbot-world-v2/examples/04/poses.npy",
        "keySource": "assets/lingbot-world-v2/examples/04/wasd_action.npy"
      }
    },
    "preview": "assets/videos/great-wall-preview.mp4",
    "previewPoster": "assets/videos/great-wall-preview.jpg",
    "overlay": "assets/exitmap/great-wall-overlay.mp4",
    "overlayPoster": "assets/exitmap/great-wall-overlay.jpg"
  },
  {
    "id": "classroom",
    "title": "Through the classroom",
    "description": "dl3dv_00039",
    "category": "world",
    "sample": "dl3dv_00039",
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
      }
    },
    "preview": "assets/videos/classroom-preview.mp4",
    "previewPoster": "assets/videos/classroom-preview.jpg",
    "overlay": "assets/exitmap/classroom-overlay.mp4",
    "overlayPoster": "assets/exitmap/classroom-overlay.jpg"
  },
  {
    "id": "decor-store",
    "title": "Aisles of decorations",
    "description": "dl3dv_00019",
    "category": "world",
    "sample": "dl3dv_00019",
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
      }
    },
    "preview": "assets/videos/decor-store-preview.mp4",
    "previewPoster": "assets/videos/decor-store-preview.jpg",
    "overlay": "assets/exitmap/decor-store-overlay.mp4",
    "overlayPoster": "assets/exitmap/decor-store-overlay.jpg"
  },
  {
    "id": "toy-display",
    "title": "Around the toy display",
    "description": "dl3dv_00079",
    "category": "world",
    "sample": "dl3dv_00079",
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
      }
    },
    "preview": "assets/videos/toy-display-preview.mp4",
    "previewPoster": "assets/videos/toy-display-preview.jpg",
    "overlay": "assets/exitmap/toy-display-overlay.mp4",
    "overlayPoster": "assets/exitmap/toy-display-overlay.jpg"
  },
  {
    "id": "dragon-flight",
    "title": "Through the jungle",
    "description": "official_00",
    "category": "world",
    "sample": "official_00",
    "source": {
      "preview": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/router/official_00.mp4",
      "overlay": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/overlays/official_00.mp4",
      "exit_map": "ForcingBooster/logs/0809_lingbot14B_patch-router/validation/checkpoint_step_001000/router/exit_maps/official_00.pt",
      "camera": {
        "kind": "Recorded WASD; joystick derived from camera trajectory",
        "poseSource": "assets/lingbot-world-v2/examples/00/poses.npy",
        "keySource": "assets/lingbot-world-v2/examples/00/wasd_action.npy"
      }
    },
    "preview": "assets/videos/dragon-flight-preview.mp4",
    "previewPoster": "assets/videos/dragon-flight-preview.jpg",
    "overlay": "assets/exitmap/dragon-flight-overlay.mp4",
    "overlayPoster": "assets/exitmap/dragon-flight-overlay.jpg"
  }
];
