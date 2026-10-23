"""Write decoded H3 latents to MP4 with a storyboard sheet."""

from __future__ import annotations


def uint8_frames(pixels, *, imagenet):
    """N,C,T,H,W decoded pixels to uint8 RGB on their own device. The official decoder returns
    ImageNet-normalized pixels, TAEHV returns RGB in [0, 1]."""
    import torch

    frame = pixels.float()
    if imagenet:
        mean = torch.tensor((0.485, 0.456, 0.406), device=frame.device).view(1, 3, 1, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=frame.device).view(1, 3, 1, 1, 1)
        frame = frame * std + mean
    frame = frame.clamp(0, 1)
    if not bool(torch.isfinite(frame).all()):
        raise FloatingPointError("Non-finite decoded frame")
    return frame.mul(255).round().byte()


def decode_write(vae, generated, geometry, output, fps, device, *, streaming=False, progress=None):
    """Use official temporal overlap blending, or TAEHV taeh3; frames become uint8 on the GPU."""
    import imageio.v2 as imageio
    import torch

    if getattr(vae, "arch_name", None) == "taeh3":
        # TAEHV decodes model-space (normalized) latents straight to RGB in [0, 1].
        z = generated[:, :, : geometry.decode_latents].to(device, torch.float16).transpose(1, 2)
        with torch.no_grad():
            pixels = vae.decode_video(z, parallel=True, show_progress_bar=False)
        chunks = (uint8_frames(pixels.transpose(1, 2), imagenet=False).cpu(),)
    elif streaming:
        from .stream_decode import decoded_chunks

        chunks = decoded_chunks(vae, generated[:, :, : geometry.decode_latents], device=device,
                                convert=lambda x: uint8_frames(x, imagenet=True))
    else:
        z = generated[:, :, : geometry.decode_latents].float()
        mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            pixels = vae.decode(z * std + mean, return_dict=False)[0]
        if pixels.shape[2] != geometry.decoded_frames:
            raise ValueError("VAE output frame count differs")
        chunks = (uint8_frames(pixels, imagenet=True).cpu(),)

    def frames():
        count = 0
        for chunk in chunks:
            for frame in chunk[0].unbind(1):
                if count < geometry.source_frames:
                    yield count, frame
                count += 1
            if progress is not None:
                progress(count)
        if count != geometry.decoded_frames:
            raise ValueError(f"VAE returned {count} frames, expected {geometry.decoded_frames}")

    part = output.with_name("." + output.stem + ".tmp.mp4")
    story_indices = sorted(
        set(round((geometry.source_frames - 1) * p) for p in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1))
    )
    story = []
    luminance = []
    with imageio.get_writer(
        str(part),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-threads", "0", "-pix_fmt", "yuv420p"],  # 0: libx264 picks the thread count
    ) as writer:
        for i, frame in frames():
            array = frame.permute(1, 2, 0).numpy()
            writer.append_data(array)
            if i in story_indices:
                from PIL import Image

                story.append(Image.fromarray(array).resize((448, 256)))
                luminance.append(float(array.mean()))
    part.replace(output)
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (448 * len(story), 280), "#202020")
    draw = ImageDraw.Draw(sheet)
    for j, (im, index) in enumerate(zip(story, story_indices, strict=True)):
        sheet.paste(im, (448 * j, 24))
        draw.text((448 * j + 6, 5), f"{index / fps:.2f}s / frame {index}", fill="white")
    sheet.save(output.with_suffix(".storyboard.jpg"), quality=90)
    return dict(sampled_frames=story_indices, mean_pixel_value=luminance)
