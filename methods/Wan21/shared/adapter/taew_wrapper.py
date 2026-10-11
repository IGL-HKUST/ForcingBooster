from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from methods.Wan21.shared.adapter.demo_utils.taehv import StreamingTAEHV, TAEHV


@dataclass
class StreamingDecodeChunk:
    """A decoded pixel chunk produced on the dedicated VAE stream."""

    pixels: torch.Tensor
    start_event: torch.cuda.Event
    done_event: torch.cuda.Event
    latent_frames: int
    pixel_frames: int


class TAEWStreamingDecodeSession:
    """Stateful TAEW2.1 decoder running on a dedicated CUDA stream."""

    def __init__(
        self,
        wrapper: "TAEW21VAEWrapper",
        device: torch.device | str | int,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        self.wrapper = wrapper
        if isinstance(device, int):
            self.device = torch.device("cuda", device)
        else:
            self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type != "cuda":
            raise ValueError("Streaming TAEW2.1 decode requires a CUDA device")

        model_device = wrapper.device
        if model_device.type != "cuda" or model_device != self.device:
            raise ValueError(
                f"TAEW2.1 model must be on {self.device} before streaming decode; "
                f"got {model_device}"
            )

        self.stream = stream if stream is not None else torch.cuda.Stream(device=self.device)
        self.decoder = StreamingTAEHV(wrapper.model)
        self._chunks: list[StreamingDecodeChunk] = []
        self._submitted_latents = 0
        self._closed = False
        self._decode_start_event: Optional[torch.cuda.Event] = None
        self._decode_end_event: Optional[torch.cuda.Event] = None
        self._finish_start_event: Optional[torch.cuda.Event] = None

    @property
    def chunks(self) -> tuple[StreamingDecodeChunk, ...]:
        return tuple(self._chunks)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def submitted_latents(self) -> int:
        return self._submitted_latents

    @property
    def expected_pixel_frames(self) -> int:
        return self.wrapper.pixel_frames_for_latents(self._submitted_latents)

    @property
    def decode_start_event(self) -> Optional[torch.cuda.Event]:
        return self._decode_start_event

    @property
    def decode_end_event(self) -> Optional[torch.cuda.Event]:
        return self._decode_end_event

    def submit(self, latent_chunk: torch.Tensor) -> StreamingDecodeChunk:
        if self._closed:
            raise RuntimeError("Cannot submit to a finished TAEW2.1 streaming session")
        if latent_chunk.ndim != 5:
            raise ValueError(f"Expected NTCHW latent chunk, got shape={tuple(latent_chunk.shape)}")
        if latent_chunk.device != self.device:
            raise ValueError(f"Expected latent chunk on {self.device}, got {latent_chunk.device}")
        if latent_chunk.shape[1] == 0:
            raise ValueError("Cannot submit an empty latent chunk")
        if latent_chunk.shape[2] != self.wrapper.model.latent_channels:
            raise ValueError(
                f"Expected {self.wrapper.model.latent_channels} latent channels, "
                f"got {latent_chunk.shape[2]}"
            )

        producer_stream = torch.cuda.current_stream(self.device)
        ready_event = torch.cuda.Event()
        ready_event.record(producer_stream)
        previous_total = self.expected_pixel_frames
        latent_frames = int(latent_chunk.shape[1])

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.stream.wait_event(ready_event)
            latent_chunk.record_stream(self.stream)
            chunk_start_event = torch.cuda.Event(enable_timing=True)
            chunk_start_event.record(self.stream)
            if self._decode_start_event is None:
                self._decode_start_event = chunk_start_event

            latents = latent_chunk.to(dtype=self.wrapper.dtype)
            frames = []
            for latent_frame in latents.split(1, dim=1):
                frame = self.decoder.decode(latent_frame)
                while frame is not None:
                    frames.append(frame)
                    frame = self.decoder.decode()

            if not frames:
                raise RuntimeError("TAEW2.1 streaming decoder produced no frames")
            pixels = torch.cat(frames, dim=1).float().mul_(2.0).sub_(1.0)
            self._submitted_latents += latent_frames
            expected_chunk_frames = self.expected_pixel_frames - previous_total
            if pixels.shape[1] != expected_chunk_frames:
                raise RuntimeError(
                    "TAEW2.1 streaming frame-count mismatch: "
                    f"expected {expected_chunk_frames}, got {pixels.shape[1]}"
                )

            done_event = torch.cuda.Event(enable_timing=True)
            done_event.record(self.stream)

        chunk = StreamingDecodeChunk(
            pixels=pixels,
            start_event=chunk_start_event,
            done_event=done_event,
            latent_frames=latent_frames,
            pixel_frames=int(pixels.shape[1]),
        )
        self._chunks.append(chunk)
        self._decode_end_event = done_event
        return chunk

    def collect(
        self,
        chunk: StreamingDecodeChunk,
        consumer_stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        if consumer_stream is None:
            consumer_stream = torch.cuda.current_stream(self.device)
        consumer_stream.wait_event(chunk.done_event)
        chunk.pixels.record_stream(consumer_stream)
        return chunk.pixels

    def release(self, chunk: StreamingDecodeChunk) -> None:
        """Drop a chunk reference after an incremental consumer has copied it out."""
        self._chunks.remove(chunk)

    def finish(self, expected_frames: Optional[int] = None) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("TAEW2.1 streaming session has already been finished")
        if not self._chunks:
            raise RuntimeError("Cannot finish a TAEW2.1 session with no submitted latents")

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self._finish_start_event = torch.cuda.Event(enable_timing=True)
            self._finish_start_event.record(self.stream)
            flushed = self.decoder.flush_decoder()
            if flushed:
                raise RuntimeError(
                    "TAEW2.1 decoder retained unexpected frames after each chunk was drained"
                )
            video = torch.cat([chunk.pixels for chunk in self._chunks], dim=1)
            done_event = torch.cuda.Event(enable_timing=True)
            done_event.record(self.stream)

        self._decode_end_event = done_event
        self._closed = True
        consumer_stream = torch.cuda.current_stream(self.device)
        consumer_stream.wait_event(done_event)
        video.record_stream(consumer_stream)

        expected_frames = self.expected_pixel_frames if expected_frames is None else expected_frames
        if video.shape[1] != expected_frames:
            raise RuntimeError(
                f"Expected {expected_frames} decoded frames, got {video.shape[1]}"
            )
        return video

    def synchronize(self) -> None:
        if self._decode_end_event is not None:
            self._decode_end_event.synchronize()

    def elapsed_ms(self) -> float:
        if not self._chunks:
            return 0.0
        self.synchronize()
        elapsed = float(sum(
            chunk.start_event.elapsed_time(chunk.done_event)
            for chunk in self._chunks
        ))
        if self._finish_start_event is not None and self._decode_end_event is not None:
            elapsed += float(self._finish_start_event.elapsed_time(self._decode_end_event))
        return elapsed

    def chunk_elapsed_ms(self) -> list[float]:
        if not self._chunks:
            return []
        self.synchronize()
        return [
            float(chunk.start_event.elapsed_time(chunk.done_event))
            for chunk in self._chunks
        ]

    def span_ms(self) -> float:
        if self._decode_start_event is None or self._decode_end_event is None:
            return 0.0
        self.synchronize()
        return float(self._decode_start_event.elapsed_time(self._decode_end_event))

    def overlap_with(
        self,
        producer_start: torch.cuda.Event,
        producer_end: torch.cuda.Event,
    ) -> float:
        """Sum decoder-active intervals that intersect the producer interval."""
        if self._decode_start_event is None or self._decode_end_event is None:
            return 0.0
        self.synchronize()
        producer_end.synchronize()
        producer_duration = float(producer_start.elapsed_time(producer_end))
        overlap = 0.0
        for chunk in self._chunks:
            decoder_start = float(producer_start.elapsed_time(chunk.start_event))
            decoder_end = float(producer_start.elapsed_time(chunk.done_event))
            overlap += max(
                0.0,
                min(producer_duration, decoder_end) - max(0.0, decoder_start),
            )
        return overlap

    def abort(self) -> None:
        """Drain outstanding work and discard temporal state after cancellation/error."""
        self.stream.synchronize()
        self.decoder.reset()
        self._chunks.clear()
        self._closed = True


class TAEW21VAEWrapper(torch.nn.Module):
    """Adapter between repository VAE conventions and the official TAEW2.1 model."""

    supports_streaming_decode = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        streaming_decode: bool = False,
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"TAEW2.1 checkpoint not found: {checkpoint_path}")

        self.checkpoint_path = str(checkpoint_path)
        self.streaming_decode_enabled = bool(streaming_decode)
        self.model = TAEHV(checkpoint_path=self.checkpoint_path)
        self.model.eval().requires_grad_(False)
        self._active_session: Optional[TAEWStreamingDecodeSession] = None
        self._decode_stream: Optional[torch.cuda.Stream] = None
        self._decode_stream_device: Optional[torch.device] = None

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def pixel_frames_for_latents(self, latent_frames: int) -> int:
        if latent_frames == 0:
            return 0
        return latent_frames * self.model.t_upscale - self.model.frames_to_trim

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        if pixel.ndim != 5 or pixel.shape[1] != 3:
            raise ValueError(f"Expected NCTHW RGB pixels, got shape={tuple(pixel.shape)}")
        video = pixel.permute(0, 2, 1, 3, 4).to(device=self.device, dtype=self.dtype)
        video = video.mul(0.5).add(0.5).contiguous()
        return self.model.encode_video(
            video,
            parallel=True,
            show_progress_bar=False,
        ).float()

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if use_cache:
            raise NotImplementedError(
                "TAEW2.1 uses StreamingTAEHV instead of WanVAE cached_decode"
            )
        if latent.ndim != 5:
            raise ValueError(f"Expected NTCHW latents, got shape={tuple(latent.shape)}")
        latent = latent.to(device=self.device, dtype=self.dtype)
        video = self.model.decode_video(
            latent,
            parallel=False,
            show_progress_bar=False,
        )
        return video.float().mul_(2.0).sub_(1.0)

    def create_streaming_session(
        self,
        device: Optional[torch.device | str | int] = None,
    ) -> TAEWStreamingDecodeSession:
        if self._active_session is not None and not self._active_session.closed:
            self._active_session.abort()
        target_device = self.device if device is None else device
        if isinstance(target_device, int):
            target_device = torch.device("cuda", target_device)
        else:
            target_device = torch.device(target_device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        if target_device.type != "cuda":
            raise ValueError("Streaming TAEW2.1 decode requires a CUDA device")
        if self._decode_stream is None or self._decode_stream_device != target_device:
            self._decode_stream = torch.cuda.Stream(device=target_device)
            self._decode_stream_device = target_device
        self._active_session = TAEWStreamingDecodeSession(
            self,
            target_device,
            stream=self._decode_stream,
        )
        return self._active_session

    def clear_cache(self) -> None:
        if self._active_session is not None and not self._active_session.closed:
            self._active_session.abort()
        self._active_session = None
