"""Shared helpers for Qwen-Image-2.1 research scripts.

Kept deliberately small: load the official Diffusers pipeline, run one generation
with an explicitly seeded generator, and collect resource stats + metadata.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import threading
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.json"


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- pipeline
VAE_TILE = 1024    # tile_sample_min_{height,width}
VAE_STRIDE = 768   # tile_sample_stride_{height,width}  (256 px overlap blend)


def load_pipeline(model_id: str, revision: str, offload: str = "model", vae_tiling: bool = True):
    """offload: 'none' | 'model' (enable_model_cpu_offload) | 'sequential'.

    vae_tiling: tiled VAE decode with LARGE tiles (1024/768). Needed on 24 GB at 2048x2048: the
    untiled decode alone OOMs (>22 GB). Measured on RTX 3090 at 2048x2048:
      Diffusers default tiles 256/192 -> visible vertical seams, ~1.1 GB peak
      1024/768                        -> matches untiled fp32 reference (mean |diff| 0.38/255, p99 2), ~7.2 GB peak
    Images <= 1024 px per side are not tiled at all with these settings.
    """
    from diffusers import QwenImage21Pipeline

    t0 = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16)
    if vae_tiling:
        pipe.vae.enable_tiling(tile_sample_min_height=VAE_TILE, tile_sample_min_width=VAE_TILE,
                               tile_sample_stride_height=VAE_STRIDE, tile_sample_stride_width=VAE_STRIDE)
    if offload == "none":
        pipe.to("cuda")
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"unknown offload mode {offload!r}")
    pipe.set_progress_bar_config(disable=True)
    return pipe, time.perf_counter() - t0


def recover_after_failure(pipe):
    """After an exception mid-call (e.g. OOM) model-offload hooks can be left inconsistent
    (observed: 'mat2 is on cpu' on the next call). Offload everything and free cached blocks."""
    try:
        pipe.maybe_free_model_hooks()
    except Exception:
        pass
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def generate(pipe, *, prompt, seed, width, height, num_inference_steps,
             negative_prompt=None, true_cfg_scale=1.0, use_kv_cache=True):
    """Run one generation. The seed drives a fresh CUDA generator (as in the official examples)."""
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    kwargs = dict(prompt=prompt, width=width, height=height, num_inference_steps=num_inference_steps,
                  generator=generator, true_cfg_scale=true_cfg_scale, use_kv_cache=use_kv_cache)
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    image = pipe(**kwargs).images[0]
    torch.cuda.synchronize()
    return image, time.perf_counter() - t0


# --------------------------------------------------------------------------- monitoring
class ResourceMonitor:
    """Samples GPU util / used memory (NVML, whole device) and process RSS in a thread."""

    def __init__(self, interval: float = 0.5):
        import psutil
        import pynvml

        self._psutil, self._nvml = psutil, pynvml
        pynvml.nvmlInit()
        # CUDA_VISIBLE_DEVICES maps torch device 0 -> physical id; NVML uses physical ids.
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        self.physical_gpu = int(vis) if vis.isdigit() else 0
        self._h = pynvml.nvmlDeviceGetHandleByIndex(self.physical_gpu)
        self._proc = psutil.Process()
        self.interval = interval
        self.reset()

    def reset(self):
        self.util, self.gpu_used_mb, self.rss_mb = [], [], []

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.util.append(self._nvml.nvmlDeviceGetUtilizationRates(self._h).gpu)
                self.gpu_used_mb.append(self._nvml.nvmlDeviceGetMemoryInfo(self._h).used / 2**20)
                self.rss_mb.append(self._proc.memory_info().rss / 2**20)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self.reset()
        torch.cuda.reset_peak_memory_stats()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        u = self.util or [0]
        return {
            "physical_gpu_id": self.physical_gpu,
            "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "torch_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "nvml_peak_device_used_gb": round(max(self.gpu_used_mb or [0]) / 1024, 2),
            "process_peak_rss_gb": round(max(self.rss_mb or [0]) / 1024, 2),
            "gpu_util_mean_pct": round(sum(u) / len(u), 1),
            "gpu_util_max_pct": max(u),
            "gpu_util_samples": len(self.util),
        }


# --------------------------------------------------------------------------- metadata
def env_versions() -> dict:
    import accelerate
    import diffusers
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
    }


def build_metadata(cfg: dict, *, prompt, seed, width, height, steps, offload, gen_time,
                   image, extra: dict | None = None, negative_prompt=None, true_cfg_scale=1.0,
                   use_kv_cache=True) -> dict:
    md = {
        "model": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "pipeline": "diffusers.QwenImage21Pipeline",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": int(seed),
        "generator_device": "cuda",
        "width": width,
        "height": height,
        "output_size": list(image.size),
        "output_mode": image.mode,
        "num_inference_steps": steps,
        "true_cfg_scale": true_cfg_scale,
        "use_kv_cache": use_kv_cache,
        "vae_tiling": (f"tile{VAE_TILE}_stride{VAE_STRIDE}" if cfg.get("vae_tiling", True) else False),
        "dtype": "bfloat16",
        "offload": offload,
        "device": f"cuda (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})",
        "generation_time_seconds": round(gen_time, 2),
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "versions": env_versions(),
    }
    if extra:
        md.update(extra)
    return md


def save_image_and_metadata(image, metadata: dict, out_path: str | os.PathLike):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.png")
    image.save(tmp)  # PNG keeps RGBA if the model produced transparency
    os.replace(tmp, out_path)
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
