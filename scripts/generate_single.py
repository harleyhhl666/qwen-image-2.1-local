#!/usr/bin/env python
"""Generate one image with Qwen-Image-2.1 (official Diffusers pipeline, BF16).

Example:
  python scripts/generate_single.py --prompt "A neon shop sign that reads \"QWEN\"" \
      --seed 42 --output outputs/test.png --gpu 0
Writes <output>.png and <output>.json (metadata).
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True, help="output .png path; metadata goes to same name .json")
    ap.add_argument("--width", type=int, default=None, help="default from config (2048)")
    ap.add_argument("--height", type=int, default=None, help="default from config (2048)")
    ap.add_argument("--steps", type=int, default=None, help="num_inference_steps, default 40")
    ap.add_argument("--gpu", default=None, help="GPU index to use (sets CUDA_VISIBLE_DEVICES); "
                                                "default: keep the current CUDA_VISIBLE_DEVICES")
    ap.add_argument("--negative-prompt", default=None, help="only used when --cfg > 1")
    ap.add_argument("--cfg", type=float, default=None, help="true_cfg_scale; official default 1.0 (no CFG)")
    ap.add_argument("--offload", choices=["none", "model", "sequential"], default=None)
    ap.add_argument("--no-kv-cache", action="store_true", help="disable prefix KV cache (changes samples)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)   # must happen before torch is imported

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import qi21_common as C

    cfg = C.load_config(a.config)
    width = a.width or cfg["width"]
    height = a.height or cfg["height"]
    steps = a.steps or cfg["num_inference_steps"]
    cfg_scale = a.cfg if a.cfg is not None else cfg["true_cfg_scale"]
    offload = a.offload or cfg["offload"]
    use_kv = cfg["use_kv_cache"] and not a.no_kv_cache
    mon = C.ResourceMonitor()
    with mon:
        pipe, load_s = C.load_pipeline(cfg["model_id"], cfg["model_revision"], offload,
                                        vae_tiling=cfg["vae_tiling"])
    load_stats = mon.summary()
    with mon:
        image, gen_s = C.generate(pipe, prompt=a.prompt, seed=a.seed, width=width, height=height,
                                  num_inference_steps=steps, negative_prompt=a.negative_prompt,
                                  true_cfg_scale=cfg_scale, use_kv_cache=use_kv)
    gen_stats = mon.summary()

    md = C.build_metadata(cfg, prompt=a.prompt, seed=a.seed, width=width, height=height, steps=steps,
                          offload=offload, gen_time=gen_s, image=image, negative_prompt=a.negative_prompt,
                          true_cfg_scale=cfg_scale, use_kv_cache=use_kv,
                          extra={"model_load_time_seconds": round(load_s, 2),
                                 "resources_load": load_stats, "resources_generate": gen_stats})
    C.save_image_and_metadata(image, md, a.output)
    print(f"saved {a.output}  gen={gen_s:.1f}s load={load_s:.1f}s "
          f"peak_alloc={gen_stats['torch_peak_allocated_gb']}GB rss={gen_stats['process_peak_rss_gb']}GB")


if __name__ == "__main__":
    main()
