#!/usr/bin/env python
"""Batch generation for Qwen-Image-2.1 from a JSON/JSONL task list.

Each record: {"id": "sample_000001", "prompt": "...", "seed": 12345}
  * id and prompt are required; id must be unique (used as the file name)
  * seed is optional; if missing it is derived from the id (sha256), so it is stable across runs
Optional per-record overrides: width, height, num_inference_steps, negative_prompt, true_cfg_scale.

Behaviour:
  * model loaded once, samples generated one by one
  * <out_dir>/<id>.png + <id>.json per sample; written atomically (PNG via tmp + rename, JSON last)
  * a sample counts as done iff both <id>.png and <id>.json exist -> re-running the same command resumes
  * one sample failing is logged to <out_dir>/failures.jsonl and the run continues
  * --shard K/N processes records where index % N == K (one independent worker per GPU)

Example:
  python scripts/generate_batch.py --input examples/prompts.json --output outputs/batch_demo --gpu 0
"""
import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path


def seed_from_id(sample_id: str) -> int:
    return int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest(), 16) % (2**31 - 1)


def read_tasks(path):
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        recs = json.loads(text)
    else:
        recs = [json.loads(l) for l in text.splitlines() if l.strip()]
    ids = [r["id"] for r in recs]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"duplicate ids in task file: {sorted(dup)[:10]}")
    for r in recs:
        if "id" not in r or "prompt" not in r:
            raise SystemExit(f"record {r} missing id/prompt")
        if "seed" not in r:
            r["seed"] = seed_from_id(r["id"])
            r["seed_source"] = "sha256(id)"
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "--tasks", dest="tasks", required=True,
                    help=".json (list of records) or .jsonl (one record per line)")
    ap.add_argument("--output", "--out-dir", dest="out_dir", required=True, help="output directory")
    ap.add_argument("--gpu", default=None, help="GPU index to use (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--offload", choices=["none", "model", "sequential"], default=None)
    ap.add_argument("--shard", default="0/1", help="K/N: take records with index %% N == K "
                                                   "(run one process per GPU for multi-GPU)")
    ap.add_argument("--max-failures", type=int, default=0, help="abort after this many failures (0 = never)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)   # must happen before torch is imported
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import qi21_common as C

    cfg = C.load_config(a.config)
    k, n = map(int, a.shard.split("/"))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out / f"run_{tag}_shard{k}of{n}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()])
    log = logging.getLogger("batch")

    tasks = [r for i, r in enumerate(read_tasks(a.tasks)) if i % n == k]
    todo = [r for r in tasks if not ((out / f"{r['id']}.png").exists() and (out / f"{r['id']}.json").exists())]
    log.info("tasks=%s shard=%s/%s total_in_shard=%d already_done=%d todo=%d",
             a.tasks, k, n, len(tasks), len(tasks) - len(todo), len(todo))
    if not todo:
        return

    offload = a.offload or cfg["offload"]
    pipe, load_s = C.load_pipeline(cfg["model_id"], cfg["model_revision"], offload,
                                        vae_tiling=cfg["vae_tiling"])
    log.info("pipeline loaded in %.1fs (offload=%s)", load_s, offload)
    mon = C.ResourceMonitor()

    ok = fail = 0
    for j, r in enumerate(todo, 1):
        width = r.get("width") or a.width or cfg["width"]
        height = r.get("height") or a.height or cfg["height"]
        steps = r.get("num_inference_steps") or a.steps or cfg["num_inference_steps"]
        cfg_scale = r.get("true_cfg_scale", cfg["true_cfg_scale"])
        neg = r.get("negative_prompt")
        try:
            with mon:
                image, gen_s = C.generate(pipe, prompt=r["prompt"], seed=r["seed"], width=width, height=height,
                                          num_inference_steps=steps, negative_prompt=neg,
                                          true_cfg_scale=cfg_scale, use_kv_cache=cfg["use_kv_cache"])
            extra = {"id": r["id"], "task_file": str(a.tasks), "shard": a.shard,
                     "resources_generate": mon.summary()}
            extra.update({f"task_{kk}": vv for kk, vv in r.items() if kk not in ("id", "prompt", "seed")})
            md = C.build_metadata(cfg, prompt=r["prompt"], seed=r["seed"], width=width, height=height,
                                  steps=steps, offload=offload, gen_time=gen_s, image=image,
                                  negative_prompt=neg, true_cfg_scale=cfg_scale,
                                  use_kv_cache=cfg["use_kv_cache"], extra=extra)
            C.save_image_and_metadata(image, md, out / f"{r['id']}.png")
            ok += 1
            log.info("[%d/%d] OK %s %dx%d %.1fs", j, len(todo), r["id"], width, height, gen_s)
        except Exception as e:  # keep going; record why
            fail += 1
            import torch
            oom = isinstance(e, torch.cuda.OutOfMemoryError)
            C.recover_after_failure(pipe)
            with open(out / "failures.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": r["id"], "time": dt.datetime.now().isoformat(timespec="seconds"),
                                    "error_type": type(e).__name__, "oom": oom, "error": str(e)[:2000],
                                    "traceback": traceback.format_exc()[-4000:]}, ensure_ascii=False) + "\n")
            log.error("[%d/%d] FAIL %s %s: %s", j, len(todo), r["id"], type(e).__name__, str(e)[:300])
            if a.max_failures and fail >= a.max_failures:
                log.error("max failures reached, aborting")
                break
    log.info("done ok=%d fail=%d log=%s", ok, fail, log_path)


if __name__ == "__main__":
    main()
