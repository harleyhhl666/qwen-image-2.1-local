# Qwen-Image-2.1 本地部署与生成

这是一个轻量项目，用于在 **Linux + NVIDIA GPU 服务器** 上部署和使用 [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)（官方 BF16 权重 + 官方 Diffusers `QwenImage21Pipeline`，不使用量化）。

支持：

- 单图生成
- 批量生成（JSON / JSONL 任务文件；可断点续跑；单张失败不影响其他；失败原因写入日志）
- 自定义 prompt、seed、分辨率、steps、GPU
- 每张图自动保存同名 `.json` metadata（参数、版本、耗时、显存）
- 同一环境、同一 seed、同一参数下可逐像素复现

> 模型许可：Qwen-Image-2.1 使用 **Qwen Research License**，**仅限非商业研究用途**。详见模型页的 LICENSE。

---

## 1. 推荐环境（已实际验证）

| 项目 | 已验证版本 |
|---|---|
| 系统 | Ubuntu 20.04（glibc 2.31） |
| GPU | NVIDIA RTX 3090 24GB |
| NVIDIA Driver | 535.183.01（`nvidia-smi` 显示 CUDA 12.2） |
| Python | 3.11 |
| PyTorch | 2.14.0 + cu126（pip wheel 自带 CUDA 12.6 运行库，不需要安装系统 CUDA） |
| Diffusers | 0.41.0.dev0，GitHub commit `e0118ade2f60234c41bacf40330a7e2f61108849` |
| Transformers / Accelerate | 5.17.0 / 1.15.0 |

- 目前只在 **RTX 3090 24GB** 上实际测试过。其他 GPU 没有测试，不保证能运行。
- cu126 的 PyTorch 在 535 驱动（CUDA 12.2）上能正常运行，已实测。
- **Diffusers 必须使用上面固定的 GitHub commit**：PyPI 上的正式版（截至 0.40.0）还没有 `QwenImage21Pipeline`。

## 2. Clone

```bash
git clone https://github.com/harleyhhl666/qwen-image-2.1-local.git
cd qwen-image-2.1-local
```

## 3. 创建环境

```bash
conda create -n qwen_image_21 python=3.11 -y
conda activate qwen_image_21
pip install -r requirements.txt
```

`requirements.txt` 已固定所有关键版本：

- PyTorch 从官方 cu126 源安装；
- Diffusers 从固定 commit 的源码包安装，这种方式不需要 `git clone` GitHub。

也可以一步完成，效果等价：`conda env create -f environment.yml && conda activate qwen_image_21`。

检查安装是否成功：

```bash
python -c "import torch, diffusers; from diffusers import QwenImage21Pipeline; print(torch.__version__, torch.cuda.is_available(), diffusers.__version__)"
# 期望输出: 2.14.0+cu126 True 0.41.0.dev0
```

## 4. 模型下载

模型来自官方 Hugging Face 仓库 **`Qwen/Qwen-Image-2.1`**。代码固定在已验证的 revision `790c92633540aa0cb11d9abf19eb46d861714758`，见 `configs/default.json`。

- **第一次运行生成脚本时会自动下载**，不需要手动操作。
- 模型文件约 **33 GB**。请确认缓存所在磁盘有足够空间：默认缓存是 `~/.cache/huggingface`。
- 如果 home 目录空间不够，先用环境变量把缓存放到大盘上：

  ```bash
  export HF_HOME=/path/to/big/disk/hf_home
  ```

- 如果想提前下载（这样可以放进 tmux 里慢慢下）：

  ```bash
  hf download Qwen/Qwen-Image-2.1 --revision 790c92633540aa0cb11d9abf19eb46d861714758 --exclude "assets/*"
  ```

- 下载完成后，可以设置 `export HF_HUB_OFFLINE=1`，之后的运行不再联网检查更新。

## 5. 单图生成

```bash
python scripts/generate_single.py \
    --prompt "一张简洁的中文海报，纯深蓝色背景，画面中央只有两行白色文字：大字标题“未来影像大会”，下方小字“2026年11月8日 上海”，没有其他文字，极简留白" \
    --seed 42 \
    --gpu 0 \
    --output outputs/test.png
```

常用参数：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--prompt` | 提示词（中英文都可以） | 必填 |
| `--output` | 输出 png 路径，同名 `.json` 保存 metadata | 必填 |
| `--seed` | 随机种子 | 42 |
| `--width` / `--height` | 分辨率 | 2048 × 2048 |
| `--steps` | 推理步数 | 40（官方推荐） |
| `--gpu` | 使用第几张 GPU（等同于设置 `CUDA_VISIBLE_DEVICES`） | 不设置则沿用当前环境变量 |
| `--cfg` / `--negative-prompt` | CFG。官方默认不开启（1.0）；大于 1 时会使每步计算量翻倍 | 1.0 |

**官方推荐的原生 2K 尺寸**（宽 × 高），都已实测可用：

| 比例 | 尺寸 |
|---|---|
| 1:1 | 2048 × 2048 |
| 4:3 / 3:4 | 2400 × 1792 / 1792 × 2400 |
| 3:2 / 2:3 | 2528 × 1696 / 1696 × 2528 |
| 16:9 / 9:16 | 2752 × 1536 / 1536 × 2752 |

- 宽和高必须是 32 的倍数，否则会被自动向下取整。
- 1024 × 1024 也可以用，速度快约 3 倍。

## 6. 批量生成

输入文件是 JSON 列表，或 JSONL（每行一条）。示例见 `examples/prompts.json`：

```json
[
  {"id": "example_001", "prompt": "...", "seed": 1001, "width": 1024, "height": 1024},
  {"id": "example_002", "prompt": "..."}
]
```

- `id`、`prompt` 必填；`id` 同时用作输出文件名，必须唯一。
- `seed` 可选：不写时由 id 计算得到（sha256），每次运行都一样。
- `width` / `height` / `num_inference_steps` 可以逐条覆盖，不写就用默认值。

运行：

```bash
python scripts/generate_batch.py --input examples/prompts.json --output outputs/batch_demo --gpu 0
```

- **断点续跑**：中断后重新执行同一条命令即可。已同时存在 `<id>.png` 和 `<id>.json` 的样本会被跳过。
- **失败处理**：单张失败（包括 OOM）会写入 `outputs/.../failures.jsonl`，然后继续下一张。
- **多卡**：每张卡跑一个独立进程，用 `--shard k/N` 分配任务，建议放在 tmux 里：

  ```bash
  python scripts/generate_batch.py --input my.json --output outputs/run1 --gpu 2 --shard 0/2
  python scripts/generate_batch.py --input my.json --output outputs/run1 --gpu 3 --shard 1/2
  ```

## 7. 显存说明（RTX 3090 24GB 实测）

- 默认启用 **model CPU offload**（官方推荐的显存优化方式）。原因是模型 BF16 权重约 31 GB（文本编码器约 17.5 GB，DiT 约 14.2 GB），无法同时放进 24 GB 显存。
- 默认启用 **VAE 分块解码**（tile 1024 / stride 768）。2K 分辨率下如果一次性解码，单是 VAE 这一步就需要超过 22 GB 显存，会 OOM。当前的分块参数已经对比过：结果与不分块解码基本一致，没有接缝。
  - 不要改用 Diffusers 默认的 256 小分块，会出现可见的竖向接缝。
- 单进程实测数据（40 步，BF16）：

| 分辨率 | GPU 峰值显存 | 进程内存 | 每张耗时 |
|---|---:|---:|---:|
| 1024 × 1024 | 约 16.4 GB | 约 32 GB | 约 82 秒 |
| 2048 × 2048 等官方 2K 尺寸 | 约 16.4 GB | 约 33 GB | 约 270–280 秒 |

- 多卡时**每个进程都要占约 33 GB 系统内存**。开多个进程前，请先确认服务器内存够用。
  - 已实测 5 张 3090 各跑一个进程互不拖慢；
  - 服务器是多人共享的，请按剩余内存决定开几个进程。

## 8. 输出文件

```text
outputs/
├── test.png                     # 生成的图片（RGBA PNG；普通 prompt 的 alpha 基本都是 255）
├── test.json                    # metadata
└── batch_demo/
    ├── example_001_street.png
    ├── example_001_street.json
    ├── failures.jsonl           # 只有出现失败时才会生成
    └── run_<时间>_shard0of1.log # 批量运行日志
```

`.json` metadata 包含以下内容，可用于复现和实验记录：

- 模型与 revision；
- prompt、seed、宽高、steps、CFG；
- dtype、offload、VAE 分块参数；
- GPU 编号；
- 生成耗时、峰值显存 / 内存、GPU 利用率；
- 时间戳，以及 torch / diffusers / transformers 版本。

## 9. 常见问题

- **CUDA out of memory**：
  - 先用 `nvidia-smi` 确认这张卡没有被别人占用；
  - 不要关闭 offload 或 VAE 分块；
  - 仍然 OOM 的话，可以试 `--offload sequential`（更省显存，但明显更慢）。
- **GPU 被其他用户占用**：运行前先看 `nvidia-smi`，用 `--gpu` 选一张空闲卡。不要默认用 0 号卡。
- **第一次运行很慢 / 卡在下载**：正在下载约 33 GB 模型，可以先按第 4 节在 tmux 里提前下载。
- **huggingface.co 连不上**：如果所在网络无法直接访问 Hugging Face，可以设置 `export HF_ENDPOINT=<你所在环境可用的 HF 镜像地址>` 后再下载。其余代码不需要改动。
- **速度较慢**：这是 24 GB 显卡开 CPU offload 的正常代价，每一步都要在 CPU 和 GPU 之间搬运权重。批量生成时，靠多卡并行提高吞吐。
- **同一 seed 结果和别人不一样**：请检查 metadata 里的版本号、`use_kv_cache`、`vae_tiling`、分辨率是否一致。这些参数任何一项不同，图都会不同。

## 目录结构

```text
qwen-image-2.1-local/
├── README.md
├── requirements.txt          # 固定版本依赖（含 cu126 PyTorch 源、固定 commit 的 Diffusers）
├── environment.yml           # conda 一步建环境（调用 requirements.txt）
├── configs/default.json      # 模型 id + revision + 默认生成参数
├── scripts/
│   ├── generate_single.py    # 单图生成
│   ├── generate_batch.py     # 批量生成
│   └── qi21_common.py        # 加载 pipeline、带 seed 的生成、metadata、资源记录
├── examples/prompts.json     # 批量输入格式示例（3 条）
└── outputs/.gitkeep
```
