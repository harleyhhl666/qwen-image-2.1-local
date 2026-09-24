# 在 RTX PRO 6000（Blackwell，96GB）上部署的注意事项

本仓库目前只在 **RTX 3090 24GB** 上实际验证过，现有的 `requirements.txt` 和 `configs/default.json` 都是按 3090 的条件定的。
这份文档列出换到 **RTX PRO 6000 Blackwell** 时需要改或可能需要改的地方。

> **说明**
> - 编写者没有 Pro 6000 的使用权限。下面每一条都标注了依据：
>   - **已确认**：在 3090 服务器上实际查过，或直接来自仓库代码；
>   - **推断**：根据硬件规格推算，未在 Pro 6000 上实测。
> - 本文档**没有修改**仓库中的任何代码、依赖或配置文件。下文提到的改动需要使用者在自己的副本里做，建议先实测再决定是否合入仓库。

---

## 一、必须修改（不改无法运行）

### 1. PyTorch 版本不支持 Blackwell 架构 —— 已确认

- RTX PRO 6000 Blackwell 的计算能力是 **sm_120**。
- 仓库固定的 `torch==2.14.0+cu126` 支持的架构是：

  ```text
  sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90
  ```

  其中**没有 sm_120**。这是在已验证环境中用 `torch.cuda.get_arch_list()` 实测得到的。
- 按现有 `requirements.txt` 直接安装，第一次调用 GPU 就会报错：

  ```text
  RuntimeError: CUDA error: no kernel image is available for execution on the device
  ```

- PyTorch 官方源中 `torch 2.14.0` 有 `+cu126`、`+cu130`、`+cu132` 三种构建，`torchvision 0.29.0` 也有对应的版本（已查询 download.pytorch.org）。

**修改方法**：把 `requirements.txt` 中的 cu126 全部换成 cu130，PyTorch 版本号不变：

```diff
-# Install torch FIRST from the PyTorch cu126 index (see README), then:  pip install -r requirements.txt
---extra-index-url https://download.pytorch.org/whl/cu126
-torch==2.14.0+cu126
-torchvision==0.29.0+cu126
+--extra-index-url https://download.pytorch.org/whl/cu130
+torch==2.14.0+cu130
+torchvision==0.29.0+cu130
```

其余依赖（Diffusers 固定 commit `e0118ade`、Transformers 5.17.0、Accelerate 1.15.0 等）和显卡架构无关，**不需要改**。

### 2. 删除两行只适用于 CUDA 12 的依赖 —— 已确认

`requirements.txt` 中有这两行：

```text
cuda-bindings==12.9.7
cuda-pathfinder==1.6.0
```

它们是为了让 3090 环境与验证时完全一致而固定的，属于 CUDA 12.x 这一套。换成 cu130 版本的 PyTorch 后，它依赖的是 13.x 的对应版本，保留这两行很可能导致安装冲突，**需要删除**。

### 3. 安装后先确认两件事

```bash
# (1) 驱动版本。cu13x 版本的 PyTorch 一般要求驱动 >= 580
nvidia-smi

# (2) 当前 PyTorch 是否包含 sm_120
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_name(0))"
```

- 如果驱动是 **570 系列**，不能用 cu13x 的版本，要改用带 cu128 的 PyTorch。PyTorch 2.14.0 没有 cu128 构建，需要换一个 PyTorch 版本，并重新检查 Diffusers 是否兼容。**这种情况没有验证过**，请先做一次最小的生图测试。
- 如果 `get_arch_list()` 的输出里**没有 `sm_120`**，说明装错了版本，不要继续往下做。

**建议做法**：新建一份 `requirements-cu130.txt`（内容为原 requirements 做完上述 2 处修改），用它来安装：

```bash
conda create -n qwen_image_21 python=3.11 -y
conda activate qwen_image_21
pip install -r requirements-cu130.txt
```

---

## 二、建议修改（不改也能运行，但浪费性能）

### 4. 关闭 CPU offload —— 推断，未实测

- 仓库默认 `"offload": "model"`，原因是 3090 只有 24GB 显存，而模型 BF16 权重约 31GB（文本编码器约 17.5GB，DiT 约 14.2GB）。代价是每一步推理都要在 CPU 和 GPU 之间搬运权重。
- 3090 上的实测数据（40 步）：

  | 分辨率 | 每张耗时 | GPU 峰值显存 | 进程内存 |
  |---|---:|---:|---:|
  | 2K | 约 270 秒 | 约 16.4GB | 约 33GB |

- Pro 6000 有 96GB 显存，**应该**能把整个模型放进显存。脚本本身支持关闭 offload，不需要改代码：

  ```bash
  python scripts/generate_single.py --prompt "..." --seed 42 --gpu 0 \
      --offload none --output outputs/test_nooffload.png
  ```

  `generate_batch.py` 同样支持 `--offload none`。

- 预期效果：速度明显提升，进程内存大幅下降。实际显存占用和速度**需要实测**，本文档不给具体数字。

### 5. VAE 分块解码可关可不关 —— 推断

- 默认 `"vae_tiling": true`（tile 1024 / stride 768）。原因是在 3090 上，2K 图不分块解码时单是 VAE 这一步就要约 24.6GB，会爆显存。96GB 下不分块应该能放下。
- 当前的分块设置在 3090 上实测和不分块解码几乎一致：平均像素差 0.38/255，99 分位差 2/255。**保持开启也没有质量问题**。
- 如果要关：命令行没有这个开关。复制一份配置，改成 `"vae_tiling": false`，再用 `--config` 指定：

  ```bash
  cp configs/default.json configs/pro6000.json
  # 编辑 configs/pro6000.json:  "offload": "none",  "vae_tiling": false
  python scripts/generate_single.py --config configs/pro6000.json --prompt "..." --output outputs/x.png
  ```

  不要直接改 `default.json`，否则 3090 用户拉取后会 OOM。

### 6. 多卡并行时的瓶颈会变化 —— 推断

- 在 3090 上，限制并行进程数的是**系统内存**：每个进程约 33GB。
- 在 Pro 6000 上关闭 offload 后，限制会变成**显存和计算能力**。一张卡上跑两个进程理论上放得下，但两个进程会抢同一块 GPU 的算力，总吞吐不一定更高。
- 建议先测清楚「一张卡一个进程」的速度，再决定是否在一张卡上放多个进程。

---

## 三、其他注意事项

### 7. 3090 与 Pro 6000 的结果不能逐像素复现 —— 推断

- 在同一台机器、同一套配置下，本项目同一 seed 可以逐像素复现（3090 上已验证）。
- 换了显卡架构后，BF16 计算的内部实现会变，图像不会逐像素一致。`offload` 和 `vae_tiling` 的设置也会改变像素。
- 每张图的 metadata JSON 里都记录了 `versions.gpu_name`、`offload` 和 `vae_tiling`，可以追溯。
- **做数据集时**，同一批数据最好固定用一种硬件和一套配置。如果需要混用，把硬件信息当作样本属性记录下来，不要当成同一分布的数据。

### 8. `--gpu` 编号与 `nvidia-smi` 可能不一致 —— 已确认（代码行为）；是否触发取决于服务器

- `--gpu` 的作用是设置 `CUDA_VISIBLE_DEVICES`，CUDA 默认按运算速度给卡编号（`FASTEST_FIRST`），而 `nvidia-smi` 按 PCI 总线顺序编号。
- 所有卡型号相同时两者一般一致。如果服务器上**混插了不同型号的卡**（例如 Pro 6000 和其他卡一起），编号可能对不上，可能在错误的卡上生图。设置以下变量可以让两者一致：

  ```bash
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  ```

### 9. metadata 中的资源记录在 MIG 或 UUID 模式下可能不准 —— 已确认（代码行为）

- `scripts/qi21_common.py` 的 `ResourceMonitor` 从 `CUDA_VISIBLE_DEVICES` 读出**数字**编号，再用 NVML 按编号读取显存和利用率。
- 如果 Pro 6000 开了 **MIG**（把一张卡切成多个实例），或者 `CUDA_VISIBLE_DEVICES` 写的是 `GPU-xxxx` 或 `MIG-xxxx` 形式的 UUID，代码会退回监控 0 号卡，metadata 里的 `nvml_peak_device_used_gb` 和 `gpu_util_*` 就会记成别的卡的数据。
- 这**只影响 metadata 记录，不影响生图**。`torch_peak_allocated_gb` 由 PyTorch 自己统计，不受影响。

### 10. 可以以后再考虑的加速（本仓库未使用，也不建议作为默认）

- Diffusers 文档提到，`QwenImage21FlexAttnProcessor` 配合 `torch.compile` 可以加速。这个方案**未测试**；文档特别提醒，不编译直接用 flex attention 会爆显存。
- Blackwell 硬件支持 FP8/FP4，但使用它们会改变模型的数值精度。按本项目的原则，**BF16 仍是基准配置**；如需使用，应单独做速度与画质的对比实验。

---

## 四、建议的验证顺序

1. `nvidia-smi` 查看驱动版本，决定用 cu130 还是 cu128 版本的 PyTorch。
2. 按第一节修改依赖并新建环境，确认 `get_arch_list()` 中有 `sm_120`。
3. **先不改任何配置**（offload 和 VAE 分块都保持开启），分别生成一张 1024×1024 和一张 2048×2048，确认能正常出图。
4. 再用 `--offload none` 生成 2048×2048，从输出的 metadata JSON 中记下这三项：
   - `generation_time_seconds`
   - `resources_generate.torch_peak_allocated_gb`
   - `resources_generate.process_peak_rss_gb`

   据此决定 Pro 6000 上默认使用哪套配置。
5. 把实测结果反馈给仓库维护者，再考虑正式加入 `requirements-cu130.txt` 和 `configs/pro6000.json`。

---

## 五、改动汇总

| # | 位置 | 改动 | 必要性 | 依据 |
|---|---|---|---|---|
| 1 | `requirements.txt` | `cu126` → `cu130`（3 处） | **必须** | 已确认：cu126 不含 sm_120 |
| 2 | `requirements.txt` | 删除 `cuda-bindings==12.9.7`、`cuda-pathfinder==1.6.0` | **必须** | 已确认：这两个版本属于 CUDA 12 |
| 3 | 环境检查 | 驱动 ≥ 580；`get_arch_list()` 含 `sm_120` | **必须** | cu130 的驱动要求；570 驱动的情况未验证 |
| 4 | 运行参数 / 配置 | `--offload none` | 建议 | 推断，需实测 |
| 5 | 配置 | `"vae_tiling": false`（可选） | 可选 | 推断；保持开启也无质量问题 |
| 6 | 并行方式 | 先测「一张卡一个进程」 | 建议 | 推断 |
| 7 | 数据集规范 | 不混用硬件和配置，或记录硬件信息 | 建议 | 推断 |
| 8 | 环境变量 | `CUDA_DEVICE_ORDER=PCI_BUS_ID`（混插不同型号时） | 视情况 | 已确认（代码行为） |
| 9 | metadata | MIG / UUID 模式下 NVML 记录可能不准 | 已知限制 | 已确认（代码行为） |
