# 基于 Stable Diffusion 的古诗转水墨画生成

本项目基于 Stable Diffusion v1.5 构建一个可运行流程，用于从中国古典诗句生成水墨画风格图像。

私有报告和最终 PDF 含有个人封面元数据，因此没有复制进来。本 README 用于概括报告内容和公开源码。

## 目标

- 将诗句和视觉意象转换为正向提示词；
- 使用反向提示词抑制水印、文字伪影、写实摄影感和西式油画风格；
- 使用节省显存的设置运行 Stable Diffusion 推理；
- 比较采样步数和 guidance scale；
- 提供数据集准备和 LoRA 轻量微调脚本。

## 文件

```text
src/
  generate.py
  compare_params.py
  prepare_dataset.py
  train_lora.py
configs/
scripts/
examples/
requirements.txt
```

大模型目录、下载权重、LoRA 权重和完整生成结果目录均未纳入公开整理目录。

## 依赖

```bash
pip install -r requirements.txt
```

原实验使用 NVIDIA RTX 2080 Ti 11GB 显存环境，主要依赖 fp16 推理、attention slicing、batch size 1 和梯度累积来降低显存压力。

## 下载模型

```bash
python scripts/download_model.py \
  --model-id stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --local-dir models/stable-diffusion-v1-5
```

## 生成图像

```bash
python src/generate.py \
  --model models/stable-diffusion-v1-5 \
  --prompts configs/prompts_ink.csv \
  --out outputs/base \
  --steps 30 \
  --guidance-scale 8.0 \
  --seed 42
```

## 参数对比

```bash
python src/compare_params.py \
  --model models/stable-diffusion-v1-5 \
  --prompt "孤舟蓑笠翁，独钓寒江雪，Chinese ink wash painting, misty river, monochrome" \
  --out outputs/compare
```

## 报告摘要

提示词工程能够引导模型生成孤舟、寒江、瀑布、春雨、群山等诗意元素。采样步数从 20 提升到 30 或 40 后，图像结构和宣纸纹理更稳定；guidance scale 过低时诗意对应关系变弱，过高时画面可能拥挤或风格偏移。短程 LoRA 测试验证了训练流程和权重保存逻辑可运行，为后续使用更高质量水墨数据集进行风格适配打下基础。
