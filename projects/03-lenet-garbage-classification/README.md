# 基于 LeNet-5 的 MNIST 与垃圾分类迁移学习

本项目使用 MindSpore 实现一个深度学习实验：

1. 在 MNIST 上预训练 LeNet-5；
2. 将模型迁移到 26 类垃圾分类任务；
3. 比较从头训练、冻结微调和全量微调；
4. 测试一个轻量的 RGB 输入版 LeNet-5 改进模型。

## 文件

```text
train_mnist.py
finetune_garbage.py
run_experiments.sh
requirements.txt
REPORT_NOTES.md
src/
results/
```

模型权重和数据集没有纳入公开整理目录。

## 依赖

```bash
pip install -r requirements.txt
```

MindSpore 的安装方式与平台和设备后端有关，需要根据本机环境选择合适版本。

## 数据集

原实验使用 MNIST 和 26 类垃圾分类数据集。垃圾分类数据集期望目录结构如下：

```text
data/data_en/data_en/
  train/
    <class-name>/
  test/
    <class-name>/
```

## 运行

```bash
python train_mnist.py --epochs 2 --train-limit 20000 --test-limit 5000
python finetune_garbage.py --strategy scratch --epochs 30 --lr 0.001
python finetune_garbage.py --strategy frozen --epochs 30 --lr 0.001
python finetune_garbage.py --strategy full --epochs 30 --lr 0.0005
python finetune_garbage.py --variant lenet5_color --strategy scratch --epochs 30 --lr 0.001
```

## 报告摘要

实验发现，MNIST 预训练可以很好地完成手写数字识别，但迁移到真实垃圾图像时存在明显领域差异。冻结微调效果较差，因为早期特征仍然偏向数字图像；全量微调略优于灰度输入的从头训练；RGB 输入版通过保留颜色信息，带来了最明显的分类性能提升。
