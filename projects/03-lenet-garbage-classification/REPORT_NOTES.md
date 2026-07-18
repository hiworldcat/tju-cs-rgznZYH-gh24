# 实验结果记录

远程项目目录：

```bash
<project-root>
```

实验环境：

- Python 虚拟环境：`.venv`
- MindSpore：2.9.0
- 运行设备：CPU
- 垃圾分类数据集：26 类，训练集 2593 张，测试集 260 张

## 实验主线

本实验不把 MobileNetV2 作为必做内容。MobileNetV2 来自旧参考链接，本实验按老师说明使用 LeNet-5 作为主要分类器，在 MNIST 手写数字预训练权重基础上迁移到垃圾分类任务。

流程：

1. 使用 MNIST 训练 LeNet-5，得到 `checkpoints/lenet5_mnist.ckpt`。
2. 将 LeNet-5 的最后一层从 10 类替换为 26 类。
3. 在垃圾分类数据集上比较随机初始化、冻结微调、全量微调。
4. 做一个模型改进对比：保留 LeNet-5 主体，将输入从灰度图改为 RGB 图像。

## 结果汇总

| 实验 | 说明 | 最佳 epoch | 最佳测试准确率 | 最终准确率 | 可训练参数 | 用时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST 预训练 | LeNet-5, 2 epochs, 20000 train / 5000 test | 2 | 94.96% | 94.96% | 61684 | 24.9s |
| scratch | 原始 LeNet-5，灰度输入，从头训练 | 30 | 57.69% | 57.69% | 63044 | 67.3s |
| frozen | 加载 MNIST 权重，冻结卷积特征层 | 28 | 51.15% | 48.46% | 60494 | 64.4s |
| full | 加载 MNIST 权重，全量微调 | 29 | 58.85% | 55.77% | 63044 | 67.5s |
| lenet5_color | 改进版：RGB 输入，从头训练 | 30 | 72.31% | 72.31% | 63344 | 84.2s |
| lenet5_bn | BatchNorm 尝试，效果失败，可作为反例不写入主结果 | 1 | 3.85% | 3.85% | 63496 | 73.0s |

## 可写进报告的结论

1. MNIST 预训练的 LeNet-5 可以正常收敛，2 个 epoch 在测试子集上达到 94.96%。
2. 在垃圾分类任务上，冻结微调可训练参数更少，但精度低于全量微调，说明手写数字特征与真实垃圾图像差异较大，完全冻结卷积层限制了特征适配能力。
3. 全量微调略优于原始灰度 LeNet-5，说明 MNIST 预训练对初始化有一定帮助，但提升有限。
4. RGB 输入改进显著提升准确率，因为垃圾分类图像中颜色是重要判别特征；灰度化会丢失塑料瓶、果皮、纸张、金属等类别的颜色信息。
5. BatchNorm 版本在当前小数据集和训练设置下泛化失败，说明改进结构需要实验验证，不能简单认为加模块一定提升。

## 复现实验命令

```bash
cd <project-root>
source .venv/bin/activate

python train_mnist.py --epochs 2 --train-limit 20000 --test-limit 5000
python finetune_garbage.py --strategy scratch --epochs 30 --lr 0.001
python finetune_garbage.py --strategy frozen --epochs 30 --lr 0.001
python finetune_garbage.py --strategy full --epochs 30 --lr 0.0005
python finetune_garbage.py --variant lenet5_color --strategy scratch --epochs 30 --lr 0.001
```

结果文件：

```bash
results/mnist_pretrain.csv
results/garbage_scratch.csv
results/garbage_frozen.csv
results/garbage_full.csv
results/lenet5_color_garbage_scratch.csv
```
