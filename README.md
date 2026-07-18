# 人工智能原理与技术课程整理

本仓库是 **人工智能原理与技术** 课程作业的公开版整理目录。

原始课程文件夹中混有实验报告、课件、源码、生成结果、提交压缩包、模型产物和带隐私信息的 PDF。当前工作区只保留适合公开上传到 GitHub 的内容：

- 四个大作业/实验的可复现源码；
- 少量轻量级示例输出；
- 已脱敏的小理论作业图片；
- 已匿名的期中调研 LaTeX 源码；
- 用 README 摘要替代不适合公开的完整报告。

## 仓库结构

```text
projects/
  01-eight-queens/                 八皇后逻辑编程与搜索
  02-robot-path-planning/          BFS、A*、Dijkstra、GA、APF 机器人避障寻径
  03-lenet-garbage-classification/ 基于 MindSpore 的 LeNet-5 迁移学习
  04-poetry-ink-diffusion/         基于 Stable Diffusion 的古诗转水墨画生成

assignments/
  theory/                          小理论作业，已整理为脱敏图片
  midterm-text-to-audio-survey/     期中调研 LaTeX 源码

course-materials/
  README.md                        未纳入课程资料说明

docs/
  privacy.md                       脱敏与排除策略
```

## 未纳入的内容

以下内容没有复制进公开整理目录：

- 含姓名、学号、任课教师、封面元数据的最终提交 PDF；
- 课程课件、教材 PDF 和第三方教学资料；
- 提交用压缩包；
- 模型 checkpoint、Stable Diffusion 模型目录、LoRA 权重和 Python 缓存；
- 大体积生成结果目录，仅保留少量轻量示例。

## 四个大作业/实验

1. **八皇后问题**：比较穷举、DFS、BFS 和 kanren 风格逻辑编程在约束满足问题中的表现。
2. **机器人避障寻径**：在栅格地图上实现并可视化 BFS、A*、Dijkstra、遗传算法和人工势场法。
3. **手写数字识别与垃圾分类**：使用 MindSpore 在 MNIST 上预训练 LeNet-5，并比较从头训练、冻结微调、全量微调和 RGB 输入改进。
4. **古诗转水墨画生成**：构建 Stable Diffusion v1.5 文生图流程，包含提示词工程、参数对比、数据准备和 LoRA 训练脚本。

## 隐私说明

本仓库是课程工作区的公开版整理结果：保留技术内容，移除个人身份信息和私有提交产物。详细规则见 [docs/privacy.md](docs/privacy.md)。
