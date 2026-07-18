# 机器人避障寻径

本项目研究栅格地图下的机器人路径规划，包含经典搜索算法和启发式算法：

- BFS;
- A*;
- Dijkstra;
- 遗传算法；
- 人工势场法。

实现支持简单地图与复杂地图，能够生成搜索过程可视化、最终路径图和性能对比图。

## 文件

```text
robot_path_planning.py
examples/
  simple/
  complex/
```

`examples/` 目录只保留原始结果中的少量轻量图片。完整 GIF 动画和提交压缩包没有复制进来。

## 依赖

```bash
pip install numpy matplotlib pillow
```

## 运行

```bash
python robot_path_planning.py
```

脚本会提示选择地图类型，运行各类算法，并将生成结果写入 `results/`。

## 报告摘要

私有报告比较了路径长度、探索节点数和运行时间。A* 在静态栅格地图上取得了较好的综合平衡；BFS 和 Dijkstra 在最短路径搜索上可靠，但探索节点更多；GA 和 APF 适合讨论启发式方法的行为特点，但在该静态栅格任务中稳定性较弱。
