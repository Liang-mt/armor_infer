# Armor Infer — RoboMaster 装甲板识别推理系统

基于 ONNX Runtime 的 RoboMaster 竞赛装甲板检测与数字识别系统，使用沈阳航空航天大学 TUP 战队 2022 赛季开源模型。

---

## 功能特性

- **装甲板检测**：基于 YOLOX 模型，实时检测画面中的装甲板，输出四点关键点坐标、颜色（蓝/红）、类别（1-5 号/哨兵/前哨站/基地）
- **能量机关检测**：支持五点关键点的 buff（能量机关）目标检测
- **数字识别**：两阶段方案 —— 先检测装甲板位置，再通过透视变换提取 ROI，使用 MLP 网络识别装甲板数字
- **缩放跟踪**：自动锁定最大装甲板并平滑缩放放大，便于远程观察

---

## 项目结构

```
armor_infer/
├── main.py                  # 入口：基础检测 + 数字识别
├── main2.py                 # 入口：带缩放跟踪的检测
├── model/                   # 预训练 ONNX 模型
│   ├── yolox.onnx           #   主检测模型（装甲板 4 点）
│   ├── buff_300.onnx        #   能量机关检测模型（5 点）
│   ├── mlp.onnx             #   MLP 数字分类模型
│   ├── og1000.onnx          #   备选检测模型（1000 epoch）
│   └── s_armor800.onnx      #   备选检测模型（800 epoch）
├── utils/                   # 核心工具包
│   ├── base_infer.py        #   推理基类 BaseInfer
│   ├── infer.py             #   单阶段推理类 Infer
│   ├── infer2.py            #   两阶段推理类 Infer2（检测 + 数字识别）
│   ├── mlp_predict.py       #   MLP 数字分类器 NumberClassifier
│   ├── utils.py             #   预处理、后处理、NMS、可视化
│   └── datasets.py          #   类别名称定义
├── video/                   # 测试视频（装甲板 / buff）
├── 前哨站/                   # 前哨站视角测试视频
└── images/                  # 测试图片
```

---

## 环境安装

```bash
pip install onnxruntime numpy opencv-python torch torchvision
```

> 如需 GPU 推理，请参考 [PyTorch 官网](https://pytorch.org/) 安装对应 CUDA 版本。

---

## 快速开始

### 基础用法 — 装甲板检测 + 数字识别

```bash
python main.py
```

默认使用 `yolox.onnx` 模型对 `./video/1.mp4` 进行推理，结果输出到 `output_video.avi` 并实时显示。

### 带缩放跟踪的检测

```bash
python main2.py
```

使用 `buff_300.onnx` 模型对 `./video/4.avi` 进行推理，同时显示原始画面和缩放跟踪画面。

按 `q` 键退出。

---

## 使用指南

### 切换模型

在入口文件中修改模型路径和参数：

```python
from utils import Infer, Infer2

# 装甲板检测（组合类别模型，如 TUP 格式）
predictor = Infer("./model/yolox.onnx", num_apex=4, num_class=9, num_color=4)

# 装甲板检测 + 数字识别（3 类颜色模型）
predictor = Infer2("./model/yolox.onnx", num_apex=4, num_class=1, num_color=3)

# 能量机关检测
predictor = Infer("./model/buff_300.onnx", num_apex=5, num_class=2, num_color=2)
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `num_apex` | 关键点数量（装甲板=4，能量机关=5） | `4` |
| `num_class` | 类别数量（根据模型训练配置） | `1` / `2` / `9` |
| `num_color` | 颜色数量 | `3` / `4` |
| `mlp_model_path` | MLP 模型路径（仅 `Infer2`） | `"./model/mlp.onnx"` |

### 自定义类别名称

在 `utils/datasets.py` 中编辑对应的类别元组：

```python
# TUP 格式（4 颜色 × 9 类型 = 36 类）
COCO_CLASSES_tup = (
    "BG", "B1", "B2", "B3", "B4", "B5", "BO", "BB",
    "RG", "R1", "R2", "R3", "R4", "R5", "RO", "RB",
    "NG", "N1", "N2", "N3", "N4", "N5", "NO", "NB",
    "PG", "P1", "P2", "P3", "P4", "P5", "PO", "PB",
)
```

颜色编码：`B`=蓝 `R`=红 `N`=中立 `P`=紫
类型编码：`G`=哨兵 `1-5`=机器人编号 `O`=前哨站 `B`=基地

---

## 架构说明

### 推理流程

```
输入帧
  │
  ▼
┌─────────────────────────────┐
│  BaseInfer.inference()       │
│  1. 预处理（resize + pad）    │
│  2. ONNX 推理               │
│  3. 解码关键点坐标            │
│  4. 颜色×类别预测合并         │
│  5. NMS 后处理               │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  visual()                    │
│  解析检测结果 → 字典列表      │
└─────────────────────────────┘
  │
  ├─ Infer:   直接绘制检测框
  │
  └─ Infer2:  透视变换提取 ROI
              → 二值化
              → MLP 数字分类
              → 绘制检测框 + 数字标签
```

### 类继承关系

```
BaseInfer                     # 公共推理逻辑
├── Infer                     # 单阶段：检测 + 颜色/类别解码
├── Infer2                    # 两阶段：检测 + MLP 数字识别
└── ZoomPredictor             # 检测 + 缩放跟踪（main2.py）
```

---

## 参考项目

- [RangerOnMars/TUP-NN-Train](https://github.com/RangerOnMars/TUP-NN-Train) — YOLOX 装甲板检测训练框架
- [tup-robomaster/TUP-InfantryVision-2022](https://github.com/tup-robomaster/TUP-InfantryVision-2022) — TUP 战队 2022 赛季步兵视觉程序

---

## 测试视频

`video/` 文件夹内提供装甲板和能量机关测试视频。如需下载更多视频：

> 链接：https://pan.baidu.com/s/1QykXf3QvKQdGDIeRvCxdzw?pwd=0000
> 提取码：0000
