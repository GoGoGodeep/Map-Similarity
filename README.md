# 🌍 Map-Similarity: 多场景地理图像相似度分析

基于OpenCV实现**林地/荒漠/雷达/红外**等多模态地理图像的场景自适应相似度计算

![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green)

## 🛠️ 场景适配方案

### 🌳 林地场景

### 🏜️ 荒漠场景

### 📡 雷达图像

### 🔥 红外图像

---

## 📂 技术架构
```bash
└── Similarity.py    # 🧮 核心算法实现
```

## 🚀 快速使用
```python
from Similarity import SceneComparator

# 初始化场景处理器（可选类型：forest/desert/radar/infrared）
comparator = SceneComparator(scene_type="forest")

# 计算相似度得分（0.0~1.0）
score = comparator.compare("img1.jpg", "img2.jpg")
```
