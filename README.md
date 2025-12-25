# Zhang 相机标定项目（NJU 数字图像处理与计算机视觉课设 231180007）

基于 Zhang (2000) 方法的相机标定工程。一次执行完成“标定 + 评估 + 打包输出”。核心流程：
- 棋盘角点检测 → 单应矩阵估计（可选 RANSAC）→ Zhang 线性内参初值 → OpenCV `calibrateCamera` 联合优化
- 统一 CLI 脚本，支持随机比例/目录/角点位置三种数据集划分；输出残差可视化与空间误差热力图，并写入综合 JSON 报告。

---

## 项目结构

- `calib/` — 标定库
  - `calib/calibrate.py` — 角点检测、单应估计、Zhang 线性解、外参恢复、联合优化（cv2.calibrateCamera）；同时提供基于“点集合”的标定接口 `calibrate_from_points`。
- `scripts/` — 命令行脚本
  - `scripts/calibrate_cli.py` — 统一 CLI（all 模式），数据划分与评估可视化、统一打包输出
  - `scripts/generate_chessboard.py` — 生成可打印的棋盘板（含多种外观/质量变体）
  - `scripts/visualize_dataset_corners.py` — 聚合散点显示数据集角点分布（按图片分色）
- `images/` — 示例/输出目录（脚本会写入）
  - `images/boards/` — 生成的棋盘板各变体（standard/blur/affine/perspective/contrast/inverted/noise）
- `run/` — 历次运行的输出（如果配置为写入此处）
- `理论文档/` — 标定原理与实现说明（HTML）
- `requirements.txt` — Python 依赖（numpy, opencv-python, scipy, matplotlib）

---

## 环境准备（Linux/bash）

```bash
# 建议使用 Conda（Python 3.9~3.11 均可）
conda create -n cam python=3.10 -y
conda activate cam

# 安装依赖
pip install -r requirements.txt

# 运行前将项目根加入 PYTHONPATH（推荐在项目根执行）
PYTHONPATH=$(pwd) python scripts/calibrate_cli.py --help
```

---

## 统一 CLI

一次运行完成标定与评估，并把所有输出集中到 `out/<bundle>/` 下。

示例（随机比例划分）：

```bash
PYTHONPATH=$(pwd) python scripts/calibrate_cli.py \
  --images 'images/11_15_2058/*.jpg' \
  --pattern 9 6 \
  --square 0.029 \
  --use-ransac \
  --val-ratio 0.3 \
  --split-seed 42 \
  --bundle run_1221 \
  --per-image-heatmap
```

示例（目录划分）：

```bash
PYTHONPATH=$(pwd) python scripts/calibrate_cli.py \
  --train-dir images/train \
  --val-dir images/val \
  --pattern 9 6 \
  --square 0.029 \
  --use-ransac \
  --bundle run_dir_split
```

示例（角点位置划分：仅用指定区域的角点进行训练，验证使用全量角点）：

```bash
PYTHONPATH=$(pwd) python scripts/calibrate_cli.py \
  --image 'images/a/*' \
  --pattern 16 12 \
  --square 0.015 \
  --point-split \
  --train-urange 0.3 0.6 \
  --train-vrange 0.3 0.6 \
  --bundle run_point_middle \
  --use-ransac
```

### 参数说明

基础输入：
- `--images` / `--image`：图片 glob（用引号包裹，避免 shell 展开）
- `--pattern cols rows`：棋盘“内角点”数（如 9 6）
- `--square`：棋盘方格物理尺寸（单位自定，影响外参尺度，不影响像素 RMS）

标定选项：
- `--use-ransac`：单应矩阵估计使用 RANSAC，提高鲁棒性
- `--free-skew`：允许估计 skew（默认固定为 0）
- `--free-tangential`：允许切向畸变 p1/p2（默认置零）
- `--enable-k3`：允许三阶径向畸变 k3（默认固定为 0）
- `--fix-principal-point`：固定主点（若 OpenCV 版本支持）
- `--iter` / `--eps`：OpenCV 联合优化终止准则（最大迭代，收敛阈值）

评估/输出选项：
- `--show-vis`：在屏幕显示残差可视化
- `--arrow-scale`：残差箭头缩放系数
- `--grid cols rows`：空间误差统计网格尺寸（默认 10×8）
- `--per-image-heatmap`：输出每张图片的热力图（默认只输出聚合热力图）
- `--bundle name`：将所有输出打包到 `out/<name>/`

数据集划分（三选一，优先级从上到下）：
- 角点位置划分：`--point-split --train-urange umin umax --train-vrange vmin vmax`（训练仅用位于该归一化区域的角点；验证使用全量角点）
- 目录划分：`--train-dir ... --val-dir ...`（各目录下的所有文件分别作为训练/验证）
- 随机比例划分：`--images ... --val-ratio r --split-seed s`

---

## 输出内容与可视化

运行后，统一输出到 `out/<bundle>/`：

- 根目录：
  - `results_all.json`：综合报告（含划分信息、标定结果、train/val/overall 评估与图片列表）
  - `rms_heatmap_all.png`：所有图片聚合的空间误差热力图
- 训练子目录 `out/<bundle>/train/`：
  - 每张训练图片的残差可视化 PNG（箭头颜色随残差值增大趋近红色）
  - `rms_heatmap.png`：训练集聚合热力图
  - `results_train.json`：训练集评估摘要
- 验证子目录 `out/<bundle>/val/`：
  - 每张验证图片的残差可视化 PNG（若角点检测成功）
  - `rms_heatmap.png`：验证集聚合热力图
  - `results_val.json`：验证集评估摘要

空间热力图说明：
- 先计算每个角点的重投影残差幅值（像素），将其按归一化坐标 $(u/W, v/H)$ 落入 `cols×rows` 网格聚合，计算每格 RMS。
- 使用 OpenCV `COLORMAP_JET` 着色；没有数据的格子显示为黑色。

`results_all.json` 字段摘要：
- 顶层：`timestamp`、`opencv_version`、`images_glob` 或 `train_dir/val_dir`、`matched_image_count`、`pattern`、`square`、`use_ransac`、`split`
- `calibrate`：`used_image_count`、`used_images`、`K_init`、`K`、`dist(k1,k2)`、`calibrate_rms`
- `train_evaluate / val_evaluate / overall_evaluate`：各子集的 `overall_reprojection_rms_px`、`per_image_rms_px`、`spatial_distribution(rms_grid/counts_grid)`、`visualization_files`

注意与建议：
- 标定与评估至少需要 3 张成功检测到角点的视图；点级划分模式下，每张训练视图需 ≥4 个角点。
- 若训练角点集中在狭小区域，Zhang 线性初值可能退化；请保证足够的视角变化与角点数量。

---

## 生成可打印棋盘板（variants）

脚本：`scripts/generate_chessboard.py`

支持生成以下变体（逗号分隔传入 `--variants`）：
- `standard`（标准黑白高对比）、`blur`（高斯模糊）、`affine`（轻微旋转缩放）、`perspective`（轻微透视倾斜）、
  `contrast`（对比度/亮度/伽马）、`inverted`（黑白反转）、`noise`（高斯噪声）

示例：

```bash
PYTHONPATH=$(pwd) python scripts/generate_chessboard.py \
  --pattern 9 6 --square 50 --count 5 \
  --variants standard,blur,perspective,noise,contrast \
  --affine-rotate 5 --perspective-tilt 0.07 --noise-sigma 6
```

输出：`images/boards/<variant>/chess_<cols>x<rows>_<index>.png`

---

## 聚合可视化数据集角点（散点）

脚本：`scripts/visualize_dataset_corners.py`

作用：将所有图片的角点按归一化坐标聚合到同一画布，按“图片分组”上色，便于观察角点全局分布与覆盖范围。

示例：

```bash
PYTHONPATH=$(pwd) python scripts/visualize_dataset_corners.py \
  --images 'images/11_15_2058/*.jpg' \
  --pattern 9 6 \
  --square 0.029 \
  --out out/corners_dataset_scatter.png
```

输出：`out/corners_dataset_scatter.png`（画布尺寸取数据集中图片的原始大小；点颜色按图片分组区分）

---

## 参考资料

- Zhang, Z. (2000). A Flexible New Technique for Camera Calibration. IEEE TPAMI.
- OpenCV Calibration: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

