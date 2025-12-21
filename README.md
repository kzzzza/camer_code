# Zhang 相机标定（NJU数字图像处理与计算机视觉大作业231180007）

基于张友定方法 (Zhang, 2000) 的灵活相机标定工程。核心流程：
- 检测棋盘角点 → 估计单应矩阵（可选 RANSAC）→ 张氏线性内参初值 → OpenCV `calibrateCamera` 联合优化
- 提供生成标定板、标定、可视化、评估脚本，便于做参数对比实验（skew/切向畸变/k3、RANSAC、迭代准则等）

---

## 工程代码目录结构

- `calib/` — 标定库实现
  - `calib/calibrate.py` — 核心：角点检测、单应估计、Zhang 线性解、外参恢复、联合优化（cv2.calibrateCamera）
- `scripts/` — 命令行脚本
  - `scripts/calibrate_cli.py` — 统一 CLI
  - `scripts/generate_chessboard.py` — 生成可打印棋盘板及多种外观/质量变体
- `images/` — 示例/输出目录（脚本会在此写入）
  - `images/boards/` — 可打印棋盘板各变体输出（standard/blur/perspective/...）
  - `images/visualized/` — 可视化输出（角点、重投影残差）
- `runs/` — 实验日志存放目录
- `理论文档` — 理论说明HTML文档存放目录
- `requirements.txt` — Python 依赖

---

## 1) Conda 环境配置

推荐在 Conda 环境中安装并运行（Linux/bash）：

```bash
# 1) 创建并激活环境（可选 Python 3.9~3.11）
conda create -n cam python=3.10 -y
conda activate cam

# 2) 安装依赖
pip install -r requirements.txt

```
请在运行脚本时设置 `PYTHONPATH` 指向项目根：

```bash
PYTHONPATH=$(pwd) python scripts/calibrate_cli.py --help
```

---

## 2) 命令行使用说明

本项目的统一 CLI 已简化为单一模式 `--mode all`，旧的子命令（calibrate / evaluate / split-eval / visualize）已在代码中移除。请使用如下方式在一次运行中完成标定与评估，并统一输出到指定 bundle 目录。

示例（建议使用 RANSAC 与默认稳健设置）：

```bash
PYTHONPATH=$(pwd) python scripts/calibrate_cli.py --mode all \
  --images 'images/11_15_2058/*.jpg' \
  --pattern 9 6 \
  --square 0.029 \
  --use-ransac \
  --val-ratio 0.3 \
  --split-seed 42 \
  --bundle run_1221 \
  --per-image-heatmap
```

参数表（all 模式）

| 选项 | 类型/默认 | 说明 |
|---|---|---|
| `--images` / `--image` | str, 必填 | 图片 glob（用引号包裹，避免 shell 展开） |
| `--pattern` | int int, 必填 | 棋盘“内角点”数：cols rows（如 9 6） |
| `--square` | float, 必填 | 方格物理尺寸（单位自定，影响外参尺度，不影响像素 RMS） |
| `--use-ransac` | flag=false | 单应使用 RANSAC，提高鲁棒性 |
| `--free-skew` | flag=false | 允许估计 skew（不推荐，一般相机应≈0） |
| `--free-tangential` | flag=false | 允许切向畸变 p1/p2（默认置零） |
| `--enable-k3` | flag=false | 允许 k3（默认固定 0） |
| `--fix-principal-point` | flag=false | 固定主点（需 OpenCV 支持该 flag） |
| `--iter` | int=100 | 最大迭代次数（传入 OpenCV 终止准则） |
| `--eps` | float=1e-6 | 收敛阈值（传入 OpenCV 终止准则） |
| `--show-vis` | flag=false | 屏幕显示可视化图像（评估阶段） |
| `--arrow-scale` | float=1.0 | 评估残差箭头缩放系数 |
| `--grid` | int int=10 8 | 空间误差统计网格（列 行），用于聚合所有图的误差分布 |
| `--per-image-heatmap` | flag=false | 同时输出每张图片的误差热力图（输出到 bundle 目录） |
| `--bundle` | str=None | 将输出统一打包到项目根 `out/<name>/` 下（单一 JSON + 所有图片） |
| `--val-ratio` | float=0.3 | 验证集占比（其余作为训练集，仅用训练集进行标定） |
| `--split-seed` | int=42 | 数据划分随机种子（保证复现） |

输出内容（bundle 模式，带训练/验证划分）：

- 根目录：`out/<bundle>/`
  - 综合 JSON：`results_all.json`（含 split 信息、训练标定结果、train/val 的评估指标与可视化文件列表）
  - 训练子目录：`out/<bundle>/train/`
    - 每张训练图片残差可视化 PNG
    - 训练集聚合热力图：`rms_heatmap.png`
    - 子集 JSON：`results_train.json`
  - 验证子目录：`out/<bundle>/val/`
    - 每张验证图片残差可视化 PNG（若角点检测成功）
    - 验证集聚合热力图：`rms_heatmap.png`
    - 子集 JSON：`results_val.json`

可视化与色彩规则：

- 残差图：在每张图片上绘制投影点与观测点的连线箭头，箭头颜色按残差像素值映射（越大越趋近红色），支持 `--arrow-scale` 控制箭头长度。
- 空间热力图：将所有图像的归一化像素坐标按 `--grid` 指定的网格（默认 10×8）进行聚合统计，对每格的 RMS 残差用 OpenCV 的 `COLORMAP_JET` 着色；没有数据的格子显示为黑色。

`results_all.json` 字段摘要：

- 顶层：`timestamp`、`opencv_version`、`mode='all'`、`images_glob`、`matched_image_count`、`pattern`、`square`、`use_ransac`
- `calibrate`：`used_image_count`、`used_images`、`K_init`、`K`、`dist`、`calibrate_rms`
- `evaluate`：`overall_reprojection_rms_px`、`per_image_rms_px`、`spatial_distribution.rms_grid` 与 `counts_grid`、`visualization_files`


以下命令需在项目根执行，或显式设置 `PYTHONPATH=/path/to/project_root`。


## 参考

- Zhang, Z. (2000). A Flexible New Technique for Camera Calibration. IEEE TPAMI.
- OpenCV Calibration: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

