# Zhang 相机标定（可调参数 | 详尽说明）

这是一个基于张友定方法 (Zhang, 2000) 的灵活相机标定工程。核心流程：
- 检测棋盘角点 → 估计单应矩阵（可选 RANSAC）→ 张氏线性内参初值 → OpenCV `calibrateCamera` 联合优化
- 提供生成、可视化、评估脚本，便于做参数对比实验（skew/切向畸变/k3、RANSAC、迭代准则等）

本文档包含：
1) Conda 环境配置与安装；2) 命令行使用说明（参数表格+示例）；3) 标定实验流程图（含可选分支）。

---

## 目录结构（关键文件）

- `calib/` — 标定库实现
  - `calib/calibrate.py` — 核心：角点检测、单应估计、Zhang 线性解、外参恢复、联合优化（cv2.calibrateCamera）
- `scripts/` — 命令行脚本
  - `scripts/calibrate_cli.py` — 运行标定（读取图片 glob、参数/自由度开关）
  - `scripts/evaluate_intrinsics.py` — 评估重投影 RMS、误差可视化、可与 GT 对比
  - `scripts/visualize_corners.py` — 检测并绘制棋盘角点
  - `scripts/generate_chessboard.py` — 生成标准棋盘与仿真拍摄图
- `images/` — 示例/输出目录（脚本会在此写入）
  - `images/generated/` — 标准正视棋盘图（无畸变，top-down）
  - `images/calib/` — 仿真相机拍摄图（透视变形、噪声、模糊）
  - `images/visualized/` — 可视化输出（角点、重投影残差）
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

# 3) 可选：本地开发模式安装（便于任何位置导入 calib 包）
pip install -e .
```

如不使用可编辑安装，请在运行脚本时设置 `PYTHONPATH` 指向项目根：

```bash
PYTHONPATH=$(pwd) python scripts/calibrate_cli.py --help
```

---

## 2) 命令行使用说明

以下命令需在项目根执行，或显式设置 `PYTHONPATH=/path/to/project_root`。

### a) 标定：`scripts/calibrate_cli.py`

示例（RANSAC + 默认稳健设置）：

```bash
PYTHONPATH=$(pwd) python scripts/calibrate_cli.py \
  --images 'images/calib/*.png' \
  --pattern 9 6 \
  --square 0.029 \
  --use-ransac
```

参数表：

| 选项 | 类型/默认 | 说明 |
|---|---|---|
| `--images` | str, 必填 | 标定图片 glob（用引号包裹，避免 shell 提前展开） |
| `--pattern` | int int, 必填 | 棋盘“内角点”数：cols rows（如 9 6） |
| `--square` | float, 必填 | 方格物理尺寸（单位自定，影响外参尺度，不影响像素 RMS） |
| `--use-ransac` | flag=false | 单应使用 RANSAC，提高鲁棒性 |
| `--free-skew` | flag=false | 允许估计 skew（不推荐，一般相机应≈0） |
| `--free-tangential` | flag=false | 允许切向畸变 p1/p2（默认置零） |
| `--enable-k3` | flag=false | 允许 k3（默认固定 0） |
| `--fix-principal-point` | flag=false | 固定主点（需 OpenCV 支持该 flag） |
| `--iter` | int=100 | 最大迭代次数（传入 OpenCV 终止准则） |
| `--eps` | float=1e-6 | 收敛阈值（传入 OpenCV 终止准则） |

输出：初始/优化 `K`、`dist(k1,k2)`、使用图像数、整体 RMS（px）等。

注意：若出现 `ModuleNotFoundError: calib`，使用上面的 `PYTHONPATH=$(pwd)` 或执行 `pip install -e .`。

---

### b) 评估与误差可视化：`scripts/evaluate_intrinsics.py`

示例（保存 JSON 报告与可视化）：

```bash
PYTHONPATH=$(pwd) python scripts/evaluate_intrinsics.py \
  --images 'images/calib/*.png' \
  --pattern 9 6 \
  --square 0.029 \
  --out eval_report.json \
  --vis-out images/eval_visualized \
  --show-vis \
  --arrow-scale 4.0
```

参数表：

| 选项 | 类型/默认 | 说明 |
|---|---|---|
| `--images` | str, 必填 | 图片 glob |
| `--pattern` | int int, 必填 | 内角点数：cols rows |
| `--square` | float, 必填 | 方格物理尺寸 |
| `--use-ransac` | flag=false | 与标定一致，便于对齐流程 |
| `--gt-k` | 路径=None | 可选：真实内参（.npy 或 JSON）做对比 |
| `--out` | 路径=None | 保存评估报告 JSON |
| `--vis-out` | 路径=None | 重投影残差可视化输出目录 |
| `--show-vis` | flag=false | 屏幕显示可视化图像 |
| `--arrow-scale` | float=1.0 | 误差向量缩放系数（仅可视化） |

输出：总体 RMS、每图 RMS、可选的 GT 对比、可视化残差图。

---

### c) 角点可视化：`scripts/visualize_corners.py`

示例：

```bash
PYTHONPATH=$(pwd) python scripts/visualize_corners.py \
  --images 'images/calib/*.png' \
  --pattern 9 6 \
  --square 0.029 \
  --out images/visualized \
  --show
```

参数表：

| 选项 | 类型/默认 | 说明 |
|---|---|---|
| `--images` | str, 必填 | 图片 glob |
| `--pattern` | int int, 必填 | 内角点数：cols rows |
| `--square` | float=1.0 | 物理尺寸（仅用于生成 objpoints，可任意非零） |
| `--out` | 路径=images/visualized | 输出目录 |
| `--show` | flag=false | 屏幕显示可视化 |

---

### d) 生成棋盘与仿真图：`scripts/generate_chessboard.py`

示例：

```bash
python scripts/generate_chessboard.py \
  --pattern 9 6 \
  --square 50 \
  --count 12 \
  --max-perturb 0.12 \
  --blur 3 \
  --noise 4.0 \
  --seed 42
```

参数表：

| 选项 | 类型/默认 | 说明 |
|---|---|---|
| `--pattern` | int int, 必填 | 内角点数：cols rows |
| `--square` | int=50 | 每格像素大小（生成图像用像素单位） |
| `--count` | int=10 | 生成张数 |
| `--out` | 路径=./images | 输出根目录（自动创建 generated/ 与 calib/） |
| `--max-perturb` | float=0.12 | 透视扰动相对幅度（相对图像尺寸） |
| `--seed` | int=None | 随机种子（可复现） |
| `--blur` | int=3 | 高斯模糊核大小（奇数，0 表示不模糊） |
| `--noise` | float=4.0 | 高斯噪声强度（像素） |

生成：
- `images/generated/` 正视棋盘（无畸变）
- `images/calib/` 仿真“拍摄”图（含随机透视、模糊与噪声）

---

## 3) 标定实验流程图（含可选分支）

下图描述了本工程的标定数据流与可选开关对流程的影响：

```mermaid
flowchart TD
  A[输入图片 (glob)] --> B{角点检测\nfindChessboardCorners}
  B -->|< 3 视图| E[错误: 需至少 3 张有效视图]
  B -->|>= 3 视图| C[单应估计\nDLT 或 RANSAC]
  C --> D[Zhang 线性内参\nK_init]
  D --> F{构建标定 flags}
  F -->|--free-skew| F1[估计 skew]
  F -->|默认| F2[固定 skew=0]
  F -->|--free-tangential| F3[启用切向 p1,p2]
  F -->|默认| F4[切向=0]
  F -->|--enable-k3| F5[启用 k3]
  F -->|默认| F6[固定 k3=0]
  F -->|--fix-principal-point| F7[固定主点]
  D --> G[Sanitize K_init\n(主点入图/焦距正/零 skew)]
  F --> H[OpenCV calibrateCamera\n联合优化]
  G --> H
  H --> I[输出: K, dist(k1,k2), rvecs, tvecs, RMS]
  I --> J{评估?}
  J -->|evaluate_intrinsics.py| K[每图 RMS + 残差可视化]
  I --> L{角点可视化?}
  L -->|visualize_corners.py| M[绘制角点并保存]
```

---

## 典型实验与对比

- RANSAC 对比：同一数据，分别开启/关闭 `--use-ransac`，比较整体 RMS 与参数稳定性。
- 畸变自由度：对高畸变镜头尝试 `--enable-k3`，或对装配偏差较大时尝试 `--free-tangential`，比较前后 RMS。
- 主点策略：`--fix-principal-point` 与默认可调主点对比（对低视角多样性数据可更稳）。

---

## 常见问题 (FAQ)

- 找不到棋盘角点：检查 `--pattern` 是否与图片一致；提高分辨率、减少模糊/噪声；尝试更好的光照。
- 线性初值失败：确保视角多样性与视图数 ≥ 3；可以开启 `--use-ransac` 提高单应鲁棒性。
- OpenCV 标志异常：不同构建的常量集合不同；本工程已做版本兼容处理（缺失常量将被忽略）。
- `ModuleNotFoundError: calib`：在命令前加 `PYTHONPATH=$(pwd)`，或用 `pip install -e .` 安装本地包。

---

## 参考与致谢

- Zhang, Z. (2000). A Flexible New Technique for Camera Calibration. IEEE TPAMI.
- OpenCV Calibration: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

