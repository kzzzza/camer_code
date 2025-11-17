# Zhang 相机标定（NJU数字图像处理与计算机视觉大作业231180007）


这是一个基于张友定方法 (Zhang, 2000) 的灵活相机标定工程。核心流程：
- 检测棋盘角点 → 估计单应矩阵（可选 RANSAC）→ 张氏线性内参初值 → OpenCV `calibrateCamera` 联合优化
- 提供生成标定板、标定、可视化、评估脚本，便于做参数对比实验（skew/切向畸变/k3、RANSAC、迭代准则等）


---

## 工程代码目录结构

- `calib/` — 标定库实现
  - `calib/calibrate.py` — 核心：角点检测、单应估计、Zhang 线性解、外参恢复、联合优化（cv2.calibrateCamera）
- `scripts/` — 命令行脚本
  - `scripts/calibrate_cli.py` — 运行标定（读取图片 glob、参数/自由度开关）
  - `scripts/evaluate_intrinsics.py` — 评估重投影 RMS、误差可视化、可与 GT 对比
  - `scripts/visualize_corners.py` — 检测并绘制棋盘角点
  - `scripts/generate_chessboard.py` — 生成标准棋盘与仿真拍摄图
- `images/` — 示例/输出目录（脚本会在此写入）
  - `images/generated/` — 标准正视棋盘图（无畸变，top-down）
  - `images/visualized/` — 可视化输出（角点、重投影残差）
- `runs/` — 实验日志存放目录
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

以下命令需在项目根执行，或显式设置 `PYTHONPATH=/path/to/project_root`。

### a) 标定：`scripts/calibrate_cli.py`

示例（RANSAC + 默认稳健设置）：

```bash
PYTHONPATH=$(pwd) python scripts/calibrate_cli.py \
  --images 'images/11_15_2058/*.jpg' \
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
| `--log-json` | 路径=None | 将“实验条件+结果”输出为 JSON；传 `-` 则输出到标准输出 |

输出：初始/优化 `K`、`dist(k1,k2)`、使用图像数、整体 RMS（px）等。

注意：若出现 `ModuleNotFoundError: calib`，使用上面的 `PYTHONPATH=$(pwd)` 或执行 `pip install -e .`。

#### OpenCV 标定 flags 说明与 CLI 对应

| OpenCV flag | 作用 | 对优化变量的影响 | 典型使用场景 | CLI 对应 |
|---|---|---|---|---|
| `CALIB_USE_INTRINSIC_GUESS` | 使用你提供的初始内参/畸变作为初值 | 从初值开始优化 | 你已有合理初值（如 Zhang 线性解） | 默认启用 |
| `CALIB_FIX_SKEW` | 固定 skew 不优化 | `s` 固定（通常为 0） | 大多数相机像素近似正交 | 取消此固定用 `--free-skew` |
| `CALIB_ZERO_TANGENT_DIST` | 切向畸变置零且固定 | `p1=p2=0`，不优化 | 装配良好、可忽略切向畸变 | 取消置零用 `--free-tangential` |
| `CALIB_FIX_TANGENT_DIST` | 固定切向畸变为“输入值” | `p1,p2` 不变 | 你想保留非零的已知 `p1/p2` | 当 ZERO 不可用时回退使用（版本兼容） |
| `CALIB_FIX_K3` | 固定三阶径向畸变不优化 | `k3` 固定（通常 0） | 畸变不大/数据不足时避免过拟合 | 取消此固定用 `--enable-k3` |
| `CALIB_FIX_PRINCIPAL_POINT` | 固定主点不优化 | `(cx,cy)` 固定 | 视角不丰富或主点已知更可信 | 开启 `--fix-principal-point` |

提示：`FIX_xxx` 的语义是“保持输入值不变”，如果使用它们，请确保输入初值就是希望固定的数值。

使用条件：
- 放开 `k3`（`--enable-k3`）用于广角/鱼眼且视图充足；
- 放开切向（`--free-tangential`）在装配偏心可疑或残差呈切向趋势时；
- 固定主点（`--fix-principal-point`）在视角不足时避免漂移；
- `skew` 一般不建议放开。

#### 实验日志输出

运行时会打印“标定实验条件”区块，便于复现实验：

- 时间戳、OpenCV 版本
- 图片 glob 与匹配到的张数
- Pattern（cols x rows）、方格物理尺寸 `square`
- 是否使用 RANSAC
- 自由度开关：`free-skew`、`free-tangential`、`enable-k3`、`fix-principal-point`
- 终止准则：最大迭代 `--iter`、收敛阈值 `--eps`
- 解析后的 OpenCV flags 名称列表（版本兼容，缺失常量会被忽略）

#### 结构化 JSON 日志

你可以把本次实验条件与标定结果保存为 JSON：

```bash
# 保存到文件
PYTHONPATH=$(pwd) python scripts/calibrate_cli.py \
  --images 'images/11_15_2058/*.jpg' \
  --pattern 9 6 \
  --square 0.029 \
  --use-ransac \
  --log-json runs/2025-11-17_calib_exp.json

# 输出到标准输出（可重定向）
PYTHONPATH=$(pwd) python scripts/calibrate_cli.py \
  --images 'images/11_15_2058/*.jpg' \
  --pattern 9 6 \
  --square 0.029 \
  --log-json - > runs/stdout_log.json
```

JSON 字段包含：时间戳、OpenCV 版本、图片 glob、匹配/使用张数、`used_images` 列表、`pattern`、`square`、各自由度开关、终止准则、解析后的 OpenCV flags（名称与数值）、以及 `K_init`、`K`、`dist`、`rms` 等结果。

---

### b) 评估与误差可视化：`scripts/evaluate_intrinsics.py`

示例（保存 JSON 报告与可视化）：

```bash
PYTHONPATH=$(pwd) python scripts/evaluate_intrinsics.py \
  --images 'images/11_15_2058/*.jpg' \
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
| `--log-json` | 路径=None | 将“评估条件+结果”输出为 JSON；传 `-` 则输出到标准输出 |

输出：总体 RMS、每图 RMS、可选的 GT 对比、可视化残差图。

示例（保存评估 JSON 日志，便于与标定 JSON 一起归档）：

```bash
PYTHONPATH=$(pwd) python scripts/evaluate_intrinsics.py \
  --images 'images/11_15_2058/*.jpg' \
  --pattern 9 6 \
  --square 0.029 \
  --use-ransac \
  --log-json runs/2025-11-17_eval.json
```

评估 JSON 包含：时间戳、OpenCV 版本、图片 glob、匹配/使用张数、可视化设置、从标定得到的 `K_init/K/dist/calibrate_rms`、评估得到的 `overall_rms/per_image_rms`，以及（若提供）与 GT 的差异指标。

---

### c) 角点可视化：`scripts/visualize_corners.py`

示例：

```bash
PYTHONPATH=$(pwd) python scripts/visualize_corners.py \
  --images 'images/11_15_2058/*.jpg' \
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

### d) 生成可打印多变体棋盘板：`scripts/generate_chessboard.py`

新版生成脚本只输出“可打印的棋盘标定板”及其多种质量/外观变体，不再直接生成模拟拍照图。你可以打印这些图，再用真实相机拍摄做标定，以测试不同板质量对标定精度的影响。

示例（生成标准+模糊+透视+噪声+对比度变体，每种 5 张）：

```bash
python scripts/generate_chessboard.py \
  --pattern 9 6 \
  --square 50 \
  --count 5 \
  --variants standard,blur,perspective,noise,contrast \
  --perspective-tilt 0.07 \
  --noise-sigma 6 \
  --blur-ksize 9 \
  --affine-rotate 5 \
  --seed 42
```

参数表：

| 选项 | 类型/默认 | 说明 |
|---|---|---|
| `--pattern` | int int, 必填 | 内角点数：cols rows |
| `--square` | int=50 | 每格像素大小（像素单位） |
| `--count` | int=5 | 针对随机变体生成的张数（standard 等也会按此数生成） |
| `--variants` | str=全部 | 要生成的变体集合，逗号分隔；可选: standard,blur,affine,perspective,contrast,inverted,noise |
| `--out-root` | 路径=images/boards | 输出根目录（下设各变体子目录） |
| `--seed` | int=None | 随机种子（可复现） |
| `--blur-ksize` | int=7 | 模糊核大小（奇数；变体 blur） |
| `--blur-sigma` | float=0.0 | 模糊 sigma，0 表示自动 |
| `--affine-rotate` | float=5.0 | 仿射旋转角度（度；变体 affine） |
| `--affine-scale` | float=1.0 | 仿射缩放系数（变体 affine） |
| `--perspective-tilt` | float=0.06 | 透视倾斜扰动比例 (0~1；变体 perspective) |
| `--contrast-alpha` | float=1.25 | 对比度因子（变体 contrast） |
| `--contrast-beta` | float=10.0 | 亮度偏移（变体 contrast） |
| `--gamma` | float=1.0 | 伽马校正（变体 contrast） |
| `--noise-sigma` | float=5.0 | 高斯噪声强度（变体 noise） |

输出结构：

```
images/boards/standard/chess_9x6_000.png
images/boards/blur/chess_9x6_000.png
images/boards/perspective/chess_9x6_002.png
...
```

变体说明：
- standard  : 高对比黑白棋盘
- blur      : 模拟打印墨迹或摄像头轻微离焦
- affine    : 小角度旋转/缩放（测试边缘对齐不完美）
- perspective: 轻微透视倾斜（贴合或打印裁切不平行）
- contrast  : 不同对比度/亮度/伽马（打印设备差异）
- inverted  : 黑白反转（测试鲁棒性）
- noise     : 叠加纹理噪声（打印纸面污点/纹理）

建议：打印时保持高分辨率（避免额外插值），确保物理方格尺寸与 `--square` 对应的单位换算记录在旁，以便后续标定时使用准确物理尺寸。

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

## 参考

- Zhang, Z. (2000). A Flexible New Technique for Camera Calibration. IEEE TPAMI.
- OpenCV Calibration: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

