"""
可视化数据集中棋盘角点的整体分布（单图聚合显示，散点模式）。

用法示例：
    python scripts/visualize_dataset_corners.py \
        --images 'images/11_15_2058/*.jpg' \
        --pattern 9 6 \
    --square 0.029 \
        --out out/corners_dataset_scatter.png

说明：
- 将所有图片的角点按归一化坐标绘制到同一画布。
- 颜色按图片分组，使用不同颜色区分不同图片的角点。
"""

import argparse
import glob
import os
import sys
import cv2
import numpy as np

# 允许从项目任意位置调用脚本
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calib.calibrate import detect_corners


def ensure_dir(p: str):
    if not os.path.exists(os.path.dirname(p)):
        os.makedirs(os.path.dirname(p), exist_ok=True)
# 为每张图片生成不同的颜色（BGR），使用均匀分布的 HSV 转换
def generate_distinct_colors(n: int) -> list:
    colors = []
    n = max(1, int(n))
    for i in range(n):
        # OpenCV HSV: H∈[0,179], S∈[0,255], V∈[0,255]
        h = int(round((i / n) * 179))
        hsv = np.uint8([[[h, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def build_parser():
    ap = argparse.ArgumentParser(description='聚合可视化数据集棋盘角点分布（散点，单图显示）', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--images', default=None, help='图片 glob 通配符，例如 /path/to/*.jpg；请使用引号避免 shell 展开')
    ap.add_argument('--image', dest='images', help='同 --images 的别名；建议用引号包裹通配符')
    ap.add_argument('--pattern', nargs=2, type=int, required=True, help='棋盘格内角点数：cols rows（例：9 6）')
    ap.add_argument('--square', type=float, required=True, help='棋盘格单元物理尺寸（任意单位，如 0.029）')
    ap.add_argument('--point-size', type=int, default=10, help='scatter 模式的点半径（像素）')
    ap.add_argument('--out', type=str, default='out/corners_dataset.png', help='输出文件路径（PNG）')
    return ap



def visualize_scatter(used_images, imgpoints_list, canvas_size: tuple, out_path: str, point_radius: int = 2):
    ensure_dir(out_path)
    W, H = int(canvas_size[0]), int(canvas_size[1])
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    # 画边框
    cv2.rectangle(canvas, (1,1), (W-2, H-2), (200,200,200), 2)
    # 聚合所有图片的角点，按归一化映射到统一画布
    total_pts = 0
    palette = generate_distinct_colors(len(used_images))
    for idx, (path, imgp) in enumerate(zip(used_images, imgpoints_list)):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        xs = imgp[:, 0] / max(w, 1)
        ys = imgp[:, 1] / max(h, 1)
        xs_vis = (xs * W).astype(np.int32)
        ys_vis = (ys * H).astype(np.int32)
        color = palette[idx % len(palette)]
        for x, y in zip(xs_vis, ys_vis):
            cv2.circle(canvas, (int(x), int(y)), point_radius, color, -1)
            total_pts += 1
    cv2.putText(canvas, f'total corners: {total_pts}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
    cv2.putText(canvas, 'normalized to dataset mean canvas', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.imwrite(out_path, canvas)


def main():
    ap = build_parser()
    args = ap.parse_args()

    if not args.images:
        print('需要提供 --images（或 --image 别名）')
        sys.exit(1)

    if args.pattern[0] < 2 or args.pattern[1] < 2:
        print(f'无效的 --pattern：{args.pattern}，列与行都应 >= 2（为“内角点”数量，例如 9 6）')
        sys.exit(1)
    if float(args.square) <= 0:
        print('无效的 --square：必须为正数（例如 0.029）')
        sys.exit(1)

    images = glob.glob(args.images)
    if not images:
        print('未找到匹配的图片：', args.images)
        sys.exit(1)

    pattern = (args.pattern[0], args.pattern[1])
    objpoints, imgpoints, used = detect_corners(images, pattern, float(args.square))
    if len(imgpoints) == 0:
        print('警告：未检测到任何角点，无法可视化')
        sys.exit(2)

    # 画布大小使用数据集中图片的原始尺寸（假设所有图片尺寸一致）
    sample_img = cv2.imread(used[0])
    if sample_img is None:
        print('无法读取示例图片确定画布尺寸：', used[0])
        sys.exit(1)
    H, W = sample_img.shape[:2]
    visualize_scatter(used, imgpoints, (W, H), args.out, point_radius=int(args.point_size))
    print('Scatter saved to', args.out)


if __name__ == '__main__':
    main()
