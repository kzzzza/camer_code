"""检测并可视化棋盘格角点。

脚本功能：
- 使用项目内的 `detect_corners` 来检测指定图片集合中的棋盘角点
- 在每张图片上绘制角点并保存到 `images/visualized/`
- 可选择在屏幕上显示图片

示例：
python scripts/visualize_corners.py --images 'images/calib/*.png' --pattern 9 6 --out images/visualized --show
"""

import argparse
import glob
import os
import sys

# 确保能导入 calib 包（当直接运行脚本时）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calib.calibrate import detect_corners
import cv2
import numpy as np


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def visualize(images_glob, pattern, square, out_dir, show=False, flags=None):
    images = glob.glob(images_glob)
    if not images:
        print('未找到匹配的图片')
        return
    ensure_dir(out_dir)
    # use detect_corners to get which images have corners and their points
    objpoints, imgpoints, used = detect_corners(images, pattern, square, flags=flags if flags is not None else cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # build mapping from path -> imgpoints
    mapping = {p: pts for p, pts in zip(used, imgpoints)}

    for p in images:
        img = cv2.imread(p)
        if img is None:
            print(f'warning: cannot read {p}, skipping')
            continue
        if p in mapping:
            pts = mapping[p]
            # cv2.drawChessboardCorners expects corners in shape (N,1,2) and a boolean found flag
            corners = pts.reshape(-1, 1, 2).astype(np.float32)
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (pattern[0], pattern[1]), corners, True)
            # also draw indices for clarity
            for i, (x, y) in enumerate(pts):
                cv2.putText(vis, str(i), (int(x)+3, int(y)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
        else:
            vis = img.copy()
            cv2.putText(vis, 'chessboard not found', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        fname = os.path.basename(p)
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, vis)
        print(f'wrote {out_path}')
        if show:
            cv2.imshow('corners', vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
    if show:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='图片通配符，例如 images/calib/*.png')
    parser.add_argument('--pattern', nargs=2, type=int, required=True, help='棋盘内角点数 cols rows')
    parser.add_argument('--square', type=float, default=1.0, help='方格物理尺寸（用于生成 objpoints，单位随意）')
    parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'images', 'visualized'), help='可视化输出目录')
    parser.add_argument('--show', action='store_true', help='在屏幕上显示每张可视化图片')
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    visualize(args.images, (args.pattern[0], args.pattern[1]), args.square, out, show=args.show)


if __name__ == '__main__':
    main()
