"""生成棋盘格图片的脚本

生成两类图片：
- `images/generated/`：标准正视无畸变棋盘图（top-down）
- `images/calib/`：对正视图做随机透视变换、模糊和噪声以模拟相机拍摄，用于标定测试

参数：
- pattern: 内角点数 cols rows（例如 9 6）
- square: 每个方格边长（像素）
- count: 生成图像数量
- out: images 根目录（默认 workspace 下的 images）
- max_perturb: 角点归一化扰动幅度（小数，相对于图像尺寸）
- seed: 随机种子（可复现）

示例：
python scripts/generate_chessboard.py --pattern 9 6 --square 50 --count 12
"""

import argparse
import os
import cv2
import numpy as np
import random


def make_chessboard(pattern, square_px, margin=0):
    cols, rows = pattern
    squares_x = cols + 1
    squares_y = rows + 1
    w = squares_x * square_px + 2*margin
    h = squares_y * square_px + 2*margin
    img = np.ones((h, w), dtype=np.uint8) * 255
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                x0 = margin + x * square_px
                y0 = margin + y * square_px
                x1 = x0 + square_px
                y1 = y0 + square_px
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), 0, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def random_perspective_warp(img, max_perturb=0.15, out_size=None):
    h, w = img.shape[:2]
    if out_size is None:
        out_w, out_h = w, h
    else:
        out_w, out_h = out_size
    # src corners
    src = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    # perturb destination corners inside a margin
    margin_x = max_perturb * w
    margin_y = max_perturb * h
    dst = src.copy().astype(np.float32)
    dst[:,0] += np.random.uniform(-margin_x, margin_x, size=(4,))
    dst[:,1] += np.random.uniform(-margin_y, margin_y, size=(4,))
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return warped


def add_noise_and_blur(img, blur_ksize=3, noise_sigma=5):
    if blur_ksize and blur_ksize > 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    if noise_sigma and noise_sigma > 0:
        noise = np.random.randn(*img.shape) * noise_sigma
        noisy = img.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    return img


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', nargs=2, type=int, required=True, help='内角点数 cols rows，例如 9 6')
    parser.add_argument('--square', type=int, default=50, help='每个方格像素大小，默认 50')
    parser.add_argument('--count', type=int, default=10, help='生成图片张数')
    parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'images'), help='images 根目录')
    parser.add_argument('--max-perturb', type=float, default=0.12, help='透视角点最大扰动相对幅度（相对于图像尺寸），默认 0.12')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（可复现）')
    parser.add_argument('--blur', type=int, default=3, help='模糊核大小（奇数），0 表示不模糊')
    parser.add_argument('--noise', type=float, default=4.0, help='噪声 sigma（像素），0 表示不加噪声')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    out_root = os.path.abspath(args.out)
    gen_dir = os.path.join(out_root, 'generated')
    calib_dir = os.path.join(out_root, 'calib')
    ensure_dir(gen_dir)
    ensure_dir(calib_dir)

    pattern = (args.pattern[0], args.pattern[1])
    square = int(args.square)

    print(f'生成 {args.count} 张棋盘图（pattern={pattern}, square={square}px）')
    for i in range(args.count):
        img = make_chessboard(pattern, square)
        fname_base = f'chess_{pattern[0]}x{pattern[1]}_{i:03d}.png'
        gen_path = os.path.join(gen_dir, fname_base)
        cv2.imwrite(gen_path, img)

        # 生成仿真相机拍摄图像（透视变形 + 模糊 + 噪声）
        warped = random_perspective_warp(img, max_perturb=args.max_perturb, out_size=None)
        warped = add_noise_and_blur(warped, blur_ksize=args.blur if args.blur>1 else 0, noise_sigma=args.noise)
        calib_path = os.path.join(calib_dir, fname_base)
        cv2.imwrite(calib_path, warped)
        print(f'  写入 {gen_path} 和 {calib_path}')

    print('完成。')

if __name__ == '__main__':
    main()
