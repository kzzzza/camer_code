"""生成用于打印的多种校准棋盘板（calibration board）图像。

原版本会生成模拟拍照图（透视+噪声）供直接标定，本需求修改为只生成可打印的基准棋盘及其多种“质量退化”变体：

生成的变体（可通过 --variants 选择，逗号分隔）：
1. standard      : 标准高对比黑白棋盘
2. blur          : 高斯模糊版本（模拟打印墨迹扩散或轻微失焦）
3. affine        : 轻微旋转+缩放后的棋盘（保持画布尺寸）
4. perspective   : 轻微透视倾斜（模拟打印裁切或贴合时倾斜）
5. contrast      : 调整对比度/亮度/伽马（模拟打印设备差异）
6. inverted      : 颜色反转（黑白互换，可测试算法鲁棒性）
7. noise         : 叠加轻微高斯噪声（模拟打印纹理/污点）

CLI 关键参数：
--pattern cols rows      棋盘内角点数（例如 9 6）
--square  px             单个方格像素尺寸（打印时决定物理尺寸换算）
--count   n              需要随机化的变体生成次数（对 perspective / noise / affine 有效）
--variants list          需要生成的变体集合（默认全部）
--out-root path          输出根目录（默认 ./images/boards）

各变体附加参数：
--blur-ksize, --blur-sigma
--affine-rotate (度), --affine-scale
--perspective-tilt (0~1)
--contrast-alpha, --contrast-beta, --gamma
--noise-sigma

示例：
python scripts/generate_chessboard.py \
    --pattern 9 6 --square 50 --count 5 \
    --variants standard,blur,perspective,noise,contrast \
    --affine-rotate 5 --perspective-tilt 0.07 --noise-sigma 6

输出结构：
images/boards/<variant>/chess_9x6_<index>.png

注意：请保证打印分辨率足够，避免因缩放产生再采样伪影；打印后再使用真实相机拍摄进行标定实验，以测试不同板质量对标定精度的影响。
"""

import argparse
import os
import cv2
import numpy as np
import random
import math


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


def perspective_tilt(img, tilt=0.05):
    """对图像施加轻微透视倾斜。tilt 表示边缘点最大归一化偏移比例 (0~1)。"""
    h, w = img.shape[:2]
    src = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    dx = tilt * w
    dy = tilt * h
    # 轻微上下或左右不对称扰动
    dst = src.copy()
    dst[0] += np.array([ random.uniform(-dx, dx), random.uniform(-dy, dy) ])
    dst[1] += np.array([ random.uniform(-dx, dx), random.uniform(-dy, dy) ])
    dst[2] += np.array([ random.uniform(-dx, dx), random.uniform(-dy, dy) ])
    dst[3] += np.array([ random.uniform(-dx, dx), random.uniform(-dy, dy) ])
    M = cv2.getPerspectiveTransform(dst, src)  # 使棋盘保持充满画布（或用 src->dst 直接倾斜）
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return warped


def apply_blur(img, ksize=5, sigma=0):
    if ksize and ksize > 1:
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img

def apply_affine(img, rotate_deg=5.0, scale=1.0):
    h, w = img.shape[:2]
    center = (w/2.0, h/2.0)
    M = cv2.getRotationMatrix2D(center, rotate_deg, scale)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return out

def apply_contrast(img, alpha=1.2, beta=10.0, gamma=1.0):
    """alpha 对比度，beta 亮度偏移，gamma 伽马。"""
    tmp = img.astype(np.float32) * alpha + beta
    tmp = np.clip(tmp, 0, 255).astype(np.uint8)
    if gamma != 1.0:
        inv = 1.0 / max(gamma, 1e-6)
        lut = np.array([((i/255.0)**inv)*255.0 for i in range(256)], dtype=np.float32)
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        tmp = cv2.LUT(tmp, lut)
    return tmp

def apply_noise(img, sigma=5.0):
    if sigma <= 0:
        return img
    noise = np.random.randn(*img.shape) * sigma
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='生成多种可打印的棋盘标定板图像')
    parser.add_argument('--pattern', nargs=2, type=int, required=True, help='内角点数 cols rows，例如 9 6')
    parser.add_argument('--square', type=int, default=50, help='每个方格像素大小，默认 50')
    parser.add_argument('--count', type=int, default=5, help='针对随机变体（perspective/affine/noise）生成的张数')
    parser.add_argument('--variants', type=str, default='standard,blur,affine,perspective,contrast,inverted,noise', help='生成的变体集合，逗号分隔')
    parser.add_argument('--out-root', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'images', 'boards'), help='输出根目录')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（可复现）')
    # 变体参数
    parser.add_argument('--blur-ksize', type=int, default=7, help='高斯模糊核大小（奇数）')
    parser.add_argument('--blur-sigma', type=float, default=0.0, help='高斯模糊 sigma (0=自动)')
    parser.add_argument('--affine-rotate', type=float, default=5.0, help='仿射旋转角度（度）')
    parser.add_argument('--affine-scale', type=float, default=1.0, help='仿射缩放系数')
    parser.add_argument('--perspective-tilt', type=float, default=0.06, help='透视倾斜扰动比例 (0~1)')
    parser.add_argument('--contrast-alpha', type=float, default=1.25, help='对比度 alpha')
    parser.add_argument('--contrast-beta', type=float, default=10.0, help='亮度 beta')
    parser.add_argument('--gamma', type=float, default=1.0, help='伽马校正 gamma')
    parser.add_argument('--noise-sigma', type=float, default=5.0, help='高斯噪声 sigma（像素）')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    pattern = (args.pattern[0], args.pattern[1])
    square = int(args.square)
    variants = [v.strip().lower() for v in args.variants.split(',') if v.strip()]
    out_root = os.path.abspath(args.out_root)
    ensure_dir(out_root)

    base = make_chessboard(pattern, square)

    print(f'生成棋盘标定板变体 (pattern={pattern}, square={square}px)：{variants}')

    def save_variant(img, variant_name, idx):
        dir_path = os.path.join(out_root, variant_name)
        ensure_dir(dir_path)
        fname = f'chess_{pattern[0]}x{pattern[1]}_{idx:03d}.png'
        path = os.path.join(dir_path, fname)
        cv2.imwrite(path, img)
        return path

    generated_paths = []

    # standard：单张即可（仍生成 count 张以保持文件数量一致可对比）
    if 'standard' in variants:
        for i in range(args.count):
            generated_paths.append(save_variant(base, 'standard', i))

    if 'blur' in variants:
        for i in range(args.count):
            img_blur = apply_blur(base, ksize=args.blur_ksize, sigma=args.blur_sigma)
            generated_paths.append(save_variant(img_blur, 'blur', i))

    if 'affine' in variants:
        for i in range(args.count):
            # 为每张加一点随机扰动在旋转角度
            rot = args.affine_rotate + random.uniform(-1.0, 1.0)
            scl = args.affine_scale * (1.0 + random.uniform(-0.01, 0.01))
            img_aff = apply_affine(base, rotate_deg=rot, scale=scl)
            generated_paths.append(save_variant(img_aff, 'affine', i))

    if 'perspective' in variants:
        for i in range(args.count):
            tilt = args.perspective_tilt * (1.0 + random.uniform(-0.2, 0.2))
            img_persp = perspective_tilt(base, tilt=tilt)
            generated_paths.append(save_variant(img_persp, 'perspective', i))

    if 'contrast' in variants:
        for i in range(args.count):
            img_con = apply_contrast(base, alpha=args.contrast_alpha, beta=args.contrast_beta, gamma=args.gamma)
            generated_paths.append(save_variant(img_con, 'contrast', i))

    if 'inverted' in variants:
        for i in range(args.count):
            img_inv = 255 - base
            generated_paths.append(save_variant(img_inv, 'inverted', i))

    if 'noise' in variants:
        for i in range(args.count):
            sigma = args.noise_sigma * (1.0 + random.uniform(-0.2, 0.2))
            img_n = apply_noise(base, sigma=sigma)
            generated_paths.append(save_variant(img_n, 'noise', i))

    print(f'共写入 {len(generated_paths)} 张变体图像，根目录：{out_root}')
    print('示例路径（前 5 条）：')
    for p in generated_paths[:5]:
        print('  -', p)
    print('完成。')

if __name__ == '__main__':
    main()
