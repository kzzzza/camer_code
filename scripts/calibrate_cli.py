"""统一相机标定与评估 CLI（仅保留 all 模式）

在一次运行中依次执行 calibrate + evaluate，并统一打包输出

示例：
python scripts/calibrate_cli.py --images 'images/calib/*.png' --pattern 9 6 --square 0.029 --bundle run_name
"""

import argparse
import glob
import sys
import os
import datetime
import json
from typing import Optional

# 确保项目根加入 sys.path（支持从任意工作目录调用脚本）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import cv2

from calib.calibrate import calibrate_from_images, detect_corners


def build_flags(args):
    """根据命令行开关与当前 OpenCV 版本安全构造标定 flags。"""
    if cv2 is None:
        return 0
    flags = getattr(cv2, 'CALIB_USE_INTRINSIC_GUESS', 0)
    if getattr(args, 'free_skew', False) and hasattr(cv2, 'CALIB_FIX_SKEW') is False:
        # nothing to do if API missing; keep default
        pass
    elif not getattr(args, 'free_skew', False) and hasattr(cv2, 'CALIB_FIX_SKEW'):
        flags |= getattr(cv2, 'CALIB_FIX_SKEW')
    if not getattr(args, 'free_tangential', False):
        if hasattr(cv2, 'CALIB_ZERO_TANGENT_DIST'):
            flags |= getattr(cv2, 'CALIB_ZERO_TANGENT_DIST')
        elif hasattr(cv2, 'CALIB_FIX_TANGENT_DIST'):
            flags |= getattr(cv2, 'CALIB_FIX_TANGENT_DIST')
    if not getattr(args, 'enable_k3', False) and hasattr(cv2, 'CALIB_FIX_K3'):
        flags |= getattr(cv2, 'CALIB_FIX_K3')
    if getattr(args, 'fix_principal_point', False) and hasattr(cv2, 'CALIB_FIX_PRINCIPAL_POINT'):
        flags |= getattr(cv2, 'CALIB_FIX_PRINCIPAL_POINT')
    return flags


def build_parser():
    parser = argparse.ArgumentParser(description='统一相机标定/评估 CLI（仅 all 模式）', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 基础参数
    parser.add_argument('--images', default=None, help='图片 glob 通配符，例如 /path/to/*.jpg；请使用引号避免 shell 展开')
    parser.add_argument('--image', dest='images', help='同 --images 的别名；建议用引号包裹通配符')
    parser.add_argument('--pattern', nargs=2, type=int, default=None, help='棋盘格内角点数：cols rows（例：9 6）')
    parser.add_argument('--square', type=float, default=None, help='棋盘格单元物理尺寸（任意单位，如 0.029）')
    
    # 标定选项
    parser.add_argument('--use-ransac', action='store_true', help='单应估计使用 RANSAC（提高鲁棒性）')
    parser.add_argument('--free-skew', action='store_true', help='允许估计 skew（默认固定为 0）')
    parser.add_argument('--free-tangential', action='store_true', help='允许切向畸变 p1/p2（默认置零）')
    parser.add_argument('--enable-k3', action='store_true', help='允许三阶径向畸变 k3（默认固定为 0）')
    parser.add_argument('--fix-principal-point', action='store_true', help='固定主点（图像中心附近），默认不固定')
    parser.add_argument('--iter', type=int, default=100, help='联合优化最大迭代次数')
    parser.add_argument('--eps', type=float, default=1e-6, help='联合优化收敛阈值')

    # 评估/可视化选项
    parser.add_argument('--show-vis', action='store_true', help='在屏幕上显示可视化图像')
    parser.add_argument('--arrow-scale', type=float, default=1.0, help='重投影向量的缩放系数')
    parser.add_argument('--grid', nargs=2, type=int, default=[10, 8], help='空间误差统计网格大小（列 行），用于聚合所有图的误差分布')
    parser.add_argument('--per-image-heatmap', action='store_true', help='同时输出每张图片的误差热力图（输出到 bundle 目录）')
    parser.add_argument('--bundle', type=str, default=None, help='将输出统一打包到项目根 out/<name>/ 下（单一 JSON + 所有图片）')

    # 数据集划分选项
    parser.add_argument('--val-ratio', type=float, default=0.3, help='验证集占比（0~1），其余作为训练集，仅用训练集进行标定')
    parser.add_argument('--split-seed', type=int, default=42, help='数据划分随机种子（确保可复现的划分）')

    return parser


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def compute_reprojection_stats(K: np.ndarray, dist: np.ndarray, objpoints, imgpoints):
    """返回 (overall_rms, per_image_rms list, residuals list)。residual = observed - projected"""
    per_rms, residuals, all_sq = [], [], []
    dist_full = np.array([dist[0], dist[1], 0.0, 0.0, 0.0], dtype=float)
    for objp, imgp in zip(objpoints, imgpoints):
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist_full, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            rvec = np.zeros((3,1), dtype=float)
            tvec = np.zeros((3,1), dtype=float)
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist_full)
        proj = proj.reshape(-1, 2)
        res = imgp - proj
        residuals.append(res)
        sq = (res ** 2).sum(axis=1)
        all_sq.extend(sq.tolist())
        per_rms.append(float(np.sqrt(sq.mean())))
    overall = float(np.sqrt(np.mean(all_sq))) if len(all_sq) else 0.0
    return overall, per_rms, residuals


def visualize_residuals(used_images, imgpoints_list, residuals_list, out_dir: str, show: bool = False, arrow_scale: float = 1.0):
    ensure_dir(out_dir)
    mags_all = [np.linalg.norm(r, axis=1) for r in residuals_list]
    max_mag = max([m.max() if m.size else 0.0 for m in mags_all]) if mags_all else 0.0
    max_mag = max(max_mag, 1e-6)
    for path, imgp, res in zip(used_images, imgpoints_list, residuals_list):
        img = cv2.imread(path)
        if img is None:
            print(f'warning: cannot read {path}')
            continue
        vis = img.copy()
        mags = np.linalg.norm(res, axis=1)
        for (pt, r, mag) in zip(imgp, res, mags):
            px, py = int(round(pt[0])), int(round(pt[1]))
            proj_x, proj_y = int(round(px - r[0])), int(round(py - r[1]))
            t = min(1.0, mag / max_mag)
            color = (int(255 * t), 0, int(255 * (1 - t)))
            cv2.circle(vis, (proj_x, proj_y), 3, (255,255,255), -1)
            cv2.circle(vis, (px, py), 3, (0,255,0), -1)
            end_x = int(round(proj_x + r[0] * arrow_scale))
            end_y = int(round(proj_y + r[1] * arrow_scale))
            cv2.arrowedLine(vis, (proj_x, proj_y), (end_x, end_y), color, 1, tipLength=0.3)
        overall_rms_img = float(np.sqrt((mags ** 2).mean())) if mags.size>0 else 0.0
        cv2.putText(vis, f'RMS(px): {overall_rms_img:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        out_path = os.path.join(out_dir, os.path.basename(path))
        cv2.imwrite(out_path, vis)
        if show:
            cv2.imshow('reproj', vis)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                break
    if show:
        cv2.destroyAllWindows()


def load_gt_K(path: str) -> Optional[np.ndarray]:
    if path is None:
        return None
    if path.endswith('.npy'):
        return np.load(path)
    with open(path, 'r') as f:
        j = json.load(f)
    if isinstance(j, list):
        return np.array(j, dtype=float)
    if isinstance(j, dict):
        fx = float(j.get('fx', 0.0)); fy = float(j.get('fy', 0.0))
        cx = float(j.get('cx', 0.0)); cy = float(j.get('cy', 0.0))
        skew = float(j.get('skew', 0.0))
        K = np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
        return K
    return None


# ------------- 空间误差分布与可视化 -------------
def compute_spatial_heatmap(used_images, imgpoints_list, residuals_list, bins_x: int, bins_y: int):
    """聚合所有图片的重投影误差，计算在归一化图像坐标上的空间 RMS 分布。

    对每个有效角点，计算 residual 的幅值 mag，归一化坐标 (x/w, y/h) 落入网格，累计 mag^2 与计数。
    返回：rms_grid (bins_y, bins_x)、counts (bins_y, bins_x)
    """
    bins_x = max(1, int(bins_x)); bins_y = max(1, int(bins_y))
    sum_sq = np.zeros((bins_y, bins_x), dtype=np.float64)
    counts = np.zeros((bins_y, bins_x), dtype=np.int32)
    for path, imgp, res in zip(used_images, imgpoints_list, residuals_list):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        mags = np.linalg.norm(res, axis=1)
        xs = imgp[:, 0] / max(w, 1)
        ys = imgp[:, 1] / max(h, 1)
        gx = np.clip((xs * bins_x).astype(np.int32), 0, bins_x - 1)
        gy = np.clip((ys * bins_y).astype(np.int32), 0, bins_y - 1)
        for xbin, ybin, m in zip(gx, gy, mags):
            sum_sq[ybin, xbin] += float(m * m)
            counts[ybin, xbin] += 1
    rms = np.zeros_like(sum_sq, dtype=np.float64)
    nonzero = counts > 0
    rms[nonzero] = np.sqrt(sum_sq[nonzero] / counts[nonzero])
    return rms, counts


def visualize_heatmap(rms_grid: np.ndarray, counts: np.ndarray, out_path: str, title: str = 'RMS heatmap'):
    """将 RMS 网格可视化为色图 PNG。零计数的网格用黑色显示。"""
    ensure_dir(os.path.dirname(out_path))
    hbins, wbins = rms_grid.shape
    vmax = float(np.max(rms_grid))
    vmin = float(np.min(rms_grid[counts > 0])) if np.any(counts > 0) else 0.0
    denom = max(vmax - vmin, 1e-12)
    norm = np.zeros_like(rms_grid, dtype=np.float32)
    if denom > 0:
        norm[counts > 0] = (rms_grid[counts > 0] - vmin) / denom
    img_small = (norm * 255.0).astype(np.uint8)
    scale = 40  # 每格像素大小
    heat = cv2.resize(img_small, (wbins * scale, hbins * scale), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    counts_big = cv2.resize((counts > 0).astype(np.uint8), (wbins * scale, hbins * scale), interpolation=cv2.INTER_NEAREST)
    mask_zero = counts_big == 0
    heat[mask_zero] = (0, 0, 0)
    cv2.putText(heat, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(heat, f'RMS range:[{vmin:.2f}, {vmax:.2f}] px', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imwrite(out_path, heat)


def evaluate_set(set_name: str,
                 images_set,
                 K_est: np.ndarray,
                 dist_est: np.ndarray,
                 bins_x: int,
                 bins_y: int,
                 out_dir: str,
                 show_vis: bool,
                 arrow_scale: float,
                 per_image_heatmap: bool):
    """对一个数据子集（train 或 val）进行评估与可视化，并返回指标与输出文件列表。"""
    ensure_dir(out_dir)
    print(f"\n=== Evaluate [{set_name}] ===")
    objpoints, imgpoints, used = detect_corners(images_set, pattern=None, square=None)
    # 上述调用需要 pattern 与 square，但当前 calib.detect_corners 签名是 (images, pattern, square)
    # 因此我们在此处不传 None，而在主流程中以闭包变量注入。为避免困扰，实际实现见主流程。
    # 此占位函数仅用于结构说明。
    return {}


# ---------------------- 单一模式实现（all） ----------------------
def cmd_all(args):
    # 1) 校验与准备
    if not args.images:
        print('all 模式需要提供 --images')
        return 1
    if args.pattern is None or args.square is None:
        print('缺少必需参数：--pattern 与 --square')
        return 1
    images = glob.glob(args.images)
    if not images:
        print('未找到匹配的图片：', args.images)
        return 1
    pattern = (args.pattern[0], args.pattern[1])
    flags = build_flags(args)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(args.iter), float(args.eps))
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    opencv_version = cv2.__version__ if cv2 is not None else 'N/A'

    # 统一打包目录
    if not args.bundle:
        print('提示：all 模式建议提供 --bundle 名称，用于统一输出到 out/<name>/')
    bundle_dir = os.path.abspath(os.path.join(ROOT, 'out', args.bundle or 'run_all'))
    ensure_dir(bundle_dir)

    # 2) 数据集划分（train/val）
    val_ratio = float(args.val_ratio)
    val_ratio = min(max(val_ratio, 0.0), 0.9)  # 防止全部进验证集
    rng = np.random.default_rng(int(args.split_seed))
    images_shuffled = images[:]
    rng.shuffle(images_shuffled)
    n_total = len(images_shuffled)
    n_val = int(np.floor(n_total * val_ratio))
    n_val = min(max(n_val, 1), n_total - 3)  # 至少预留 3 张给训练
    val_images = images_shuffled[:n_val]
    train_images = images_shuffled[n_val:]

    print(f"\n=== [Split] total={n_total}, train={len(train_images)}, val={len(val_images)}, ratio={val_ratio:.2f}, seed={int(args.split_seed)} ===")

    # 3) calibrate（仅使用训练集）
    print('\n=== [1/3] Calibrate (Train) ===')
    try:
        calib_res = calibrate_from_images(train_images, pattern, args.square, use_ransac=args.use_ransac, calib_flags=flags, criteria=criteria)
    except Exception as e:
        print('标定失败：', repr(e))
        return 2
    print('训练集使用图片数量：', len(calib_res['used_images']))
    print('K (优化)：\n', calib_res['K'])
    print('dist (k1,k2)：', calib_res['dist'])
    print('训练 RMS (calibrateCamera)：', calib_res.get('rms', 'N/A'))

    K_est, dist_est = calib_res['K'], calib_res['dist']
    bins_x, bins_y = int(args.grid[0]), int(args.grid[1])

    # 4) evaluate（分别对训练集与验证集）
    print('\n=== [2/3] Evaluate (Train) ===')
    obj_tr, img_tr, used_tr = detect_corners(train_images, pattern, args.square)
    if len(obj_tr) < 3:
        print('训练集有效图片不足 3 张，无法可靠评估（当前: %d）' % len(obj_tr))
        return 2
    overall_rms_tr, per_rms_tr, residuals_tr = compute_reprojection_stats(K_est, dist_est, obj_tr, img_tr)
    rms_grid_tr, counts_grid_tr = compute_spatial_heatmap(used_tr, img_tr, residuals_tr, bins_x, bins_y)

    train_dir = os.path.join(bundle_dir, 'train')
    ensure_dir(train_dir)
    print('\nVisualizing residuals (train) to', train_dir)
    visualize_residuals(used_tr, img_tr, residuals_tr, train_dir, show=args.show_vis, arrow_scale=args.arrow_scale)
    heat_train = os.path.join(train_dir, 'rms_heatmap.png')
    visualize_heatmap(rms_grid_tr, counts_grid_tr, heat_train, title=f'Train RMS heatmap ({bins_x}x{bins_y})')
    eval_files_tr = [os.path.join(train_dir, os.path.basename(p)) for p in used_tr]
    eval_files_tr.append(heat_train)
    if args.per_image_heatmap:
        for path, pts, residual in zip(used_tr, img_tr, residuals_tr):
            rms_i, counts_i = compute_spatial_heatmap([path], [pts], [residual], bins_x, bins_y)
            fname = os.path.splitext(os.path.basename(path))[0]
            out_i = os.path.join(train_dir, f'{fname}_heatmap.png')
            visualize_heatmap(rms_i, counts_i, out_i, title=f'Train RMS {fname}')
            eval_files_tr.append(out_i)

    print('\n=== [3/3] Evaluate (Val) ===')
    obj_val, img_val, used_val = detect_corners(val_images, pattern, args.square)
    if len(obj_val) < 1:
        print('验证集角点检测失败或数量为 0（当前: %d）' % len(obj_val))
        # 允许继续，但标记为空评估
    overall_rms_val, per_rms_val, residuals_val = (0.0, [], []) if len(obj_val)==0 else compute_reprojection_stats(K_est, dist_est, obj_val, img_val)
    rms_grid_val, counts_grid_val = (np.zeros((bins_y, bins_x)), np.zeros((bins_y, bins_x), dtype=np.int32)) if len(obj_val)==0 else compute_spatial_heatmap(used_val, img_val, residuals_val, bins_x, bins_y)

    val_dir = os.path.join(bundle_dir, 'val')
    ensure_dir(val_dir)
    print('\nVisualizing residuals (val) to', val_dir)
    if len(obj_val) > 0:
        visualize_residuals(used_val, img_val, residuals_val, val_dir, show=args.show_vis, arrow_scale=args.arrow_scale)
    heat_val = os.path.join(val_dir, 'rms_heatmap.png')
    visualize_heatmap(rms_grid_val, counts_grid_val, heat_val, title=f'Val RMS heatmap ({bins_x}x{bins_y})')
    eval_files_val = [] if len(obj_val)==0 else [os.path.join(val_dir, os.path.basename(p)) for p in used_val]
    eval_files_val.append(heat_val)
    if args.per_image_heatmap and len(obj_val)>0:
        for path, pts, residual in zip(used_val, img_val, residuals_val):
            rms_i, counts_i = compute_spatial_heatmap([path], [pts], [residual], bins_x, bins_y)
            fname = os.path.splitext(os.path.basename(path))[0]
            out_i = os.path.join(val_dir, f'{fname}_heatmap.png')
            visualize_heatmap(rms_i, counts_i, out_i, title=f'Val RMS {fname}')
            eval_files_val.append(out_i)


    # 5) 综合 JSON 输出（根目录 + 子目录）
    combined = {
        'timestamp': timestamp,
        'opencv_version': opencv_version,
        'images_glob': args.images,
        'matched_image_count': len(images),
        'pattern': [pattern[0], pattern[1]],
        'square': float(args.square),
        'use_ransac': bool(args.use_ransac),
        'split': {
            'val_ratio': float(val_ratio),
            'seed': int(args.split_seed),
            'counts': {'total': n_total, 'train': len(train_images), 'val': len(val_images)}
        },
        'calibrate': {
            'used_image_count': len(calib_res['used_images']),
            'used_images': calib_res['used_images'],
            'K_init': calib_res['K_init'].tolist(),
            'K': calib_res['K'].tolist(),
            'dist': calib_res['dist'].tolist(),
            'calibrate_rms': float(calib_res.get('rms', float('nan')))
        },
        'train_evaluate': {
            'used_image_count': len(used_tr),
            'used_images': used_tr,
            'overall_reprojection_rms_px': float(overall_rms_tr),
            'per_image_rms_px': per_rms_tr,
            'spatial_distribution': {
                'rms_grid': rms_grid_tr.tolist(),
                'counts_grid': counts_grid_tr.tolist()
            },
            'visualization_dir': train_dir,
            'visualization_files': eval_files_tr
        },
        'val_evaluate': {
            'used_image_count': len(used_val),
            'used_images': used_val,
            'overall_reprojection_rms_px': float(overall_rms_val),
            'per_image_rms_px': per_rms_val,
            'spatial_distribution': {
                'rms_grid': rms_grid_val.tolist(),
                'counts_grid': counts_grid_val.tolist()
            },
            'visualization_dir': val_dir,
            'visualization_files': eval_files_val
        },
    }
    out_json = os.path.join(bundle_dir, 'results_all.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print('\n[All] Bundle saved to', bundle_dir)
    print('[All] JSON:', out_json)

    # 写子目录 JSON
    train_json = {
        'subset': 'train',
        'images': train_images,
        'evaluate': combined['train_evaluate']
    }
    with open(os.path.join(train_dir, 'results_train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_json, f, ensure_ascii=False, indent=2)
    val_json = {
        'subset': 'val',
        'images': val_images,
        'evaluate': combined['val_evaluate']
    }
    with open(os.path.join(val_dir, 'results_val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_json, f, ensure_ascii=False, indent=2)
    return 0



def main():
    parser = build_parser()
    args = parser.parse_args()
    code = cmd_all(args)
    sys.exit(code)


if __name__ == '__main__':
    main()
