"""Calibrate on a training subset, evaluate on a held-out test subset.

Usage example:
PYTHONPATH=$(pwd) python scripts/calibrate_split_eval.py \
  --train-images 'images/train/*.png' \
  --test-images 'images/test/*.png' \
  --pattern 9 6 \
  --square 0.029 \
  --use-ransac \
  --enable-k3 --free-tangential \
  --iter 150 --eps 1e-6 \
  --log-json runs/2025-11-18_split.json \
  --vis-out images/eval_visualized_split
"""

import argparse
import glob
import os
import sys
import json
import datetime

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import cv2
from calib.calibrate import calibrate_from_images, detect_corners


def build_flags(args):
    import cv2
    flags = getattr(cv2, 'CALIB_USE_INTRINSIC_GUESS', 0)
    if not args.free_skew and hasattr(cv2, 'CALIB_FIX_SKEW'):
        flags |= getattr(cv2, 'CALIB_FIX_SKEW')
    if not args.free_tangential:
        if hasattr(cv2, 'CALIB_ZERO_TANGENT_DIST'):
            flags |= getattr(cv2, 'CALIB_ZERO_TANGENT_DIST')
        elif hasattr(cv2, 'CALIB_FIX_TANGENT_DIST'):
            flags |= getattr(cv2, 'CALIB_FIX_TANGENT_DIST')
    if not args.enable_k3 and hasattr(cv2, 'CALIB_FIX_K3'):
        flags |= getattr(cv2, 'CALIB_FIX_K3')
    if args.fix_principal_point and hasattr(cv2, 'CALIB_FIX_PRINCIPAL_POINT'):
        flags |= getattr(cv2, 'CALIB_FIX_PRINCIPAL_POINT')
    return flags


def compute_reprojection_stats(K: np.ndarray, dist: np.ndarray, objpoints, imgpoints):
    """Return (overall_rms, per_image_rms, residuals_list).
    residual = observed - projected
    """
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


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def visualize_residuals(used_images, imgpoints_list, residuals_list, out_dir: str, show: bool = False, arrow_scale: float = 1.0):
    ensure_dir(out_dir)
    # compute max magnitude for coloring
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
            proj_x, proj_y = int(round(pt[0] - r[0])), int(round(pt[1] - r[1]))
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
        print('wrote', out_path)
        if show:
            cv2.imshow('reproj', vis)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                break
    if show:
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description='Train/Test split calibration and evaluation')
    # 两种用法：
    # 1) 提供 --train-images 与 --test-images
    # 2) 仅提供 --images 与 --split 比例，脚本内部随机划分
    p.add_argument('--train-images', required=False, default=None, help="训练集图片 glob（可选，若使用 --images+--split 则无需提供）")
    p.add_argument('--test-images', required=False, default=None, help="测试集图片 glob（可选，若使用 --images+--split 则无需提供）")
    p.add_argument('--images', required=False, default=None, help='单一图片 glob（与 --split 配合随机划分 train/test）')
    p.add_argument('--split', type=float, default=None, help='训练集比例 (0~1)，仅在使用 --images 时有效，如 0.7')
    p.add_argument('--split-seed', type=int, default=None, help='随机划分种子（可复现）')
    p.add_argument('--pattern', nargs=2, type=int, required=True, help='棋盘内角点数 cols rows')
    p.add_argument('--square', type=float, required=True, help='棋盘方格物理尺寸')
    p.add_argument('--use-ransac', action='store_true', help='单应估计使用 RANSAC')
    # Degrees of freedom
    p.add_argument('--free-skew', action='store_true')
    p.add_argument('--free-tangential', action='store_true')
    p.add_argument('--enable-k3', action='store_true')
    p.add_argument('--fix-principal-point', action='store_true')
    # Criteria
    p.add_argument('--iter', type=int, default=100)
    p.add_argument('--eps', type=float, default=1e-6)
    # Visualization on test
    p.add_argument('--vis-out', type=str, default=None)
    p.add_argument('--show-vis', action='store_true')
    p.add_argument('--arrow-scale', type=float, default=1.0)
    # JSON
    p.add_argument('--log-json', type=str, default=None, help="保存条件+结果 JSON（路径或 '-' 输出到标准输出）")
    return p.parse_args()


def main():
    args = parse_args()

    # 解析数据来源
    if args.images:
        all_images = glob.glob(args.images)
        if not all_images:
            print('未找到图片：', args.images)
            return
        if args.split is None or not (0.0 < args.split < 1.0):
            print('请提供 0~1 之间的 --split 比例（如 0.7）以从单一 glob 随机划分 train/test')
            return
        rng = np.random.RandomState(args.split_seed) if args.split_seed is not None else np.random
        idx = np.arange(len(all_images))
        rng.shuffle(idx)
        n_train = max(1, min(len(all_images) - 1, int(round(len(all_images) * args.split))))
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        train_images = [all_images[i] for i in train_idx]
        test_images = [all_images[i] for i in test_idx]
    else:
        # 传统模式：显式提供 train/test globs
        if not args.train_images or not args.test_images:
            print('请提供 --images 与 --split，或同时提供 --train-images 与 --test-images')
            return
        train_images = glob.glob(args.train_images)
        test_images = glob.glob(args.test_images)
        if not train_images:
            print('未找到训练集图片：', args.train_images)
            return
        if not test_images:
            print('未找到测试集图片：', args.test_images)
            return

    pattern = (args.pattern[0], args.pattern[1])
    flags = build_flags(args)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(args.iter), float(args.eps))

    # Log experiment conditions
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n=== 训练/测试 拆分实验条件 ===')
    print('Time:', timestamp)
    print('OpenCV version:', cv2.__version__)
    if args.images:
        print('Images glob:', args.images, '| matched:', len(train_images) + len(test_images))
        print('Split ratio:', args.split, '| seed:', args.split_seed)
    else:
        print('Train glob:', args.train_images, '| matched:', len(train_images))
        print('Test  glob:', args.test_images,  '| matched:', len(test_images))
    print('Pattern:', pattern, 'square:', args.square)
    print('Use RANSAC:', args.use_ransac)
    print('Free skew:', args.free_skew, 'Free tangential:', args.free_tangential,
          'Enable k3:', args.enable_k3, 'Fix principal point:', args.fix_principal_point)
    print('Criteria: iter=', args.iter, ', eps=', args.eps)

    # Calibrate on training set
    print('\n[Calibrate] 使用训练集进行标定 ...')
    calib_res = calibrate_from_images(
        train_images, pattern, args.square,
        use_ransac=args.use_ransac, calib_flags=flags, criteria=criteria)

    print('训练集 - 使用图片数量：', len(calib_res['used_images']))
    print('K (优化)：\n', calib_res['K'])
    print('dist (k1,k2)：', calib_res['dist'])
    print('训练集整体 RMS (由 calibrateCamera 返回)：', calib_res.get('rms', 'N/A'))

    # Detect corners on train & test for evaluation RMS under learned K
    obj_tr, img_tr, used_tr = detect_corners(train_images, pattern, args.square)
    obj_te, img_te, used_te = detect_corners(test_images, pattern, args.square)

    K_est = calib_res['K']
    dist_est = calib_res['dist']

    train_overall, train_per, _ = compute_reprojection_stats(K_est, dist_est, obj_tr, img_tr)
    test_overall, test_per, test_residuals = compute_reprojection_stats(K_est, dist_est, obj_te, img_te)

    print('\n=== 拆分评估结果 ===')
    print('Train used images:', len(used_tr), '| Overall RMS (px):', train_overall)
    print('Test  used images:', len(used_te), '| Overall RMS (px):', test_overall)
    print('Per-image RMS (test):')
    for i, r in enumerate(test_per):
        print('  %02d: %.4f' % (i, r))

    # Optional visualization on test set
    if args.vis_out is not None or args.show_vis:
        vis_dir = args.vis_out or os.path.join(ROOT, 'images', 'eval_visualized_split')
        vis_dir = os.path.abspath(vis_dir)
        print('\nVisualizing test residuals to', vis_dir)
        visualize_residuals(used_te, img_te, test_residuals, vis_dir, show=args.show_vis, arrow_scale=args.arrow_scale)

    # JSON log
    if args.log_json:
        log = {
            'timestamp': timestamp,
            'opencv_version': cv2.__version__,
            'data': {
                'mode': 'single_glob_split' if args.images else 'separate_globs',
                'images_glob': args.images if args.images else None,
                'split_ratio': float(args.split) if args.images and args.split is not None else None,
                'split_seed': int(args.split_seed) if args.images and args.split_seed is not None else None,
                'train_glob': None if args.images else args.train_images,
                'test_glob': None if args.images else args.test_images,
                'train_matched': len(train_images),
                'test_matched': len(test_images)
            },
            'pattern': [pattern[0], pattern[1]],
            'square': float(args.square),
            'use_ransac': bool(args.use_ransac),
            'free_skew': bool(args.free_skew),
            'free_tangential': bool(args.free_tangential),
            'enable_k3': bool(args.enable_k3),
            'fix_principal_point': bool(args.fix_principal_point),
            'criteria': {'max_iter': int(args.iter), 'epsilon': float(args.eps)},
            'results': {
                'K_init': calib_res['K_init'].tolist(),
                'K': calib_res['K'].tolist(),
                'dist': calib_res['dist'].tolist(),
                'calibrate_rms': float(calib_res.get('rms', float('nan'))),
                'train_overall_rms': float(train_overall),
                'test_overall_rms': float(test_overall),
                'test_per_image_rms': test_per
            }
        }
        # 另存各自 used_images（便于复现）
        log['train'] = {
            'used': len(calib_res['used_images']),
            'used_images': calib_res['used_images']
        }
        log['test'] = {
            'used': len(used_te),
            'used_images': used_te
        }
        try:
            if args.log_json.strip() == '-':
                print('\n=== JSON LOG (split eval) ===')
                print(json.dumps(log, ensure_ascii=False, indent=2))
            else:
                out_path = os.path.abspath(args.log_json)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(log, f, ensure_ascii=False, indent=2)
                print('Split JSON log saved to', out_path)
        except Exception as e:
            print('写入拆分评估 JSON 日志失败：', repr(e))


if __name__ == '__main__':
    main()
