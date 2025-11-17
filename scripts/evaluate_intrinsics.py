"""评估标定结果的脚本。

功能：
- 调用项目中的标定 pipeline（`calibrate_from_images`）获得估计的内参 `K` 与畸变 `dist`。
- 使用每张成功检测到角点的图片，基于估计的 `K` 用 `solvePnP` 恢复外参并计算重投影误差。
- 计算总体 RMS 重投影误差与每张图片的 RMS，输出可读报告并可将结果保存为 JSON。
- 可选：传入真实内参（npy 或 JSON）以比较参数误差。

示例：
python scripts/evaluate_intrinsics.py --images 'images/calib/*.png' --pattern 9 6 --square 0.025 --out report.json --gt-k gt_K.npy
"""

import argparse
import glob
import json
import os
import sys
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import cv2
from calib.calibrate import calibrate_from_images, detect_corners


def load_gt_K(path: str) -> Optional[np.ndarray]:
    if path is None:
        return None
    if path.endswith('.npy'):
        return np.load(path)
    else:
        # try json
        with open(path, 'r') as f:
            j = json.load(f)
        # expect dict with keys fx,fy,cx,cy,skew or full 3x3
        if isinstance(j, list):
            arr = np.array(j, dtype=float)
            return arr
        elif isinstance(j, dict):
            fx = float(j.get('fx', 0.0))
            fy = float(j.get('fy', 0.0))
            cx = float(j.get('cx', 0.0))
            cy = float(j.get('cy', 0.0))
            skew = float(j.get('skew', 0.0))
            K = np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
            return K
    return None


def compute_reprojection_stats(K: np.ndarray, dist: np.ndarray, objpoints, imgpoints):
    """返回 (overall_rms, per_image_rms list, residuals list)
    residuals list 每项为 (Npoints,2) 的误差向量（img - proj）
    """
    per_rms = []
    residuals = []
    all_sq = []
    # pad dist to 5-length for projectPoints (k1,k2,p1,p2,k3)
    dist_full = np.array([dist[0], dist[1], 0.0, 0.0, 0.0], dtype=float)
    for objp, imgp in zip(objpoints, imgpoints):
        # solvePnP to get accurate extrinsics under current K
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist_full, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            # fallback: zeros
            rvec = np.zeros((3,1), dtype=float)
            tvec = np.zeros((3,1), dtype=float)
        img_proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist_full)
        img_proj = img_proj.reshape(-1, 2)
        res = (imgp - img_proj)
        residuals.append(res)
        sq = (res ** 2).sum(axis=1)
        all_sq.extend(sq.tolist())
        rms = float(np.sqrt(sq.mean()))
        per_rms.append(rms)
    overall_rms = float(np.sqrt(np.mean(all_sq))) if len(all_sq) > 0 else 0.0
    return overall_rms, per_rms, residuals


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def visualize_residuals(images_paths, used_images, imgpoints_list, residuals_list, out_dir: str, show: bool = False, arrow_scale: float = 1.0):
    """在每张图片上绘制重投影残差向量并保存。

    images_paths: list of all input image paths (used to find files)
    used_images: list of image paths that were used/detected (same order as imgpoints_list and residuals_list)
    imgpoints_list: list of (N,2) observed image points
    residuals_list: list of (N,2) residual vectors (observed - projected)
    out_dir: directory to save visualized images
    arrow_scale: scale factor for drawing arrows (像素放大倍数)
    """
    ensure_dir(out_dir)
    # compute a reasonable max magnitude for color scaling
    all_mags = [np.linalg.norm(r, axis=1) for r in residuals_list]
    if len(all_mags) == 0:
        print('no residuals to visualize')
        return
    max_mag = max([m.max() if m.size>0 else 0.0 for m in all_mags])
    max_mag = max(max_mag, 1e-6)

    for path, imgp, res in zip(used_images, imgpoints_list, residuals_list):
        img = cv2.imread(path)
        if img is None:
            print(f'warning: cannot read {path} for visualization')
            continue
        vis = img.copy()
        mags = np.linalg.norm(res, axis=1)
        for (pt, r, mag) in zip(imgp, res, mags):
            px = int(round(pt[0]))
            py = int(round(pt[1]))
            # projected point = observed - residual
            proj_x = int(round(pt[0] - r[0]))
            proj_y = int(round(pt[1] - r[1]))
            # color: blue (small) -> red (large)
            t = min(1.0, mag / max_mag)
            color = (int(255 * t), 0, int(255 * (1 - t)))  # BGR
            # draw projected (small blue) and observed (filled circle)
            cv2.circle(vis, (proj_x, proj_y), 3, (255, 255, 255), -1)
            cv2.circle(vis, (px, py), 3, (0, 255, 0), -1)
            # draw arrow from projected to observed
            end_x = int(round(proj_x + r[0] * arrow_scale))
            end_y = int(round(proj_y + r[1] * arrow_scale))
            cv2.arrowedLine(vis, (proj_x, proj_y), (end_x, end_y), color, 1, tipLength=0.3)
        # overlay legend and stats
        overall_rms_img = float(np.sqrt((mags ** 2).mean())) if mags.size>0 else 0.0
        cv2.putText(vis, f'RMS(px): {overall_rms_img:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        # save
        fname = os.path.basename(path)
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, vis)
        print(f'wrote visualization {out_path}')
        if show:
            cv2.imshow('reproj', vis)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                break
    if show:
        cv2.destroyAllWindows()


def compare_K(K_est: np.ndarray, K_gt: np.ndarray):
    # return dict of absolute and relative errors for fx,fy,cx,cy,skew
    keys = ['fx', 'fy', 'cx', 'cy', 'skew']
    est = {'fx': float(K_est[0,0]), 'fy': float(K_est[1,1]), 'cx': float(K_est[0,2]), 'cy': float(K_est[1,2]), 'skew': float(K_est[0,1])}
    gt = {'fx': float(K_gt[0,0]), 'fy': float(K_gt[1,1]), 'cx': float(K_gt[0,2]), 'cy': float(K_gt[1,2]), 'skew': float(K_gt[0,1])}
    diffs = {}
    for k in keys:
        a = est[k]
        b = gt[k]
        abs_err = a - b
        rel_err = abs_err / b if b != 0 else float('inf')
        diffs[k] = {'est': a, 'gt': b, 'abs_err': abs_err, 'rel_err': rel_err}
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='图片通配符，例如 images/calib/*.png')
    parser.add_argument('--pattern', nargs=2, type=int, required=True, help='棋盘内角点数 cols rows')
    parser.add_argument('--square', type=float, required=True, help='方格物理尺寸（例如 0.025 米）')
    parser.add_argument('--use-ransac', action='store_true')
    parser.add_argument('--gt-k', type=str, default=None, help='可选：真实内参文件（.npy 或 JSON）')
    parser.add_argument('--out', type=str, default=None, help='可选：将评估报告保存为 JSON（仅评估结果；更完整可用 --log-json）')
    parser.add_argument('--vis-out', type=str, default=None, help='可选：保存误差可视化图像的目录（例如 images/eval_visualized）')
    parser.add_argument('--show-vis', action='store_true', help='可选：在屏幕上显示可视化图像')
    parser.add_argument('--arrow-scale', type=float, default=1.0, help='可选：重投影向量的缩放系数（用于可视化）')
    parser.add_argument('--log-json', type=str, default=None, help="可选：将本次评估的条件+结果输出为 JSON（提供路径，或 '-' 输出到标准输出）")
    args = parser.parse_args()

    images = glob.glob(args.images)
    if not images:
        print('未找到图片，检查 --images 通配符')
        return

    pattern = (args.pattern[0], args.pattern[1])
    # 先检测角点
    objpoints, imgpoints, used = detect_corners(images, pattern, args.square)
    if len(objpoints) < 3:
        print('至少需要 3 张成功检测到角点的图片进行可靠评估（当前: %d）' % len(objpoints))
        return

    # 运行标定 pipeline
    res = calibrate_from_images(images, pattern, args.square, use_ransac=args.use_ransac)
    K_est = res['K']
    dist_est = res['dist']

    overall_rms, per_rms, residuals = compute_reprojection_stats(K_est, dist_est, objpoints, imgpoints)

    report = {
        'K_est': K_est.tolist(),
        'dist_est': dist_est.tolist(),
        'overall_reprojection_rms_px': overall_rms,
        'per_image_rms_px': per_rms,
        'num_images_used': len(objpoints),
        'used_images': used,
    }

    if args.gt_k is not None:
        K_gt = load_gt_K(args.gt_k)
        if K_gt is None:
            print('无法载入 ground-truth K, 忽略对比')
        else:
            diffs = compare_K(K_est, K_gt)
            report['K_gt'] = K_gt.tolist()
            report['K_diffs'] = diffs

    # 打印报告摘要
    print('\n=== Intrinsics Evaluation Report ===')
    print('Images used for detection:', len(objpoints))
    print('Overall reprojection RMS (px):', overall_rms)
    print('Per-image reprojection RMS (px):')
    for i, r in enumerate(per_rms):
        print('  %02d: %.4f' % (i, r))
    if 'K_diffs' in report:
        print('\nK parameter differences (est vs gt):')
        for k, v in report['K_diffs'].items():
            print('  %s: est=%.6g gt=%.6g abs=%.6g rel=%.6g' % (k, v['est'], v['gt'], v['abs_err'], v['rel_err']))

    if args.out:
        outp = os.path.abspath(args.out)
        with open(outp, 'w') as f:
            json.dump(report, f, indent=2)
        print('\nReport saved to', outp)

    # 可视化残差
    vis_out_dir = args.vis_out if args.vis_out is not None else None
    if vis_out_dir is not None or args.show_vis:
        # default vis dir when user only asked to show
        if vis_out_dir is None:
            vis_out_dir = os.path.join(os.path.dirname(__file__), '..', 'images', 'eval_visualized')
        vis_out_dir = os.path.abspath(vis_out_dir)
        print('\nVisualizing residuals to', vis_out_dir)
        visualize_residuals(images, used, imgpoints, residuals, vis_out_dir, show=args.show_vis, arrow_scale=args.arrow_scale)

    # 结构化 JSON 日志（条件 + 标定 + 评估）
    if args.log_json:
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log = {
            'timestamp': timestamp,
            'opencv_version': cv2.__version__,
            'images_glob': args.images,
            'matched_image_count': len(images),
            'used_image_count': len(used),
            'used_images': used,
            'pattern': [pattern[0], pattern[1]],
            'square': float(args.square),
            'use_ransac': bool(args.use_ransac),
            'visualization': {
                'vis_out': vis_out_dir if vis_out_dir is not None else None,
                'show_vis': bool(args.show_vis),
                'arrow_scale': float(args.arrow_scale)
            },
            'calibration_results': {
                'K_init': res['K_init'].tolist(),
                'K': res['K'].tolist(),
                'dist': res['dist'].tolist(),
                'calibrate_rms': float(res.get('rms', float('nan')))
            },
            'evaluation_results': {
                'overall_reprojection_rms_px': float(overall_rms),
                'per_image_rms_px': per_rms
            }
        }
        if args.gt_k is not None and 'K_diffs' in locals() or 'K_diffs' in report:
            # 将 GT 对比并入日志（若存在）
            if 'K_gt' in report:
                log['evaluation_results']['K_gt'] = report['K_gt']
            if 'K_diffs' in report:
                log['evaluation_results']['K_diffs'] = report['K_diffs']
        try:
            if args.log_json.strip() == '-':
                print('\n=== JSON LOG (evaluate_intrinsics) ===')
                print(json.dumps(log, ensure_ascii=False, indent=2))
            else:
                out_path = os.path.abspath(args.log_json)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(log, f, ensure_ascii=False, indent=2)
                print('Evaluation JSON log saved to', out_path)
        except Exception as e:
            print('写入评估 JSON 日志失败：', repr(e))


if __name__ == '__main__':
    main()
