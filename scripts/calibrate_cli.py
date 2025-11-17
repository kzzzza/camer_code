"""Zhang 相机标定命令行工具 (calibrate_cli)

功能概述：
- 读取一批棋盘格标定图片（支持 glob 通配符）
- 运行：角点检测 -> 单应估计 (可选 RANSAC) -> Zhang 线性内参初值 -> OpenCV 联合优化
- 输出：初始/优化内参矩阵、径向畸变系数、使用的图片数量、整体重投影 RMS

重要说明：
1. pattern 参数是棋盘“内角点”数量 (cols rows)，与生成脚本保持一致。
2. square 是棋盘格物理尺寸（单位任意，影响外参尺度，不影响像素 RMS）。
3. 默认更稳健：固定 skew、固定 k3、切向畸变置零、主点可调。
4. 若视图少或角点检测失败会导致初值不稳，重投影误差大。
5. 可用 evaluate_intrinsics.py + visualize_corners.py 做误差和角点质量诊断。

典型使用：
python scripts/calibrate_cli.py --images 'images/calib/*.png' --pattern 9 6 --square 0.029 --use-ransac
"""

import argparse
import glob
import sys
import os
import datetime
import json

# 确保项目根加入 sys.path（支持从任意工作目录调用脚本）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calib import calibrate_from_images


def build_flags(args):
    """根据命令行开关与当前 OpenCV 版本安全构造标定 flags。
    安全策略：
    - 缺失的常量忽略，不抛异常
    - 默认使用 CALIB_USE_INTRINSIC_GUESS
    - 固定 skew（除非 --free-skew）
    - 切向畸变置零（除非 --free-tangential）
    - 固定 k3（除非 --enable-k3）
    - 固定主点（仅在传 --fix-principal-point 且版本支持）
    """
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


def parse_args():
    """解析命令行参数并返回命名空间."""
    parser = argparse.ArgumentParser(
        description='Zhang 相机标定 CLI（支持 RANSAC 与多自由度对比实验）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--images', required=True,
                        help='标定图片 glob 通配符，例如 /path/to/*.jpg')
    parser.add_argument('--pattern', nargs=2, required=True, type=int,
                        help='棋盘格内角点数：cols rows（例：9 6）')
    parser.add_argument('--square', required=True, type=float,
                        help='棋盘格单元物理尺寸（任意单位，如 0.029）')
    parser.add_argument('--use-ransac', action='store_true',
                        help='单应估计使用 RANSAC（提高鲁棒性）')

    # 自由度控制
    parser.add_argument('--free-skew', action='store_true',
                        help='允许估计 skew（大多相机不推荐，默认固定为 0）')
    parser.add_argument('--free-tangential', action='store_true',
                        help='允许切向畸变 p1/p2（默认置零）')
    parser.add_argument('--enable-k3', action='store_true',
                        help='允许三阶径向畸变 k3（默认固定为 0）')
    parser.add_argument('--fix-principal-point', action='store_true',
                        help='固定主点（图像中心附近），默认不固定')

    # 迭代与收敛
    parser.add_argument('--iter', type=int, default=100,
                        help='联合优化最大迭代次数')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='联合优化收敛阈值 (criteria epsilon)')

    # 日志输出
    parser.add_argument('--log-json', type=str, default=None,
                        help="将本次实验条件与标定结果保存为 JSON（提供路径，或 '-' 输出到标准输出）")

    return parser.parse_args()


def main():
    args = parse_args()

    # 解析图片列表
    images = glob.glob(args.images)
    if not images:
        print('未找到匹配的图片：', args.images)
        return

    pattern = (args.pattern[0], args.pattern[1])
    flags = build_flags(args)

    # ===== 实验条件日志（标定前） =====
    import cv2
    opencv_version = cv2.__version__
    def decode_flags(f):
        names = []
        candidates = [
            'CALIB_USE_INTRINSIC_GUESS', 'CALIB_FIX_SKEW', 'CALIB_ZERO_TANGENT_DIST',
            'CALIB_FIX_TANGENT_DIST', 'CALIB_FIX_K3', 'CALIB_FIX_PRINCIPAL_POINT'
        ]
        for n in candidates:
            if hasattr(cv2, n):
                val = getattr(cv2, n)
                if f & val:
                    names.append(n)
        return names
    flag_names = decode_flags(flags)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n=== 标定实验条件 ===')
    print(f'Time: {timestamp}')
    print(f'OpenCV version: {opencv_version}')
    print(f'Images glob: {args.images}')
    print(f'Matched image count: {len(images)}')
    print(f'Pattern (cols x rows): {pattern[0]} x {pattern[1]} (total corners: {pattern[0]*pattern[1]})')
    print(f'Square size (physical units): {args.square}')
    print(f'Use RANSAC: {args.use_ransac}')
    print(f'Free skew: {args.free_skew}')
    print(f'Free tangential: {args.free_tangential}')
    print(f'Enable k3: {args.enable_k3}')
    print(f'Fix principal point: {args.fix_principal_point}')
    print(f'Max iterations (criteria): {args.iter}')
    print(f'Epsilon (criteria): {args.eps}')
    print('Resolved OpenCV flags:', ', '.join(flag_names) if flag_names else '(none / unsupported)')
    print('================================')

    # 构造终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                int(args.iter),
                float(args.eps))

    # 调用整体标定
    try:
        res = calibrate_from_images(
            images,
            pattern,
            args.square,
            use_ransac=args.use_ransac,
            calib_flags=flags,
            criteria=criteria
        )
    except Exception as e:
        print('标定失败：', repr(e))
        return

    # 输出结果
    print('\n=== 标定结果 ===')
    print('图片数量（匹配到）：', len(images))
    print('成功使用图片数量：', len(res['used_images']))
    # 打印使用的图片（过长时截断）
    MAX_LIST = 10
    used_list = res['used_images']
    if len(used_list) <= MAX_LIST:
        print('Used images:')
        for p in used_list:
            print('  -', p)
    else:
        print(f'Used images (first {MAX_LIST}/{len(used_list)}):')
        for p in used_list[:MAX_LIST]:
            print('  -', p)
        print('  ... (truncated)')
    # 记录实验条件快照（便于日志对比）
    print('\n=== 条件快照复核 ===')
    print(f'RANSAC={args.use_ransac}, skew_free={args.free_skew}, tangential_free={args.free_tangential}, k3_enabled={args.enable_k3}, fix_pp={args.fix_principal_point}')
    print('Flags decoded:', ', '.join(flag_names) if flag_names else '(none)')
    print('================================')
    print('K (初始)：\n', res['K_init'])
    print('K (优化)：\n', res['K'])
    print('dist (k1,k2)：', res['dist'])
    print('整体重投影 RMS (px)：', res.get('rms', 'N/A'))
    # 给出简单合理性提示
    fx, fy = res['K'][0, 0], res['K'][1, 1]
    skew = res['K'][0, 1]
    cx, cy = res['K'][0, 2], res['K'][1, 2]
    print(f'焦距 fx, fy：{fx:.2f}, {fy:.2f} | skew：{skew:.4f} | 主点 (cx, cy)：({cx:.2f}, {cy:.2f})')
    if abs(skew) > 1e-3:
        print('提示：skew 非常规接近 0，若无需 skew 请移除 --free-skew 开关。')
    if res.get('rms', 1e9) > 2.0:
        print('警告：RMS > 2 像素，检查角点检测质量、视角多样性，或启用 --use-ransac。')

    print('\n可下一步运行 evaluate_intrinsics.py 进行误差可视化。')

    # ===== 结构化 JSON 日志输出 =====
    if args.log_json:
        log = {
            'timestamp': timestamp,
            'opencv_version': opencv_version,
            'images_glob': args.images,
            'matched_image_count': len(images),
            'used_image_count': len(res['used_images']),
            'used_images': res['used_images'],
            'pattern': [pattern[0], pattern[1]],
            'square': args.square,
            'use_ransac': bool(args.use_ransac),
            'free_skew': bool(args.free_skew),
            'free_tangential': bool(args.free_tangential),
            'enable_k3': bool(args.enable_k3),
            'fix_principal_point': bool(args.fix_principal_point),
            'criteria': {
                'type': 'EPS+MAX_ITER',
                'max_iter': int(args.iter),
                'epsilon': float(args.eps)
            },
            'opencv_flags': {
                'names': flag_names,
                'value': int(flags)
            },
            'results': {
                'K_init': res['K_init'].tolist() if hasattr(res['K_init'], 'tolist') else res['K_init'],
                'K': res['K'].tolist() if hasattr(res['K'], 'tolist') else res['K'],
                'dist': res['dist'].tolist() if hasattr(res['dist'], 'tolist') else res['dist'],
                'rms': float(res.get('rms', float('nan')))
            }
        }
        try:
            if args.log_json.strip() == '-':
                print('\n=== JSON LOG ===')
                print(json.dumps(log, ensure_ascii=False, indent=2))
            else:
                out_path = os.path.abspath(args.log_json)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(log, f, ensure_ascii=False, indent=2)
                print('JSON log saved to', out_path)
        except Exception as e:
            print('写入 JSON 日志失败：', repr(e))


if __name__ == '__main__':
    main()
