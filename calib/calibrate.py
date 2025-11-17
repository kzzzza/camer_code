"""
Zhang 标定法 + OpenCV 联合优化的标定流程实现。

模块功能概览：
1. generate_object_points         生成棋盘格物体点 (z=0 平面)。
2. detect_corners                 检测角点并返回 (objpoints/imgpoints) 列表。
3. compute_homography / DLT       计算单应矩阵（可选 RANSAC）。
4. intrinsic_from_homographies    Zhang 线性解内参初值。
5. extrinsics_from_homography     从单应与内参恢复每张图的外参（仅用于分析）。
6. refine_parameters              使用 cv2.calibrateCamera 联合优化内参+外参。
7. calibrate_from_images          整合以上步骤形成完整 pipeline。

设计要点与约束：
- 最终标定依赖 OpenCV 的 calibrateCamera 联合优化，避免只调内参导致发散。
- 线性初值可能退化时，会通过 sanitize 修正（主点入图、skew 归零、焦距合理）。
- 默认固定 skew、切向畸变、k3（可通过 CLI 打开自由度）。
- 支持 RANSAC 提高单应鲁棒性，但不是必须。
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import cv2

# 若需要最小二乘自定义优化，可扩展使用 scipy；当前版本已弃用之前的 least_squares 优化外参方式。
# from scipy.optimize import least_squares


# ---------------------------
# 1. 生成棋盘格物体点
# ---------------------------
def generate_object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    """
    生成棋盘格平面上的 3D 点（Z=0）。
    pattern_size: (cols, rows) 内角点数量（注意：是角点，不是方格数）
    square_size: 单个格子物理尺寸（单位可自行约定，如米、毫米）。
    返回: (N,3) float64 (按行扫描：X 横向，Y 纵向，Z 固定 0)
    """
    cols, rows = pattern_size
    # rows*cols 个点，初始化为零，Z=0
    objp = np.zeros((rows * cols, 3), dtype=np.float64)
    # 使用 mgrid 生成网格坐标（列优先）
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # 缩放到实际物理单位
    objp *= square_size
    return objp


# ---------------------------
# 2. 检测角点
# ---------------------------
def detect_corners(image_paths: List[str],
                   pattern_size: Tuple[int, int],
                   square_size: float,
                   flags: int = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                   refine: Dict[str, Any] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    逐图检测棋盘角点，返回成功的物体点列表、图像点列表与对应图像路径列表。
    refine: cornerSubPix 参数，包含：
        win      亚像素窗口大小
        zero     死区 (-1,-1)
        criteria 终止条件
    返回:
        objpoints: List[(N,3)] 每张图对应的 3D 物体点
        imgpoints: List[(N,2)] 每张图对应的 2D 像素点
        used:     成功检测的图像路径
    """
    if refine is None:
        refine = {
            'win': (11, 11),
            'zero': (-1, -1),
            'criteria': (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        }

    template_objp = generate_object_points(pattern_size, square_size)
    objpoints, imgpoints, used = [], [], []

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"warning: could not read {p}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray,
                                                   (pattern_size[0], pattern_size[1]),
                                                   flags)
        if not found:
            print(f"warning: chessboard not found in {p}")
            continue

        # 亚像素精细化
        corners_refined = cv2.cornerSubPix(gray,
                                           corners,
                                           refine['win'],
                                           refine['zero'],
                                           refine['criteria'])

        objpoints.append(template_objp.copy())
        imgpoints.append(corners_refined.reshape(-1, 2))
        used.append(p)

    return objpoints, imgpoints, used


# ---------------------------
# 3. 单应矩阵（DLT / RANSAC）
# ---------------------------
def compute_homography(objp: np.ndarray, imgp: np.ndarray) -> np.ndarray:
    """
    使用 DLT 直接线性法估计平面单应矩阵 H。
    输入:
        objp: (N,2)/(N,3) 只用前两个坐标 X,Y
        imgp: (N,2) 像素点 u,v
    约定: s * [u v 1]^T = H * [X Y 1]^T
    返回: H (3x3) 归一化使 H[2,2]=1
    """
    pts_obj = objp[:, :2]
    n = pts_obj.shape[0]
    A = []
    for i in range(n):
        X, Y = pts_obj[i]
        u, v = imgp[i]
        # 经典 DLT 两行
        A.append([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    return H / H[2, 2]


def compute_homographies(objpoints_list: List[np.ndarray],
                         imgpoints_list: List[np.ndarray],
                         ransac: bool = False,
                         ransac_thresh: float = 3.0) -> List[np.ndarray]:
    """
    为每一视图计算单应矩阵列表。
    ransac=True 时使用 cv2.findHomography 提高鲁棒性；失败则退回 DLT。
    返回: List[H]
    """
    Hs = []
    for objp, imgp in zip(objpoints_list, imgpoints_list):
        src = objp[:, :2].astype(np.float64)
        dst = imgp.astype(np.float64)
        if ransac:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
            if H is None:
                # 回退 DLT
                H = compute_homography(src, dst)
        else:
            H = compute_homography(src, dst)
        Hs.append(H)
    return Hs


# ---------------------------
# 4. 线性估计内参（Zhang）
# ---------------------------
def intrinsic_from_homographies(Hs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据多张平面单应矩阵估计初始内参 K。
    使用文献中构造 V b = 0 的方式求解对称矩阵 B 的元素，再恢复 K。

    返回:
        K: 3x3 初始内参
        b: 解向量（用于调试/分析条件数）
    注意:
        - 对退化情况未强力正则化，需调用方在后续做体检与修正。
    """
    def v_pq(H, p, q):
        return np.array([
            H[0, p] * H[0, q],
            H[0, p] * H[1, q] + H[1, p] * H[0, q],
            H[1, p] * H[1, q],
            H[2, p] * H[0, q] + H[0, p] * H[2, q],
            H[2, p] * H[1, q] + H[1, p] * H[2, q],
            H[2, p] * H[2, q]
        ], dtype=np.float64)

    V = []
    for H in Hs:
        H = H / H[2, 2]
        V.append(v_pq(H, 0, 1))              # v_01
        V.append(v_pq(H, 0, 0) - v_pq(H, 1, 1))  # v_00 - v_11
    V = np.array(V)

    # 最小特征向量对应 b
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]

    B11, B12, B22, B13, B23, B33 = b
    denom = (B11 * B22 - B12 ** 2)
    v0 = (B12 * B13 - B11 * B23) / denom
    lamb = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    if B11 == 0 or lamb * B11 < 0:
        raise RuntimeError('invalid solution for intrinsics (division by zero or negative scale)')
    alpha = np.sqrt(lamb / B11)
    beta = np.sqrt(lamb * B11 / denom)
    gamma = -B12 * alpha ** 2 * beta / lamb
    u0 = gamma * v0 / beta - B13 * alpha ** 2 / lamb
    K = np.array([[alpha, gamma, u0],
                  [0.0,   beta,  v0],
                  [0.0,   0.0,   1.0]], dtype=np.float64)
    return K, b


# ---------------------------
# 5. 外参恢复（仅用于分析，不参与最终优化）
# ---------------------------
def extrinsics_from_homography(H: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据单应矩阵和内参恢复外参 (rvec,tvec)。
    数学：H ~ K [r1 r2 t]，通过正交化保证旋转矩阵有效。
    返回:
        rvec: (3,) Rodrigues 旋转向量
        tvec: (3,)
    """
    K_inv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lam = 1.0 / np.linalg.norm(K_inv @ h1)
    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    r3 = np.cross(r1, r2)
    t = lam * (K_inv @ h3)

    R = np.column_stack((r1, r2, r3))
    # 正交化（SVD 保证 R 是最近的正交矩阵）
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    rvec, _ = cv2.Rodrigues(R_ortho)
    return rvec.reshape(3,), t.reshape(3,)


# ---------------------------
# 6. 联合优化（OpenCV calibrateCamera）
# ---------------------------
def refine_parameters(K_init: np.ndarray,
                      dist_init: np.ndarray,
                      objpoints: List[np.ndarray],
                      imgpoints: List[np.ndarray],
                      image_size: Tuple[int, int],
                      flags: int = None,
                      criteria: Tuple[int, int, float] = None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], float]:
    """
    使用 OpenCV calibrateCamera 对内参 + 外参 + 部分畸变参数做联合最小化。

    输入：
        K_init: 线性初值（可能偏差较大）
        dist_init: 初始畸变 (k1,k2)
        objpoints/imgpoints: 每张图的点列表
        image_size: (width, height)
        flags: OpenCV 优化标志组合（自动兼容缺失常量）
        criteria: 终止准则 (type, max_iter, epsilon)

    返回：
        K_opt: 优化后内参
        dist_opt: 优化后前两个径向畸变系数 (k1,k2)
        rvecs/tvecs: 所有视图外参
        rms: 重投影总 RMS (像素)

    关键稳健处理：
        - sanitize 初始 K（主点入图、焦距正、skew 置 0）
        - 只输出前两个径向畸变，其他保持为 0，可按需扩展
    """
    # 构造默认 flags（安全组合）
    if flags is None:
        flags = getattr(cv2, 'CALIB_USE_INTRINSIC_GUESS', 0)
        if hasattr(cv2, 'CALIB_FIX_SKEW'):
            flags |= getattr(cv2, 'CALIB_FIX_SKEW')
        if hasattr(cv2, 'CALIB_ZERO_TANGENT_DIST'):
            flags |= getattr(cv2, 'CALIB_ZERO_TANGENT_DIST')
        elif hasattr(cv2, 'CALIB_FIX_TANGENT_DIST'):
            flags |= getattr(cv2, 'CALIB_FIX_TANGENT_DIST')
        if hasattr(cv2, 'CALIB_FIX_K3'):
            flags |= getattr(cv2, 'CALIB_FIX_K3')

    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # OpenCV 期望 vector<vector<Point*>>
    obj_cv = [op.reshape(-1, 3).astype(np.float32) for op in objpoints]
    img_cv = [ip.reshape(-1, 2).astype(np.float32) for ip in imgpoints]

    # 将 (k1,k2) 放入 5 参数数组，其余设置 0
    dist5 = np.array([
        dist_init[0] if dist_init.size > 0 else 0.0,
        dist_init[1] if dist_init.size > 1 else 0.0,
        0.0, 0.0, 0.0
    ], dtype=np.float64)

    def _sanitize_initial_K(K0: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """
        修正初始 K 以避免 OpenCV 报错:
        - fx/fy 非正或无效 => 设置为 1.2 * max(w,h)
        - 主点不在图像范围 => 移到中心
        - skew => 归零（多数版本不支持非零 skew）
        """
        w, h = image_size
        K = K0.copy()
        # 非有限值的整体替换
        if not np.isfinite(K).all():
            K = np.array([[max(w, h), 0.0, 0.5 * w],
                          [0.0, max(w, h), 0.5 * h],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
        # 单独检查 fx, fy
        if (not np.isfinite(K[0, 0])) or K[0, 0] <= 0:
            K[0, 0] = 1.2 * max(w, h)
        if (not np.isfinite(K[1, 1])) or K[1, 1] <= 0:
            K[1, 1] = 1.2 * max(w, h)
        # 主点必须在图像范围内
        if (not np.isfinite(K[0, 2])) or K[0, 2] < 0 or K[0, 2] >= w:
            K[0, 2] = 0.5 * w
        if (not np.isfinite(K[1, 2])) or K[1, 2] < 0 or K[1, 2] >= h:
            K[1, 2] = 0.5 * h
        # 强制 skew=0
        K[0, 1] = 0.0
        # 保持齐次形式
        K[2, 0] = 0.0
        K[2, 1] = 0.0
        K[2, 2] = 1.0
        return K

    K0 = _sanitize_initial_K(K_init, image_size)

    # 执行联合优化
    ret_rms, K_opt, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_cv, img_cv, image_size, K0, dist5, flags=flags, criteria=criteria
    )

    # 仅取前两个径向系数，便于保持与上层接口一致
    dist_opt = np.array([
        distCoeffs.ravel()[0] if distCoeffs.size > 0 else 0.0,
        distCoeffs.ravel()[1] if distCoeffs.size > 1 else 0.0
    ], dtype=np.float64)

    return K_opt, dist_opt, rvecs, tvecs, float(ret_rms)


# ---------------------------
# 7. 整体 Pipeline
# ---------------------------
def calibrate_from_images(image_paths: List[str],
                          pattern_size: Tuple[int, int],
                          square_size: float,
                          use_ransac: bool = False,
                          refine_opts: Dict[str, Any] = None,
                          opt_options: Dict[str, Any] = None,
                          calib_flags: int = None,
                          criteria: Tuple[int, int, float] = None) -> Dict[str, Any]:
    """
    完整标定流程封装：
        角点检测 -> 单应估计 -> 张氏线性初值 -> 联合优化 -> 返回结果

    输入：
        image_paths: 图片路径集合
        pattern_size: (cols, rows)
        square_size: 单个格物理尺寸
        use_ransac: 是否在单应估计中使用 RANSAC
        refine_opts: 角点亚像素参数（传给 detect_corners）
        calib_flags: OpenCV 标定 flags（为空则构造兼容默认）
        criteria: 标定终止条件

    返回字典：
        {
            'K_init': 线性初值内参,
            'K':      优化后内参,
            'dist':   (k1,k2),
            'rvecs':  所有视图旋转向量,
            'tvecs':  所有视图平移向量,
            'used_images': 成功检测到角点的图像路径,
            'b_vec':  张氏线性方程解向量（调试用）,
            'rms':    重投影 RMS 像素
        }
    异常：
        - <3 张有效视图会抛 RuntimeError
        - 无法读示例图尺寸会抛 RuntimeError
        - 线性内参解退化会抛 RuntimeError
    """
    objpoints, imgpoints, used = detect_corners(
        image_paths, pattern_size, square_size, refine=refine_opts
    )
    if len(objpoints) < 3:
        raise RuntimeError('至少需要 3 张成功检测到棋盘角点的图片')

    Hs = compute_homographies(objpoints, imgpoints, ransac=use_ransac)
    K_init, b = intrinsic_from_homographies(Hs)

    if len(used) == 0:
        raise RuntimeError('未获得有效的角点检测图像')

    sample_img = cv2.imread(used[0])
    if sample_img is None:
        raise RuntimeError(f'无法读取示例图像获取尺寸: {used[0]}')
    h, w = sample_img.shape[:2]
    image_size = (w, h)

    dist_init = np.zeros(2, dtype=np.float64)

    if calib_flags is None:
        calib_flags = getattr(cv2, 'CALIB_USE_INTRINSIC_GUESS', 0)
        if hasattr(cv2, 'CALIB_FIX_SKEW'):
            calib_flags |= getattr(cv2, 'CALIB_FIX_SKEW')
        if hasattr(cv2, 'CALIB_ZERO_TANGENT_DIST'):
            calib_flags |= getattr(cv2, 'CALIB_ZERO_TANGENT_DIST')
        elif hasattr(cv2, 'CALIB_FIX_TANGENT_DIST'):
            calib_flags |= getattr(cv2, 'CALIB_FIX_TANGENT_DIST')
        if hasattr(cv2, 'CALIB_FIX_K3'):
            calib_flags |= getattr(cv2, 'CALIB_FIX_K3')

    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    K_opt, dist_opt, rvecs, tvecs, rms = refine_parameters(
        K_init, dist_init, objpoints, imgpoints, image_size,
        flags=calib_flags, criteria=criteria
    )

    return {
        'K_init': K_init,
        'K': K_opt,
        'dist': dist_opt,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'used_images': used,
        'b_vec': b,
        'rms': rms
    }
