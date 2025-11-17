import numpy as np
from calib.calibrate import compute_homography, intrinsic_from_homographies


def test_intrinsic_from_identity():
    # 构造两个简单的仿射/单应用于接口测试
    H1 = np.array([[1.0, 0.0, 0.0],[0.0,1.0,0.0],[0.0,0.001,1.0]])
    H2 = np.array([[0.9, 0.1, 10.0],[-0.05,1.02,5.0],[0.0005,0.0003,1.0]])
    K, b = intrinsic_from_homographies([H1, H2, H1*1.1])
    assert K.shape == (3,3)
    assert np.isfinite(K).all()
