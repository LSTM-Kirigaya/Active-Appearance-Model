import numpy as np
from typing import Tuple

def compute_kabsch_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Description
    	compute rotatiom matrix transform X to Y
    Param
        X: [k, 2]
        Y: [k, 2]
    """
    U, Sigma, Vt = np.linalg.svd(X @ Y.T)
    R = Vt.T @ U.T
    if np.linalg.det(R) == 1:
        return R
    else:
        I_star = np.eye(U.shape[0])
        I_star[-1, -1] = -1
        R = Vt.T @ I_star @ U.T
        return R

def normalise_shape(points_sets: np.ndarray) -> np.ndarray:
    """
   	Param
   		points_sets : [m, k, 2]
   		e.g. points_sets.shape=[110, 71, 2] 代表有 110 张图像，每张图像有 71 个控制点
   	Return
		normalise_z: [m, k, 2]
    """
    centroids = np.mean(points_sets, axis=1, keepdims=True)
    normalise_points_sets = points_sets - centroids
    points_sets_norm = np.linalg.norm(normalise_points_sets, axis=(1, 2), keepdims=True)
    normalise_points_sets /= points_sets_norm
    return normalise_points_sets

def denormalise_shape(points_sets: np.ndarray, reference_points_sets: np.ndarray):
    """
    将中心化空间内的 points_sets 变换到真实空间 reference_points_sets 中
    Param
   		points_sets : [m, k, 2]
        reference_points_sets: [m, k, 2] 
   		e.g. points_sets.shape=[110, 71, 2] 代表有 110 张图像，每张图像有 71 个控制点
   	Return
		normalise_z: [m, k, 2]
    """
    centroids = np.mean(reference_points_sets, axis=1, keepdims=True)
    normalise_points_sets = reference_points_sets - centroids
    points_sets_norm = np.linalg.norm(normalise_points_sets, axis=(1, 2), keepdims=True)
    denormalise_points_sets = points_sets * points_sets_norm + centroids
    return denormalise_points_sets

def shape_mean(points_sets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description
    	根据输入点集计算形体均值 s_0
    Param
    	points_sets: [m, k, 2]
    Return
    	s_0: 计算得到的形体均值
    	C: 朝向 s_0 以 kabsch 矩阵旋转后的结果
    """
    C = points_sets
    # 使用第一个图的形体作为 s_0 的初始值
    s_0 = np.expand_dims(C[0], axis=0)
    image_num = C.shape[0]
    last_s_0 = s_0
    while True:
        C = normalise_shape(C)
        s_0 = normalise_shape(s_0)
        for i in range(image_num):
            R = compute_kabsch_matrix(C[i], s_0[0])
            C[i] = R @ C[i]
        s_0 = np.mean(C, axis=0, keepdims=True)
        s_0 /= np.linalg.norm(s_0)
        if np.linalg.norm(last_s_0 - s_0) < 1e-5:
            break
        last_s_0 = s_0
    return s_0, C

def shape_modeling(points_sets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    assert len(points_sets.shape) == 3, 'ensure `points_sets` is a 3D tensor!'
    s_0, aligned_C = shape_mean(points_sets)
    
    # 将 控制点集合 转换为 形体集合
    s_0 = s_0[0].reshape(-1)
    aligned_C = aligned_C.reshape(len(aligned_C), -1)
    
    # 进行 PCA，压缩图像数量的维度
    cov_matrix = np.cov(aligned_C.T)
    eig_values, eig_vecs = np.linalg.eig(cov_matrix)
    
    indice = eig_values.argsort()[::-1]
    eig_values = eig_values[indice]
    eig_vecs = eig_vecs[:, indice]
    
    return s_0, aligned_C, eig_values, eig_vecs
