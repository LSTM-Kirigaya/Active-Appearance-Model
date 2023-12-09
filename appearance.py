from typing import List, Tuple

import numpy as np
import skimage.transform

def normalise_appearance(image: np.ndarray, s: np.ndarray, s_0: np.ndarray):
    assert len(s.shape) == 1
    assert len(s_0.shape) == 1
    
    source_control_points = s_0.reshape(-1, 2)
    target_control_points = s.reshape(-1, 2)
    h, w = image.shape[:2]
    affine_trasnform = skimage.transform.PiecewiseAffineTransform()
    affine_trasnform.estimate(src=source_control_points, dst=target_control_points)
    warped_image = skimage.transform.warp(image, affine_trasnform)
    return warped_image


def appearance_mean(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_num = images.shape[0]
    
    A_0 = np.expand_dims(images[np.random.randint(image_num)], 0)
    
    last_A_0 = A_0
    while True:
        beta = np.mean(images, axis=(1, 2), keepdims=True)
        alpha = np.mean(images * A_0, axis=(1, 2), keepdims=True)
        
        normalise_images = (images - beta) / alpha
        
        A_0 = np.mean(normalise_images, axis=0, keepdims=True)
        A_0 -= np.mean(A_0, axis=(1, 2), keepdims=True)
        A_0 /= np.mean(A_0 ** 2, axis=(1, 2), keepdims=True)
        
        if np.linalg.norm(last_A_0 - A_0) < 1e-6:
            break
        
        last_A_0 = A_0
    
    return A_0, normalise_images, beta, alpha
    

def appearance_modeling(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert len(images.shape) == 3
    image_num = len(images)
    A_0, normalise_images, beta, alpha = appearance_mean(images)
    cov_matrix = np.cov((normalise_images.reshape(image_num, -1) - A_0.reshape(1, -1)).T)
    eig_values, eig_vecs = np.linalg.eig(cov_matrix)
    indice = eig_values.argsort()[::-1]
    eig_values = eig_values[indice]
    eig_vecs = eig_vecs[:, indice]
    
    return A_0, normalise_images, eig_values, eig_vecs, beta, alpha