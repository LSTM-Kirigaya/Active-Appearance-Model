import os
import numpy as np
import cv2
from PIL import Image
from typing import Tuple

from shape import shape_mean

def get_coordinates(img_path, scale_x: int = 1, scale_y : int = 1):
    img_name = img_path.split('/')[-1].split('.')[0]
    coord_path = os.path.join(os.path.join(*img_path.split('/')[0:-1]), 'asf/{}.asf'.format(img_name))
    with open(coord_path) as f:
        content = f.readlines()
    lines = content[16: 89]
    points, from_ids, to_ids = [], [], []
    for info in lines:
        datas = info.split('\t')
        x = int(float(datas[2]) * scale_x)
        y = int(float(datas[3]) * scale_y)
        points.append((x, y))
        from_ids.append(int(datas[5]))
        to_ids.append(int(datas[6]))
    
    return points, from_ids, to_ids

if __name__ == '__main__':
    datasets_home = './Active-Appearance-Models/data/imm3943/IMM-Frontal Face DB SMALL'
    C = []
    shapes = []
    
    for image_file in os.listdir(datasets_home):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):           
            image_path = os.path.join(datasets_home, image_file)
            points, _, _ = get_coordinates(image_path, scale_x=800, scale_y=600)
            C.append(points)
    
    C = np.array(C)
    s_0, aligned_C = shape_mean(C)
    print(s_0.shape)