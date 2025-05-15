import cv2
from abf import grad, smooth_image
import numpy as np
import os
import glob
from joblib import Parallel, delayed


def if_smooth_images(img_name):
    print(os.path.basename(img_name) + ' start!')
    img = cv2.imread(img_name)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    perpend_grad_image = grad(gray_img, iterations=1, mu=5)
    smoothed_img = smooth_image(img, perpend_grad_image, iterations=10)
    cv2.imwrite('Dataset/smoothed_images/' + os.path.basename(img_name), smoothed_img.astype(np.uint8))


if __name__ == '__main__':
    img_list = glob.glob('Dataset/ori_scenes/*.*')
    target_list = glob.glob('Dataset/smoothed_images/*.*')
    target_base_list = []

    for target_name in target_list:
        target_base_list.append(os.path.basename(target_name))

    k = 0
    while k < len(img_list):
        batch_images = []
        batch_size = 0
        while batch_size <= 32 and k < len(img_list):
            if os.path.basename(img_list[k]) not in target_base_list:
                batch_images.append(img_list[k])
                batch_size += 1
            k += 1
        if batch_size > 0:
            Parallel(n_jobs=batch_size)(delayed(if_smooth_images)(ba_img) for ba_img in batch_images)
