import cv2
import numpy as np
import glob


def sort_image(image_list, start_index, row_num, col_num, file_path, img_height=320, img_width=320):
    new_image = np.zeros(shape=(int(img_height * row_num), (img_width * col_num), 3), dtype=np.uint8)
    for j in range(start_index, row_num * col_num):
        new_i = int(j / col_num)
        new_j = int(j % col_num)
        row_index = new_i * img_height
        col_index = new_j * img_width
        for m in range(img_height):
            for n in range(img_width):
                new_image[row_index + m][col_index + n] = image_list[j][m][n]
    cv2.imwrite(file_path + str(start_index) + '.jpg', new_image)

    return start_index + row_num * col_num


def decompose_img(img, start_index, img_height, img_width, file_path):
    new_img = np.zeros(shape=(img_height, img_width, 3), dtype=np.uint8)
    col_num = int(img.shape[1] / img_width)
    row_num = int(img.shape[0] / img_height)
    for m in range(img.shape[0]):
        new_i = int(m / img_height)
        real_i = int(m % img_height)
        for n in range(img.shape[1]):
            new_j = int(n / img_width)
            real_j = int(n % img_width)
            new_index = start_index + int(new_i * col_num) + new_j
            new_img[real_i][real_j] = img[m][n]
            if real_i == img_height - 1 and real_j == img_width - 1:
                cv2.imwrite(file_path + str(new_index + start_index) + '.jpg', new_img)
    return start_index + col_num * row_num


image_names = glob.glob('Dataset/ori_scenes/*.*')
image_num = len(image_names)

i = 0

while i < image_num:
    left_num = image_num - i
    if left_num >= 100:
        i = sort_image(image_names, i, 10, 10, 'Dataset/sort_images/')
    else:
        i = sort_image(image_names, i, 1, left_num, 'Dataset/sort_images/')