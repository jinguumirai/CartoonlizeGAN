import glob
import cv2

image_list = glob.glob(r'C:\Users\jinguumirai\Pictures\genshin\*.*')
i = 0

for image_name in image_list:
    img = cv2.imread(image_name, cv2.CV_8UC2)
    img_height = img.shape[0]
    img_width = img.shape[1]
    new_size = 0
    if img_height > img_width:
        new_height = int(img_height / 2)
        mid = new_height
        if new_height >= img_width:
            new_size = img_width
        else:
            new_size = new_height
        new_delta = int(new_size / 2)
        mid_width = int(img_width / 2)
        img1 = img[mid - new_size: mid, mid_width - new_delta: mid_width + new_delta, :]
        img2 = img[mid: mid + new_size, mid_width - new_delta: mid_width + new_delta, :]
    else:
        new_width = int(img_width / 2)
        mid = new_width
        if new_width >= img_height:
            new_size = img_height
        else:
            new_size = new_width
        new_delta = int(new_size / 2)
        mid_height = int(img_height / 2)
        img1 = img[mid_height - new_delta: mid_height + new_delta, mid - new_size: mid, :]
        img2 = img[mid_height - new_delta: mid_height + new_delta, mid: mid + new_size, :]
    img1 = cv2.resize(img1, (480, 480))
    img1 = img1[40: 360, 80: 400, :]
    img2 = cv2.resize(img2, (500, 500))
    img2 = img2[40: 360, 80: 400, :]
    cv2.imwrite('Dataset/genshin_scenes/' + str(i) + '.jpg', img1)
    i += 1
    cv2.imwrite('Dataset/genshin_scenes/' + str(i) + '.jpg', img2)
    i += 1
