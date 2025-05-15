import cv2
import numpy as np

x = cv2.imread(r'C:\Users\jinguumirai\Pictures\genshin\8b6f30b30ba0c7cc515aa99af7a98658_2445295498380995909.png')

y = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
num = 0
sat = 0
light = 0.
s = y[:, :, 1]
for i in range(y.shape[0]):
    for j in range(y.shape[1]):
        if 19 / 36 * 255 < y[i][j][0] < 3 / 4 * 255:
            sat += y[i][j][1]
            light += y[i][j][2]
            num += 1

print(sat / num)
print(light / num)


