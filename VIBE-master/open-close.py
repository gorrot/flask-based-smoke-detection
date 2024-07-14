import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素

img1 = cv2.imread('j_noise_out.bmp')  # 开运算原始图像
img2 = cv2.imread('j_noise_in.bmp')  # 闭运算原始图像

opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)  # 开运算
closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)  # 闭运算

titles = ['Original_OP', 'Opening', 'Original_CL', 'Closing']
images = [img1, opening, img2, closing]
plt.figure(dpi=150)  # 指定输出像素大小
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i], fontsize=10)
    plt.xticks([]), plt.yticks([])
plt.show()
