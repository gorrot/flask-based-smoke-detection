import cv2
import numpy as np
import os

# 视频的色彩处理
# def extrace_object_demo():
#     capture = cv2.VideoCapture("VIBE-master/data/input/viedo/Cotton_rope_smoke_04.avi")  # 打开视频文件
#     while (True):
#         ret, frame = capture.read()  # 读取视频
#         if ret == False:  # 如果打开失败，跳出循环
#             break;
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 进行色彩值转换，RGB到HSV
#         lower_hsv = np.array([0, 0, 80])  # 色彩范围h s v三变量的最小取值
#         upper_hsv = np.array([110, 40, 255])  # 色彩范围h s v三变量的最小取值
#         mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)  # 进行色值去范围，取出对应的色彩范围进行过滤
#         dst = cv2.bitwise_and(frame, frame, mask)  # 进行过滤frame=frame&mask
#         cv2.imshow("dst", dst)
#         c = cv2.waitKey(40)
#         if c == 27:
#             break
from matplotlib.pyplot import hsv

rootDir = r'data/output/vibe'
n = 0


def click_event(event, x, y, flags, hsv):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_pixel = hsv[y, x]
        print("HSV Value at ({}, {}): {}".format(x, y, hsv_pixel))


# def extrace_object_demo(n):
#     for lists in os.listdir(rootDir):
#         n+= 1
#         path = os.path.join(rootDir, lists)
#         frame = cv2.imread(path)
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 进行色彩值转换，RGB到HSV
#         lower_hsv = np.array([0, 0, 80])  # 色彩范围h s v三变量的最小取值
#         upper_hsv = np.array([110, 40, 255])  # 色彩范围h s v三变量的最小取值
#         mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)  # 进行色值去范围，取出对应的色彩范围进行过滤
#         dst = cv2.bitwise_and(frame, frame, mask)  # 进行过滤frame=frame&mask
#         cv2.namedWindow('123', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('123', 800, 600)
#         cv2.setMouseCallback('123', click_event, hsv)
#         origin_path = os.path.join('data/output/hsv', f"origin_{n}.jpg")
#         cv2.imwrite(origin_path, mask)
#         cv2.imshow("123", dst)
#         cv2.waitKey(0)

def extrace_object_demo(capture,n):
    #capture = cv2.VideoCapture("data/input/viedo/Cotton_rope_smoke_04.avi")  # 打开视频文件
    while (True):
        ret, frame = capture.read()  # 读取视频
        n += 1
        if ret == False:  # 如果打开失败，跳出循环
            break;
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 进行色彩值转换，RGB到HSV
        lower_hsv = np.array([0, 0, 160])  # 色彩范围h s v三变量的最小取值
        upper_hsv = np.array([110, 40, 255])  # 色彩范围h s v三变量的最小取值
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)  # 进行色值去范围，取出对应的色彩范围进行过滤
        dst = cv2.bitwise_and(frame, frame, mask)  # 进行过滤frame=frame&mask
        # cv2.namedWindow('123', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('123', 800, 600)
        # cv2.setMouseCallback('123', click_event, hsv)
        # origin_path = os.path.join('data/output/hsv', f"origin_{n}.jpg")
        # cv2.imwrite(origin_path, mask)
        cv2.imshow("123", mask)
        cv2.waitKey(1)



print("-------Hello Python--------")
extrace_object_demo(0)  # 色彩过滤

cv2.destroyAllWindows()
