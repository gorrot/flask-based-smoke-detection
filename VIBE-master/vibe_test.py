import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def initial_background(I_gray, N):
    I_pad = np.pad(I_gray, 1, 'symmetric')
    height = I_pad.shape[0]
    width = I_pad.shape[1]
    samples = np.zeros((height, width, N))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            for n in range(N):
                x, y = 0, 0
                while (x == 0 and y == 0):
                    x = np.random.randint(-1, 1)
                    y = np.random.randint(-1, 1)
                ri = i + x
                rj = j + y
                samples[i, j, n] = I_pad[ri, rj, 0]
    samples = samples[1:height - 1, 1:width - 1]
    return samples


def vibe_detection(I_gray, samples, _min, N, R):
    height = I_gray.shape[0]
    width = I_gray.shape[1]
    segMap = np.zeros((height, width)).astype(np.uint8)
    for i in range(height):
        for j in range(width):
            count, index, dist = 0, 0, 0
            while count < _min and index < N:
                dist = np.abs(I_gray[i, j] - samples[i, j, index])
                if dist < R:
                    count += 1
                index += 1
            if count >= _min:
                r = np.random.randint(0, N - 1)
                if r == 0:
                    r = np.random.randint(0, N - 1)
                    samples[i, j, r] = I_gray[i, j]
                r = np.random.randint(0, N - 1)
                if r == 0:
                    x, y = 0, 0
                    while (x == 0 and y == 0):
                        x = np.random.randint(-1, 1)
                        y = np.random.randint(-1, 1)
                    r = np.random.randint(0, N - 1)
                    ri = i + x
                    rj = j + y
                    try:
                        samples[ri, rj, r] = I_gray[i, j]
                    except:
                        pass
            else:
                segMap[i, j] = 255

    return segMap, samples


rootDir = r'data/input/viedo'
video_file = os.path.join(rootDir, os.listdir(rootDir)[0])  # 使用目录中的第一个视频文件
cap = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
ret, frame = cap.read()
prevframe = frame  # 第一帧
N = 20
R = 20
n = 0
_min = 2
phai = 16
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
samples = initial_background(frame, N)

output_video_path = 'VIBE-master/data/output/vibe/1.MP4'
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
#
# for lists in os.listdir(rootDir):
#     path = os.path.join(rootDir, lists)
#     frame = cv2.imread(path)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     segMap, samples = vibe_detection(gray, samples, _min, N, R)
#     cv2.imshow('segMap', segMap)
#     if cv2.waitKey(1) and 0xff == ord('q'):
#         break
# cv2.destroyAllWindows()
while True:
    for i in range(1):
        ret, frame = cap.read()
        n += 1
    if not ret:
        print("Error: Could not read frame.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower_hsv = np.array([0, 0, 160])  # 色彩范围h s v三变量的最小取值
    upper_hsv = np.array([110, 40, 255])  # 色彩范围h s v三变量的最小取值
    mask = cv2.inRange(frame, lowerb=lower_hsv, upperb=upper_hsv)  # 进行色值去范围，取出对应的色彩范围进行过滤
    segMap, samples = vibe_detection(gray, samples, _min, N, R)
    opening = cv2.morphologyEx(segMap, cv2.MORPH_OPEN, kernel)  # 开运算
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  # 闭运算
    dst = cv2.bitwise_and(closing, mask)
    origin_path = os.path.join('data/output', f"origin_{n}.jpg")
    vibe_path = os.path.join('data/output', f"vibe_{n}.jpg")
    dst_path = os.path.join('data/output/vibe', f"dst_{n}.jpg")
    oc_path = os.path.join('data/output/vibe', f"oc_{n}.jpg")
    cv2.imwrite(oc_path, closing)
    # cv2.imwrite(vibe_path, segMap)
    # cv2.imwrite(dst_path, dst)
    # out.write(closing)
    cv2.imshow('segMap', closing)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放 VideoCapture 对象和关闭窗口（如果有的话）
cap.release()
cv2.destroyAllWindows()
