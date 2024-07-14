import cv2

cap = cv2.VideoCapture('data/input/viedo/19.mp4')
ret, frame = cap.read()
prevframe = frame  # 第一帧
while True:
    ret, frame = cap.read()
    nextframe = frame
    if ret:
        diff = cv2.absdiff(prevframe, nextframe)
        diff = cv2.resize(diff, [600, 400])
        cv2.imshow('video', diff)
        prevframe = nextframe  # 帧差法 背景变化
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()
