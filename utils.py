import cv2
import numpy as np
import settings
def compute_TVL1(dir):
    "这个算法可能有点问题需要修改一下"
    cap = cv2.VideoCapture(dir)
    # 获取第一帧
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame3=np.expand_dims(frame1,axis=0)
    hsv = np.zeros_like(frame1)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获得视频文件的帧率
    # 遍历每一行的第1列
    hsv[..., 1] = 255
    if cap.isOpened():
        open, frame = cap.read()
        ideo = frame
        cv2.waitKey(0)
    else:
        open = False
    num = 0
    while num < fps * settings.time:
        num+=1
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        if ret == True and num % (int(fps / settings.fps)) == 0:
            # 返回一个两通道的光流向量，实际上是每个点的像素位移值
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # 笛卡尔坐标转换为极坐标，获得极轴和极角
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            rgb = np.expand_dims(rgb, axis=0)
            frame3 = np.vstack((frame3, rgb))
    frame3 = frame3[1:] / 255
    cap.release()
    return list(frame3)