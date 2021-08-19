#这里是数据处理的文件处理成数据集的形式
import cv2
import numpy as np
import settings
import os
import tensorflow as tf
from utils import compute_TVL1
def cv_show(name,img,time=0):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img.shape
def opencv_video(dir):
    #"video/008.mp4"
    vc=cv2.VideoCapture(dir)
    frame_count = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vc.get(cv2.CAP_PROP_FPS)  # 获得视频文件的帧率
    open, frame1 = vc.read()
    frame1=np.expand_dims(frame1,axis=0)
    if vc.isOpened():
        open,frame=vc.read()
        ideo=frame
        cv2.waitKey(0)
    else:
        open=False
    num=0
    while num<fps*settings.time:
        num+=1
        ret,frame=vc.read()
        frame=np.expand_dims(frame,axis=0)
        if frame is None:
            break

        if ret==True and num%(int(fps/settings.fps))==0:
            frame1=np.vstack((frame1,frame))
    #把第一张重的图片删去
    frame1=frame1[1:]/255
    vc.release()
    return list(frame1)
def train_test_get(train_test_inf):
    for root,dir,files in os.walk(train_test_inf, topdown=False):
        #print(root)
        #print(files)
        list1=[root+"/"+i for i in files]
        return list1
def make_dataset(dir):
    fake_list=train_test_get(dir+"/fake")
    true_list=train_test_get(dir+"/true")
    fake_len=len(fake_list)
    true_len=len(true_list)
    label_fake=[[1,0] for _ in range(fake_len)]
    label_true=[[0,1] for _ in range(true_len)]
    list_get1=fake_list+true_list
    list_get=tf.constant([opencv_video(i) for i in list_get1])
    list_get=tf.expand_dims(list_get,axis=1)
    list_get_TVL1=tf.constant([compute_TVL1(i) for i in list_get1])
    list_get_TVL1=tf.expand_dims(list_get_TVL1,axis=1)
    list_get=tf.concat([list_get,list_get_TVL1],axis=1)
    label_get=label_fake+label_true
    dataest = tf.data.Dataset.from_tensor_slices((list_get, label_get))
    dataest = dataest.shuffle(buffer_size=settings.buffer_size).prefetch(tf.data.experimental.AUTOTUNE).repeat(settings.repeat).batch(settings.batch_size)
    print(dataest)
    return dataest
