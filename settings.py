import os
#定义截取视频的秒数
time=10
#定义每秒截取多少张照片
fps=1
#训练集地址
train="video"
#测试集地址
test="test_video"
# 如果不存在这个out文件夹，就自动创建一个
if not os.path.exists(train):
    os.mkdir(train)
    os.mkdir(train+"/fake")
    os.mkdir(train+"/true")
if not os.path.exists(test):
    os.mkdir(test)
    os.mkdir(test + "/fake")
    os.mkdir(test + "/true")
#数据集操作
#代表数据集混乱度
buffer_size=1000
#batchsize
batch_size=10
#数据集重复次数
repeat=10

#视频高度
height=480
widght=640
#保存model路径，设为none默认不报错路径
out_model="model"
if not os.path.exists(out_model):
    os.mkdir(out_model)

epoch=15
#每过几次保存模型
each_epoch=5