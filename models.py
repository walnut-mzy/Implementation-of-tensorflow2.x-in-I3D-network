import tensorflow as tf
import settings
from tensorflow import keras,metrics
class Inc(keras.layers.Layer):
    def __init__(self,fitter1,fitter2,fitter3,fitter4,fitter5,fitter6):
        super(Inc, self).__init__()
        self.conv2d1=tf.keras.layers.Conv3D(
            filters=fitter1,
            kernel_size=(1,1,1),
            strides=1,
            padding="same",
            activation="relu"
        )
        self.conv2d2 = tf.keras.layers.Conv3D(
            filters=fitter2,
            kernel_size=(1, 1, 1),
            strides=1,
            padding="same",
            activation="relu"
        )
        self.conv2d3 = tf.keras.layers.Conv3D(
            filters=fitter3,
            kernel_size=(1, 1, 1),
            strides=1,
            padding="same",
            activation="relu"
        )
        self.maxpool3d1=tf.keras.layers.MaxPooling3D(
            strides=1,
            padding="same",
            pool_size=(3,3,3),
        )
        self.conv2d4=tf.keras.layers.Conv3D(
            filters=fitter4,
            kernel_size=(3,3,3),
            strides=1,
            padding="same",
            activation="relu"
        )
        self.conv2d5 = tf.keras.layers.Conv3D(
            filters=fitter5,
            kernel_size=(3, 3, 3),
            strides=1,
            padding="same",
            activation="relu"
        )
        self.conv2d6 = tf.keras.layers.Conv3D(
            filters=fitter6,
            kernel_size=(1, 1, 1),
            strides=1,
            padding="same",
            activation="relu"
        )
        self.bn1=tf.keras.layers.BatchNormalization()
        self.bn2=tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.bn6 = tf.keras.layers.BatchNormalization()
    def call(self,inputs,**kwargs):
        #batch1
        batch1=self.conv2d1(inputs)
        batch1=self.bn1(batch1)
        #batch2
        batch2=self.conv2d2(inputs)
        batch2 = self.bn2(batch2)
        batch2=self.conv2d4(batch2)
        batch2 = self.bn3(batch2)
        #batch3
        batch3=self.conv2d3(inputs)
        batch3 = self.bn4(batch3)
        batch3=self.conv2d5(batch3)
        batch3=self.bn5(batch3)
        #batch4
        batch4=self.maxpool3d1(inputs)
        batch4=self.conv2d6(batch4)
        batch4=self.bn6(batch4)
        x = tf.concat([batch1,batch2,batch3,batch4], axis=4)
        return x
class Inception3D_part(keras.layers.Layer):
    def __init__(self):
        super(Inception3D_part, self).__init__()
        self.cnv3d=tf.keras.layers.Conv3D(
            filters=64,
            kernel_size=(7,7,7),
            strides=2,
            padding="same",
            use_bias=False,
            activation="relu"
        )
        self.cnv3d1=tf.keras.layers.Conv3D(
            filters=64,
            kernel_size=(1, 1, 1),
            strides=1,
            padding="same",
            use_bias = False,
            activation="relu"
        )
        self.cnv3d2=tf.keras.layers.Conv3D(
            filters=192,
            kernel_size=(3,3,3),
            strides=(1,1,1),
            padding="same",
            use_bias=False,
            activation="relu"
        )

        self.maxpool3d1=tf.keras.layers.MaxPooling3D(
            padding="same",
            strides=(1,2,2),
            pool_size=(1,3,3)
        )
        self.maxpool3d2 = tf.keras.layers.MaxPooling3D(
            padding="same",
            strides=(1, 2, 2),
            pool_size=(1, 3, 3)
        )
        self.Inc1=Inc(64,96,16,128,32,32)
        self.Inc2=Inc(128,128,32,192,96,64)
        self.maxpool3d3=tf.keras.layers.MaxPooling3D(
            padding="same",
            strides=(2, 2, 2),
            pool_size=(3, 3, 3)
        )
        self.Inc3=Inc(192,96,16,208,48,64)
        self.Inc4=Inc(160,112,24,224,64,64)
        self.Inc5=Inc(128,128,24,256,64,64)
        self.Inc6=Inc(112,144,32,288,64,64)
        self.Inc7=Inc(256,160,32,320,128,128)
        self.maxpool3d4=tf.keras.layers.MaxPooling3D(
            padding="same",
            strides=(2, 2, 2),
            pool_size=(2, 2, 2)
        )
        self.Inc8=Inc(256,160,32,320,128,128)
        self.Inc9=Inc(384,192,48,384,128,128)
        self.avpool3d=tf.keras.layers.AveragePooling3D(
            padding="valid",
            strides=1,
            pool_size=(2, 7, 7)
        )
        self.cnv3d3=tf.keras.layers.Conv3D(
            filters=400,
            kernel_size=(1, 1, 1),
            strides=1,
            padding="same",
        )
        self.bn1=tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
    def call(self,inputs,**kwargs):
        x=self.cnv3d(inputs)
        x=self.bn1(x)
        x=self.maxpool3d1(x)
        x=self.cnv3d1(x)
        x=self.bn2(x)
        x=self.cnv3d2(x)
        x=self.bn3(x)
        x=self.maxpool3d2(x)
        x=self.Inc1(x)
        x=self.Inc2(x)
        x=self.maxpool3d3(x)
        x=self.Inc3(x)
        x=self.Inc4(x)
        x=self.Inc5(x)
        x=self.Inc6(x)
        x=self.Inc7(x)
        x=self.Inc8(x)
        x=self.Inc9(x)
        x=self.avpool3d(x)
        x=self.cnv3d3(x)
        return x
class Inception3D(keras.layers.Layer):
    def __init__(self):
        super(Inception3D, self).__init__()
        self.Inception3D1=Inception3D_part()
        self.Inception3D2=Inception3D_part()
    def call(self,inputs,**kwargs):
        x=tf.split(inputs,[1,1],axis=1)
        batch1=x[0]
        batch2=x[1]
        batch1 = tf.squeeze(batch1, axis=1)
        batch2 = tf.squeeze(batch2, axis=1)
        x1=self.Inception3D1(batch1)
        x2=self.Inception3D2(batch2)
        x=tf.concat([x1,x2],axis=1)
        return x
model=tf.keras.Sequential([
    Inception3D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024,activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dense(2,activation="softmax"),
])
model.build(input_shape=(None,2,settings.time*settings.fps,settings.height,settings.widght,3))
model.summary()