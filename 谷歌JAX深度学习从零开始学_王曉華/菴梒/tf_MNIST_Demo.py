import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.expand_dims(x_train,axis=3)
y_train = tf.one_hot(y_train,depth=10)
train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(1024).batch(256)



class MnistDemo(tf.keras.layers.Layer):
    def __init__(self):
        super(MnistDemo, self).__init__()

    def build(self, input_shape):

        self.conv_1 = tf.keras.layers.Conv2D(filters=1,kernel_size=3,activation=tf.nn.relu)
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.conv_2 = tf.keras.layers.Conv2D(filters=1,kernel_size=3,activation=tf.nn.relu)
        self.bn_2 = tf.keras.layers.BatchNormalization()

        self.dense = tf.keras.layers.Dense(10,activation=tf.nn.sigmoid)

        super(MnistDemo, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        embedding = inputs

        embedding = self.conv_1(embedding)
        embedding = self.bn_1(embedding)

        embedding = self.conv_2(embedding)
        embedding = self.bn_2(embedding)

        embedding = tf.keras.layers.Flatten()(embedding)
        logits = self.dense(embedding)
        return logits


import time
with tf.device("/GPU:0"):

    img = tf.keras.Input(shape=(28, 28, 1))
    logits = MnistDemo()(img)
    model = tf.keras.Model(img, logits)
    for i in range(4):
        start = time.time()
        model.compile(optimizer=tf.keras.optimizers.SGD(1e-3),loss= tf.keras.losses.categorical_crossentropy,metrics=["accuracy"])
        model.fit(train_data, epochs=50,verbose=0)
        end = time.time()
        print(f"开始第{i}个测试，循环运行时间:%.12f秒" % (end - start))