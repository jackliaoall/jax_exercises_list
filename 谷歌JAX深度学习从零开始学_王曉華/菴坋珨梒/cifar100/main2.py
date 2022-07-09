import jax.nn
import jax.numpy as jnp
from 第十一章.cifar100 import get_data, resnet
from jax import grad
from jax.experimental import optimizers
import tensorflow as tf
import tensorflow_datasets as tfds


with tf.device("/CPU:0"):
    train_dataset, label_dataset, test_dataset, test_label_dataset = get_data.get_CIFAR100_dataset(root="./cifar-10-batches-py/")

    train_dataset = jnp.reshape(train_dataset,[-1,3,32,32])
    x_train = jnp.transpose(train_dataset,[0,2,3,1])
    y_train = jax.nn.one_hot(label_dataset,num_classes=100)

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    ds_train = tfds.as_numpy(ds_train)

    test_dataset = jnp.reshape(test_dataset,[-1,3,32,32])
    x_test = jnp.transpose(test_dataset,[0,2,3,1])
    y_test = jax.nn.one_hot(test_label_dataset,num_classes=100)


init_random_params, predict = resnet.ResNet50(100)

def pred_check(params, batch):
    inputs, targets = batch
    predict_result = predict(params, inputs)
    predicted_class = jnp.argmax(predict_result, axis=1)
    targets = jnp.argmax(targets, axis=1)
    return jnp.sum(predicted_class == targets)



def loss(params, batch):
    inputs, targets = batch
    return jnp.mean(jnp.sum(-targets * predict(params, inputs), axis=1))


def update(i, opt_state, batch):
    """ Single optimization step over a minibatch. """
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)


key = jax.random.PRNGKey(17)
input_shape = [-1,32,32,3]
#这里的step_size就是学习率
opt_init, opt_update, get_params = optimizers.adam(step_size = 2e-4)
_, init_params = init_random_params(key, input_shape)
opt_state = opt_init(init_params)

batch_size = 32
total_num = 50000   	 #这里读者根据硬件水平自由设定全部的训练数据，总量为50000
for _ in range(17):
    epoch_num = int(total_num//batch_size)
    print(f"{_}轮训练开始")

    for batch_raw in ds_train:
        data = batch_raw[0]
        targets = batch_raw[1]
        opt_state = update((929), opt_state, (data, targets))
    print(f"{_}轮训练结束")

    # for i in range(epoch_num):
    #     start = i * batch_size
    #     end = (i + 1) * batch_size
    #
    #
    #
    #     if (i + 1)%50 == 0:
    #         params = get_params(opt_state)
    #         loss_value = loss(params,(data, targets))
    #         print(f"loss:{loss_value}")
    # params = get_params(opt_state)


    # train_acc = []
    # correct_preds = 0.0
    # for i in range(epoch_num):
    #     start = i * batch_size
    #     end = (i + 1) * batch_size
    #     data = x_test[start:end]
    #     targets = y_test[start:end]
    #     correct_preds += pred_check(params, (data, targets))
    # train_acc.append(correct_preds / float(total_num))
    # print(f"Training set accuracy: {(train_acc)}")
