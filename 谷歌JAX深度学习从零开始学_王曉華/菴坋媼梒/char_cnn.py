import jax
import jax.numpy as jnp
from jax import grad
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental import optimizers
from jax.experimental.stax import (Conv, Dense,MaxPool,
                                   Flatten,
                                   Relu, LogSoftmax)

import get_char_embedding
x_train, y_train = get_char_embedding.get_dataset()
key = jax.random.PRNGKey(17)
x_train = jax.random.shuffle(key,x_train)
y_train = jax.random.shuffle(key,y_train)

x_test = x_train[:12000]
y_test = y_train[:12000:]

x_train = x_train[12000:]
y_train = y_train[12000:]



def charCNN(num_classes):
    return stax.serial(

        Conv(1, (3, 3)),Relu,
        Conv(1, (5, 5)),Relu,
        MaxPool((3,3),(1,1)),
        Conv(1, (3, 3)), Relu,
        Flatten,
        Dense(256),Relu,
        Dense(num_classes), LogSoftmax

    )

init_random_params, predict = charCNN(5)

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



input_shape = [-1,64,28,1]
#这里的step_size就是学习率
opt_init, opt_update, get_params = optimizers.adam(step_size = 2.17e-3)
_, init_params = init_random_params(key, input_shape)
opt_state = opt_init(init_params)

batch_size = 128
total_num = (120000-12000)   	 #这里读者根据硬件水平自由设定全部的训练数据，总量为120000
for _ in range(170):
    epoch_num = int(total_num//batch_size)
    print(f"{_}轮训练开始")
    for i in range(epoch_num):
        start = i * batch_size
        end = (i + 1) * batch_size

        data = x_train[start:end]
        targets = y_train[start:end]
        opt_state = update((i), opt_state, (data, targets))

        if (i + 1)%79 == 0:
            params = get_params(opt_state)
            loss_value = loss(params,(data, targets))
            print(f"loss:{loss_value}")
    params = get_params(opt_state)
    print(f"{_}轮训练结束")


    train_acc = []
    correct_preds = 0.0
    test_epoch_num = int(12000 // batch_size)
    for i in range(test_epoch_num):
        start = i * batch_size
        end = (i + 1) * batch_size
        data = x_test[start:end]
        targets = y_test[start:end]
        correct_preds += pred_check(params, (data, targets))
    train_acc.append(correct_preds / float(total_num))
    print(f"Training set accuracy: {(train_acc)}")

