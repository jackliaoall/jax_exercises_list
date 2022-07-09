import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax

num_classes = 10
reshape_args = [(-1, 28 * 28), (-1,)]
input_shape = reshape_args[0]

step_size = 0.001
num_epochs = 10
momentum_mass = 0.9
rng = random.PRNGKey(0)

x_train = jnp.load("../第一章/mnist_train_x.npy")
y_train = jnp.load("../第一章/mnist_train_y.npy")

print(x_train.shape)
def one_hot_nojit(x, k=10, dtype=jnp.float32):
    """ Create a one-hot encoding of x of size k. """
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


total_train_imgs = len(y_train)
y_train = one_hot_nojit(y_train)
# ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(256).prefetch(
#     tf.data.experimental.AUTOTUNE)
# ds_train = tfds.as_numpy(ds_train)


def pred_check(params, batch):
    """ Correct predictions over a minibatch. """
    # 这里我做了修正，因为预测生成的结果是[-1,10],而输入的target也被我改成了[-1,10],
    # 所以这里需要2个jnp.argmax做一个转换。
    inputs, targets = batch
    predict_result = predict(params, inputs)
    predicted_class = jnp.argmax(predict_result, axis=1)
    targets = jnp.argmax(targets, axis=1)
    return jnp.sum(predicted_class == targets)


# {Dense(1024) -> ReLU}x2 -> Dense(10) -> LogSoftmax
init_random_params, predict = stax.serial(
    stax.Dense(1024), stax.Relu,
    stax.Dense(1024), stax.Relu,
    stax.Dense(10), stax.LogSoftmax)


def loss(params, batch):
    """ Cross-entropy loss over a minibatch. """
    inputs, targets = batch
    return jnp.mean(jnp.sum(-targets * predict(params, inputs), axis=1))


# def update(i, opt_state, batch):
#     """ Single optimization step over a minibatch. """
#     params = get_params(opt_state)
#     return opt_update(i, grad(loss)(params, batch), opt_state)


#这里的step_size就是学习率
opt_init, opt_update, get_params = optimizers.adam(step_size = 2e-4)
_, init_params = init_random_params(rng, input_shape)
opt_state = opt_init(init_params)



for _ in range(17):

    data = x_train.reshape((-1, 28 * 28))
    targets = y_train.reshape((-1, 10))
    opt_state = opt_update(_,grad(loss)(get_params(opt_state),(data, targets)),opt_state)

    params = get_params(opt_state)
    #上面是训练部分，这里是存档部分，这里直接仿照numpy进行存档即可

    train_acc = []
    #上面是载入部分，直接仿照numpy中数据进行载入即可
    #params = jnp.load("params.npy",allow_pickle =True)
    # Train Acc
    correct_preds = 0.0
    correct_preds += pred_check(params, (data, targets))

    train_acc.append(correct_preds / float(total_train_imgs))
    print(f"Training set accuracy: {train_acc}")
