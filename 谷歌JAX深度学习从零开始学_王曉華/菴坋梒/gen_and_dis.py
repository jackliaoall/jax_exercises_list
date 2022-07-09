import jax
import jax.numpy as jnp
from jax import grad
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental import optimizers

# import tensorflow as tf
# import tensorflow_datasets as tfds

mnist_data = jnp.load("../第一章/mnist_train_x.npy")
mnist_data = jnp.expand_dims(mnist_data,axis=-1)
mnist_data = (mnist_data - 256)/256.#这样确保值在 -1,1之间

features = 32

def generator(features = 32):
    return stax.serial(

        stax.ConvTranspose(features * 2,[3, 3], [2, 2]),stax.BatchNorm(),stax.Relu,
        stax.ConvTranspose(features * 4, [4, 4], [1, 1]), stax.BatchNorm(), stax.Relu,
        stax.ConvTranspose(features * 2, [3, 3], [2, 2]), stax.BatchNorm(), stax.Relu,
        stax.ConvTranspose(1, [4, 4], [2, 2]), stax.Tanh
    )


def discriminator(features = 32):
    return stax.serial(
        stax.Conv(features,[4, 4], [2, 2]),stax.BatchNorm(), stax.LeakyRelu,
        stax.Conv(features, [4, 4], [2, 2]), stax.BatchNorm(), stax.LeakyRelu,
        stax.Conv(2, [4, 4], [2, 2]),stax.Flatten
    )



if __name__ == '__main__':
    key = jax.random.PRNGKey(17)
    #下面是测试fake_image的处理
    # fake_image = jax.random.normal(key,shape=[10,1,1,1])
    #
    # init_random_params, predict = generator()
    # fake_shape = (-1,1, 1, 1)
    #
    # opt_init, opt_update, get_params = optimizers.adam(step_size=2e-4)
    # _, init_params = init_random_params(key, fake_shape)
    # opt_state = opt_init(init_params)
    #
    # params = get_params(opt_state)
    # result = predict(params,fake_image)
    #
    # print(result.shape)


    #下面是测试real_image的处理
    real_image = jax.random.normal(key,shape=[10,28,28,1])
    init_random_params, predict = discriminator()
    real_shape = (-1, 28,28, 1)

    opt_init, opt_update, get_params = optimizers.adam(step_size=2e-4)
    _, init_params = init_random_params(key, real_shape)
    opt_state = opt_init(init_params)
    params = get_params(opt_state)

    result = predict(params, real_image)
    print(result.shape)

