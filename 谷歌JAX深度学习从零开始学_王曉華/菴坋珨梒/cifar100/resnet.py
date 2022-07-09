import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, Identity,
                                   MaxPool, Relu, LogSoftmax,softmax)


def ConvBlock(kernel_size, filters, strides=(1, 1)):
    ks = kernel_size
    filters1, filters2, filters3 = filters
    Main = stax.serial(
        Conv(filters1, (1, 1), strides, padding='SAME'), BatchNorm(), Relu,
        Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
        Conv(filters3, (1, 1), padding='SAME'), BatchNorm())
    Shortcut = stax.serial(Conv(filters3, (1, 1), strides, padding='SAME'), BatchNorm())
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
  ks = kernel_size
  filters1, filters2 = filters
  def make_main(input_shape):
    # the number of output channels depends on the number of input channels
    return stax.serial(
        Conv(filters1, (1, 1), padding='SAME'), BatchNorm(), Relu,
        Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
        Conv(input_shape[3], (1, 1), padding='SAME'), BatchNorm())
  Main = stax.shape_dependent(make_main)
  return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)


def ResNet50(num_classes):
    return stax.serial(
        Conv(64, (3, 3), padding='SAME'),
        BatchNorm(), Relu,
        #MaxPool((3, 3), strides=(2, 2)),

        ConvBlock(3, [64, 64, 256]),
        IdentityBlock(3, [64, 64]),
        IdentityBlock(3, [64, 64]),

        # ConvBlock(3, [128, 128, 512]),
        # IdentityBlock(3, [128, 128]),
        # IdentityBlock(3, [128, 128]),
        # IdentityBlock(3, [128, 128]),

        # ConvBlock(3, [256, 256, 1024]),
        # IdentityBlock(3, [256, 256]),
        # IdentityBlock(3, [256, 256]),
        # IdentityBlock(3, [256, 256]),
        # IdentityBlock(3, [256, 256]),
        # IdentityBlock(3, [256, 256]),
        #
        # ConvBlock(3, [512, 512, 2048]),
        # IdentityBlock(3, [512, 512]),
        # IdentityBlock(3, [512, 512]),

        #AvgPool((7, 7)),
        Flatten, Dense(num_classes),
        LogSoftmax
    )


if __name__ == '__main__':
    key = jax.random.PRNGKey(17)
    image = jax.random.normal(key,shape=[10,32,32,3])

    init_random_params, predict = ResNet50(100)
    input_shape = (-1,32, 32, 3)

    opt_init, opt_update, get_params = optimizers.adam(step_size=2e-4)
    _, init_params = init_random_params(key, input_shape)
    opt_state = opt_init(init_params)

    params = get_params(opt_state)
    result = predict(params,image)

    print(result.shape)