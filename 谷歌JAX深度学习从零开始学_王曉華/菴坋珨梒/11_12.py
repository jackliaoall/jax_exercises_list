import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax,GeneralConv)

from jax.experimental.stax import Conv
filter_num = 64 #卷积核数目，处理后生成的数据维度
filter_size = (3,3) #卷积核大小
strides = (2,2) #步进strides大小

Conv(filter_num, filter_size, strides)
Conv(filter_num, filter_size, strides, padding='SAME')

window_shape = (3,3) #池化窗大小
strides = (2,2) #步进strides大小
jax.experimental.stax.AvgPool(filter_size,strides)

# def ConvBlock(kernel_size, filters, strides=(2, 2)):
#   ks = kernel_size
#   filters1, filters2, filters3 = filters
#   Main = stax.serial(
#       Conv(filters1, (1, 1), strides), BatchNorm(), Relu,
#       Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
#       Conv(filters3, (1, 1)), BatchNorm())
#   Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
#   return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
  ks = kernel_size
  filters1, filters2 = filters
  print(filters1)
  def make_main(input_shape):
    # the number of output channels depends on the number of input channels
    return stax.serial(
        Conv(filters1, (1, 1)), BatchNorm(), Relu,
        Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
        Conv(input_shape[3], (1, 1)), BatchNorm())
  Main = stax.shape_dependent(make_main)
  return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)

IdentityBlock(3, [64, 64])