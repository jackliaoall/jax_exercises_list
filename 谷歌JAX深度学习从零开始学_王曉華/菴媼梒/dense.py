import jax.numpy as jnp
from jax import random

def Dense(dense_shape = [2, 1]):
  rng = random.PRNGKey(17)

  weight = random.normal(rng, shape=dense_shape)
  bias = random.normal(rng, shape=(dense_shape[-1],))
  params = [weight,bias]

  def apply_fun(inputs,params = params):				#apply_fun是python特性之一，称为内置函数
    W, b = params
    return jnp.dot(inputs, W) + b
  return apply_fun

rng = random.PRNGKey(18)
dense_shape = [2, 1]
weight = random.normal(rng, shape=dense_shape)
bias = random.normal(rng, shape=(dense_shape[-1],))
params2 = [weight,bias]

mat_a = jnp.array([[1.7,1.7],[2.14,2.14]])
res = Dense()(mat_a,params2)

print(res)