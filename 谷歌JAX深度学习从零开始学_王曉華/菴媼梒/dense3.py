import jax.numpy as jnp
from jax import random



mat_a = jnp.array([[1.7,1.7],[2.14,2.14]])

def Dense17(dense_shape = [2, 1]):
  def init_fun(input_shape = dense_shape):
    rng = random.PRNGKey(17)
    W, b = random.normal(rng, shape=input_shape), random.normal(rng, shape=(input_shape[-1],))
    return (W, b)
  def apply_fun(inputs,params):
    W, b = params
    return jnp.dot(inputs, W) + b
  return init_fun, apply_fun


init_fun, apply_fun = Dense17()
res = apply_fun(mat_a,init_fun())

print(res)
params17 = (init_fun())


def Dense18(dense_shape = [2, 1]):
  def init_fun(input_shape = dense_shape):
    rng = random.PRNGKey(18)    #注意这里我修正了随机数
    W, b = random.normal(rng, shape=input_shape), random.normal(rng, shape=(input_shape[-1],))
    return (W, b)
  def apply_fun(inputs,params):
    W, b = params
    return jnp.dot(inputs, W) + b
  return init_fun, apply_fun
print("---------------------------")
init_fun, apply_fun18 = Dense18()
res = apply_fun18(mat_a,params17)   #注意这里我依旧使用的是Dense17生成的参数
print(res)