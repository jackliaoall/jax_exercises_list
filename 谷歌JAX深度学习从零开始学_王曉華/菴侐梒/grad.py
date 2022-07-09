import jax
import time
import jax.numpy as jnp

@jax.jit
def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

start = time.time()

x_small = jnp.arange(1024.)
derivative_fn = jax.grad(sum_logistic)
print(derivative_fn(x_small))

end = time.time()
print("循环运行时间:%.2f秒"%(end-start))

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

start = time.time()

jit_sum_logistic = jax.jit(sum_logistic)
x_small = jnp.arange(1024.)
derivative_fn = jax.grad(sum_logistic)
print(derivative_fn(x_small))

end = time.time()
print("循环运行时间:%.2f秒"%(end-start))

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

start = time.time()

derivative_fn = jax.grad(sum_logistic)
jit_sum_logistic = jax.jit(derivative_fn)
x_small = jnp.arange(1024.)
print(jit_sum_logistic(x_small))

end = time.time()
print("循环运行时间:%.2f秒"%(end-start))