import jax
import time
import jax.numpy as jnp

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

start = time.time()
x_small = jnp.arange(1024000.)
derivative_fn = (jax.grad(sum_logistic))
end = time.time()
print("循环运行时间:%.2f秒"%(end-start))


start = time.time()
x_small = jnp.arange(1024000.)
derivative_fn = jax.vmap(jax.grad(sum_logistic))
end = time.time()
print("循环运行时间:%.2f秒"%(end-start))


start = time.time()
x_small = jnp.arange(1024000.)
derivative_fn = jax.jit(jax.vmap(jax.grad(sum_logistic)))
end = time.time()
print("循环运行时间:%.2f秒"%(end-start))

