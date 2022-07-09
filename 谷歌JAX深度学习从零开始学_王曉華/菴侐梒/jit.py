import time
import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

rng = jax.random.PRNGKey(17)
x = jax.random.normal(rng, (100000000,))

start = time.time()
selu(x)
end = time.time()
print("循环运行时间:%.2f秒"%(end-start))

selu_jit = jax.jit(selu)
start = time.time()
selu_jit(x)
end = time.time()
print("循环运行时间:%.10f秒"%(end-start))

@jax.jit
def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
start = time.time()
selu(x)
end = time.time()
print("循环运行时间:%.2f秒"%(end-start))
