import jax
import jax.numpy as jnp


def f(x, y):
  print("Running f():")
  print(f"  x = {x}")
  print(f"  y = {y}")
  result = jnp.dot(x + 1, y + 1)
  print(f"  result = {result}")
  return result

key = jax.random.PRNGKey(17)
x = jax.random.normal(key,shape=[5,3])
y = jax.random.normal(key,shape=[3,4])

# f(x,y)
# print("-------------------------------")
# jax.jit(f)(x,y)

@jax.jit
def f(x, neg):
  return -x if neg else x

#f(1, True)

from functools import partial

@partial(jax.jit, static_argnums=(1,))
def f(x, neg):
  return -x if neg else x

print(f(1, True))
