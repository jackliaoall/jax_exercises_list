import jax
from jax import random
import jax.numpy as jnp

def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

#print(jax.grad(f)(2.))
#print(jax.grad(f)(3.))

#f_jit = jax.jit(f)
#print(f_jit(2.))


#f = jax.jit(f, static_argnums=(0,))
#print(f(2.))

def example_fun(length, val):
  return jnp.ones((length,)) * val

#print(example_fun(5, 4))
#jit_example_fun = jax.jit(example_fun)  #这个是错误的
jit_example_fun = jax.jit(example_fun,static_argnums=(0,))
#print(jit_example_fun(5,4))


def example_fun(length_list, val):
  return length_list * val
jit_example_fun = jax.jit(example_fun)
print(jit_example_fun((jnp.array([1,1,1,1])),4))