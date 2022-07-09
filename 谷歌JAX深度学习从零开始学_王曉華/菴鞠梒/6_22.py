import jax
import time
import jax.numpy as jnp

def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

f_jit = jax.make_jaxpr(f,static_argnums=(0,))
print(f_jit(10.))

# f_jit = jax.jit(f)
# print(f_jit(10.))

#f_grad = jax.grad(f)
#print(f_grad(10.))





from jax import lax
def f(x):
  result = lax.cond(x>0,lambda x:x,lambda x:2*x,x)
  return result
f_jit = jax.jit(f)
print(f_jit(10.))
print(jax.make_jaxpr(f)(10.))



