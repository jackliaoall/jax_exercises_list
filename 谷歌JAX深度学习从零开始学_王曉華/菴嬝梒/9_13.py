import jax
import jax.numpy as jnp
from jax import custom_vjp,custom_jvp
from jax import jit, grad, vmap


@custom_jvp
def f(x):
  return (x)

def f_jvp(primals, tangents):
  x, = primals
  t, = tangents
  return f(x),t*x

f.defjvp(f_jvp)

print(f(3.))
print(jax.grad(f)(2.))

y, y_dot = jax.jvp(f, (3.,), (2.,))
print(y,y_dot)




