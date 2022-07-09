import jax
import jax.numpy as jnp
from jax import custom_vjp,custom_jvp
from jax import jit, grad, vmap

@custom_jvp
def f(x):
  return 2*x

# @f.defjvp
# def f_jvp(primals, tangents):
#   x, y = primals
#   x_dot, y_dot = tangents
#   print(x)
#   print(y)
#   primal_out = f(x, y)
#   tangent_out = y * x_dot + x * y_dot
#   return primal_out, tangent_out
#
# #print(grad(f)(2., 3.))
# print(grad(f)(primals = (2.,3.), tangents = (2.,3.)))

# @custom_jvp
# def f(x):
#   return 2 * x
#
# f.defjvps(lambda primals, tangents ,t:  primals )
# print(grad(f)(3.))

@custom_jvp
def f(x):
  return 2*x
@f.defjvp
def f_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  if x >= 0:
    return f(x),x_dot
  else:
    return f(x),2 * x_dot

print(jax.grad(f)(1.))
print(jax.grad(f)(-1.))


