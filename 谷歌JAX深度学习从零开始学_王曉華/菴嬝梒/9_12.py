import jax
from jax import custom_jvp

# def f(x, y):
#   return x * y
#print(f(2., 3.))
#print(jax.grad(f)(2., 3.))

@custom_jvp
def f(x, y):
  return x * y

@f.defjvp
def f_jvp(primals, tangents):
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = f(x, y)
  tangent_out = y_dot + x_dot
  return primal_out, tangent_out

print("经过JVP自定义的f函数：",jax.grad(f,argnums=[0,1])(2., 3.))

def f(x, y):
  return x * y
print("原始JAX求导函数：",jax.grad(f,argnums=[0,1])(2., 3.))