import jax
from jax import lax
from jax._src import api

def multiply_add_lax(x, y, z):
  #使用了jax.lax中自带的函数
  return lax.add(lax.mul(x, y), z)

def square_add_lax(a, b):
  #使用了自定义的函数
  return multiply_add_lax(a, a, b)

#使用grad计算函数的微分
print("square_add_lax = ", square_add_lax(2., 10.))
print("grad(square_add_lax) = ", api.grad(square_add_lax, argnums= [0])(2.0, 10.))
print("grad(square_add_lax) = ", jax.grad(square_add_lax, argnums=[0,1])(2.0, 10.))



