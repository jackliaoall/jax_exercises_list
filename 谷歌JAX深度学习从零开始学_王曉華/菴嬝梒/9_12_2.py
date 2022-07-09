from jax import custom_vjp

import jax

@custom_vjp
def f(x, y):
  return x * y

def f_fwd(x, y):
  return f(x, y), (y, x)  #定义正向计算函数以及每个参数的倒函数

def f_bwd(res,g):
  y, x = res              #定义求导结果
  return (y, x)

f.defvjp(f_fwd, f_bwd)    #在自定义的函数中注册正向求导和反向求导函数

print(jax.grad(f)(2., 3.))
print(jax.grad(f,[0,1])(2., 3.))