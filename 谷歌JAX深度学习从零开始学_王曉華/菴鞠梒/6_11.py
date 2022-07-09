import jax
import jax.numpy as jnp

def body_fun(x):
    return x**2

#print(jax.grad(body_fun)(1.))

#print(jax.value_and_grad(body_fun)(1.))

def body_fun(x,y):
    return x*y

grad_body_fun = jax.grad(body_fun)
x = (2.)
y = (3.)
#print(grad_body_fun(x,y))

dx,dy = (jax.grad(body_fun, argnums=(0, 1))(x, y))
#print(f"dx:{dx}")
#print(f"dy:{dy}")

def body_fun(x,y,z):
    return x*y*z

grad_body_fun = jax.grad(body_fun)
x = (2.)
y = (3.)
z = (4.)

#print((jax.grad(body_fun, argnums=(0, 1))(x, y, z)))


def body_fun(x,y,z):
    return x*y*z

grad_body_fun = jax.grad(body_fun)
x = (2.)
y = (3.)
z = (4.)
#print((jax.grad(body_fun, argnums=(0, 1))(x, y, z)))

def body_fun(x,y):
    return x*y,x**2+y**2
grad_body_fun = jax.grad(body_fun)
x = (2.)
y = (3.)
print((jax.grad(body_fun,has_aux=True)(x, y)))



# def sum_squared_error(x, y):
#   return jnp.sum((x-y)**2)
# sum_squared_error_dx = jax.grad(sum_squared_error)
# x = jnp.asarray([0.1, 0.1, 1.1, 1.1])
# y = jnp.asarray([1.1, 2.1, 3.1, 4.1])
# print(sum_squared_error_dx(x, y))







