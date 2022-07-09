import jax.numpy as jnp
from jax import jit, grad, vmap

# def logxp(x):
#   return jnp.log(1. + jnp.exp(x))
#
# print(jit(logxp)(3.))
# print(jit(grad(logxp))(3.))
# print(vmap(jit(grad(logxp)))(jnp.arange(4.)))
#
# print((grad(logxp))(99.))

# from jax import custom_jvp
#
# @custom_jvp
# def log1pexp(x):
#   return jnp.log(1. + jnp.exp(x))
#
# @log1pexp.defjvp
# def log1pexp_jvp(primals, tangents):
#   x, = primals
#   x_dot, = tangents
#   ans = log1pexp(x)
#   ans_dot = (1 - 1/(1 + jnp.exp(x))) * x_dot
#   return ans, ans_dot


def logxp(x):
  return jnp.log(1. + jnp.exp(x))

print(jit(logxp)(3.))
print(jit(grad(logxp))(3.))
print(vmap(jit(grad(logxp)))(jnp.arange(4.)))

print((grad(logxp))(50.))