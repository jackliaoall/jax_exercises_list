import jax
import jax.numpy as jnp
from jax import lax
from functools import wraps
from jax import core
from jax._src.util import safe_map

#确认需要被追踪的函数
inverse_registry = {}
inverse_registry[lax.exp_p] = jnp.log
inverse_registry[lax.tanh_p] = jnp.arctanh

#提供后向遍历的方案
def inverse_jaxpr(jaxpr, consts, *args):
    env = {}

    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    # 参数被写入到Jaxpr outvars
    write(core.unitvar, core.unit)
    safe_map(write, jaxpr.outvars, args)
    safe_map(write, jaxpr.constvars, consts)

    # 向后遍历
    for eqn in jaxpr.eqns[::-1]:
        invals = safe_map(read, eqn.outvars)
        if eqn.primitive not in inverse_registry:
            raise NotImplementedError("{} does not have registered inverse.".format(
                eqn.primitive
            ))
        outval = inverse_registry[eqn.primitive](*invals)
        safe_map(write, eqn.invars, [outval])
    return safe_map(read, jaxpr.invars)

#在程序中建立后向遍历
def inverse(fun):
  @wraps(fun)
  def wrapped(*args, **kwargs):
    closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
    out = inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
    return out[0]
  return wrapped

def f(x):
    return jnp.exp(jnp.tanh(x))

print(jax.make_jaxpr(f)(1.))

print("-----------------")
f_inv = inverse(f)
print(jax.make_jaxpr(inverse(f))(f(1.)))



