import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.PRNGKey(0), (5000, 5000))


def f(x):
    a = 2.
    return x + a


#fast_f = jax.jit(f)
#print(jax.make_jaxpr(f)(1.0, 2.0, 3.0))


def examine_jaxpr(closed_jaxpr):
    jaxpr = closed_jaxpr.jaxpr
    print("invars:", jaxpr.invars)
    print("outvars:", jaxpr.outvars)
    print("constvars:", jaxpr.constvars)
    for eqn in jaxpr.eqns:
        print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
    print()
    print("jaxpr:", jaxpr)


#print(examine_jaxpr(jax.make_jaxpr(f)(1.0)))


closed_jaxpr = jax.make_jaxpr(f)(1.0)
print(closed_jaxpr)
print(closed_jaxpr.literals)

