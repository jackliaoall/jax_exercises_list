import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    print(f"x = {x}")
    print(f"x.shape = {x.shape}")
    print(f"jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")

x = jnp.ones((2, 3))
f(x)



