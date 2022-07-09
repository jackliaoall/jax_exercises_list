import jax.numpy as jnp

print(jnp.add(1, 1.0))

from jax import lax

print(lax.add(1, 1))

print(lax.add(jnp.float32(1), 1.0))

#print(lax.add(1, 1.0))




x = jnp.array([1, 2, 1])
y = jnp.ones(10)
print(x)
print(y)

print(jnp.convolve(x, y))