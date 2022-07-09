import jax
import jax.numpy as jnp

array = jnp.arange(9)
print(array)
print(array[-1])
print(array[11])

print(jnp.sum(jnp.arange(9)))
#print(jnp.sum(range(9)))

print(jnp.sum(jnp.array(jnp.arange(9))))