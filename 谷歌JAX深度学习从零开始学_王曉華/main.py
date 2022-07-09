import jax.numpy as jnp

res = jnp.array([0,1])
res = jnp.tile(res,[12,1])
print(res.shape)

print(res)