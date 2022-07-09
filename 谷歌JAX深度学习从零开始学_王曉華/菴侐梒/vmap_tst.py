import jax
import jax.numpy as jnp
rng = jax.random.PRNGKey(17)

mat = jax.random.normal(rng, (150, 100))
batched_x = jax.random.normal(rng, (10, 100))

def apply_matrix(v):
  return jnp.dot(mat, v)

def naively_batched_apply_matrix(v_batched):

  result = []
  for v in v_batched:
    res = apply_matrix(v)
    result.append(res)

  return jnp.stack(result)

naively_batched_apply_matrix(batched_x)

