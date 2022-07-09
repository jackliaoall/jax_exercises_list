import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(17)
params = dict(weight=jax.random.normal(key,(2,2)),biases=jax.random.normal(key+1,(2,)))
print(params)
print(jax.tree_map(lambda x: x.shape, params))














