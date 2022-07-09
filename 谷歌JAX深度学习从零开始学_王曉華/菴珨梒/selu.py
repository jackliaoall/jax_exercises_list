import jax.numpy as jnp
from jax import random

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

key = random.PRNGKey(17)
x = random.normal(key, (5,))
print(selu(x))