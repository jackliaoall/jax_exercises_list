import jax.nn
import jax.numpy as jnp

def softmax(x, axis = -1):
  unnormalized = jnp.exp(x)
  return unnormalized / unnormalized.sum(axis, keepdims=True)

arr = jnp.array([[3,1,-3]])
print(softmax(arr))
print(jax.nn.softmax(arr))
from jax.experimental import stax

a = jnp.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
y = jnp.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])



def cross_entropy(y_true, y_pred):
  res = -jnp.sum(jnp.nan_to_num(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred)))
  return res

print(cross_entropy(y,a))