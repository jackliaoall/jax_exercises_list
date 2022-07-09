import jax.nn
import jax.numpy as jnp

jax_array = jnp.arange(10)
#
# print(jax_array)
# print(jax_array[17])


new = (jax_array.broadcast(sizes=[2, 3,]))
print(new.shape)

jax.nn.softmax(new)