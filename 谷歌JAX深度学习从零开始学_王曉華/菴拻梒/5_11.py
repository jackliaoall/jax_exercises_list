import jax.numpy as jnp
import jax.random

x_jnp = jnp.linspace(0, 9, 10)
print(x_jnp)

import jax.numpy as jnp
import jax.random
key = jax.random.PRNGKey(17)
mat_a = jax.random.normal(key,shape=[2,3])
mat_b = jax.random.normal(key,shape=[3,1])

print(jax.numpy.matmul(mat_a,mat_b))
print(jax.numpy.dot(mat_a,mat_b))

import jax.numpy as jnp
import jax.random

x_jnp = jnp.linspace(0, 9, 10)
print(type(x_jnp))
#x_jnp[0] = 17
print(x_jnp)


import jax.numpy as jnp
import jax.random

x_jnp = jnp.linspace(0, 9, 10)
y_jnp = x_jnp.at[0].set(17)
print(f"x_jnp:{x_jnp}")
print(f"y_jnp:{y_jnp}")



