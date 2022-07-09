import jax.numpy as jnp

kernel = jnp.array([[0,0,0],[0,0,0],[0,0,0]])

for i in range(3):
    for j in range(3):
        kernel = kernel.at[i,j].set(i + j)
print(kernel)

