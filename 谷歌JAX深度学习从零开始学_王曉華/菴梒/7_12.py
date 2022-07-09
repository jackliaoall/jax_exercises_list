import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(17)
xs = jnp.linspace(0,9,10)
print("xs:",xs)
kernel = jnp.ones(3)/10
print("kernel:",kernel)
y_smooth = jnp.convolve(xs, kernel)
print("y_smooth:",y_smooth)


import jax.scipy as jsp
img = jax.random.normal(key,shape=(128,128,3))
kernerl_2d = jnp.array([[[0,1,0],[1,0,1],[0,1,0]]])
smooth_image = jsp.signal.convolve(img, kernerl_2d, mode='same')
print(smooth_image)