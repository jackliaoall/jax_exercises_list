import jax
import jax.numpy as jnp

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

xs = jnp.stack([x, x])
ws = jnp.stack([w, w])

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)


auto_batch_convolve = jax.vmap(convolve)
print(auto_batch_convolve(xs, ws))


auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)
wst = jnp.transpose(ws)

print(auto_batch_convolve_v2(xst, wst))

batch_convolve_v3 = jax.vmap(convolve, in_axes=[0, None])
print(batch_convolve_v3(xs, w))
