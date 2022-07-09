import jax
from jax import lax
import jax.numpy as jnp

# kernel = jnp.zeros(shape=(10,3,3,3),dtype=jnp.float32)
#
# img = jnp.zeros((1, 200, 198, 3), dtype=jnp.float32)
#
# print("kernel shape:",kernel.shape)
# print("img shape:",img.shape)
#
# out = lax.conv(jnp.transpose(img,[0,3,1,2]),kernel,window_strides=[1,1],padding="SAME")
# print("out shape:",out.shape)
#
# out = lax.conv_general_dilated(jnp.transpose(img,[0,3,1,2]),kernel,window_strides=[2,2],padding="SAME")
# print("out shape:",out.shape)

img = jnp.zeros((1, 200, 200, 3), dtype=jnp.float32)
kernel = jnp.zeros(shape=(3,3,3,10),dtype=jnp.float32)
dn = lax.conv_dimension_numbers(img.shape,     # only ndim matters, not shape
                                kernel.shape,  # only ndim matters, not shape
                                ('NHWC', 'HWIO', 'NCHW'))  # the important bit
print(dn)

out = lax.conv_general_dilated(img,kernel,window_strides=[2,2],padding="SAME",dimension_numbers=dn)
print("dimension numbers out shape:",out.shape)



