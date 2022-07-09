import jax
import jax.numpy as jnp

jax_array = jnp.zeros((3,3), dtype=jnp.float32)

#print(jax_array)

from jax.ops import index, index_add, index_update,index_max,index_mul
print("原始数组:",jax_array)

print("-----------------------------")
new_jax_array = index_update(jax_array, index[1, :], 1.)
print("new_jax_array:",new_jax_array)

print("-----------------------------")
new_add_jax_array = index_add(jax_array,index[1,:],1.)
print("new_add_jax_array:",new_add_jax_array)

print("-----------------------------")
max_jax_array = index_max(jax_array,index[1,:],-1)
print("neg_max_jax_array:",max_jax_array)
max_jax_array = index_max(jax_array,index[1,:],1)
print("pos_max_jax_array:",max_jax_array)

print("-----------------------------")
mul_jax_array = index_mul(jax_array,index[1,:],2)
print("mul_jax_array:",mul_jax_array)