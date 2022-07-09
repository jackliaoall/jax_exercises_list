from jax.experimental import sparse
import jax.numpy as jnp
import numpy as np

array = jnp.array([[0., 1., 0., 2.],
               [3., 0., 0., 0.],
               [0., 0., 4., 0.]])
sparsed_array = sparse.BCOO.fromdense(array)

#print(sparsed_array)
#print(sparsed_array.todense())
#print(sparsed_array.data)
# print(sparsed_array.indices)
#
# for i,j in zip(sparsed_array.indices[0],sparsed_array.indices[1]):
#     print(array[i,j])

# print(sparsed_array.ndim)       #原始矩阵矩阵的维度个数
# print(sparsed_array.shape)      #原始矩阵的维度大小
# print(sparsed_array.dtype)      #原始矩阵的数据类型
# print(sparsed_array.nse)        #原始矩阵中不为0的数据个数

# dot_array = jnp.array([[1.],[2.],[2.]])
# print(sparsed_array.T)
# print(sparsed_array.T@dot_array)
# print((jnp.dot(sparsed_array.T.todense(),dot_array)))

from jax import grad, jit


def f(sparsed_array,dot_array):
    return (jnp.dot(sparsed_array.T,dot_array))

dot_array = jnp.array([[1.],[2.],[2.]])
#(f(sparsed_array,dot_array))
f_sp = sparse.sparsify(f)
print(f_sp(sparsed_array,dot_array))
