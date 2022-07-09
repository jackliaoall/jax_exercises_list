from jax import lax
import time
import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def dropout(x,keep_rate = 0.5):
    binary_tensor = np.random.rand(*x.shape) < keep_rate

    x = jnp.divide(x,keep_rate)*binary_tensor
    return x

@jax.jit
def batch_normalization(x,gamma = 0.9,beta = 0.25,eps = 1e-12):

    mean_x = jnp.mean(x, axis=(0, 1, 2))
    mean_x = jnp.reshape(mean_x, (1, 1, 1, -1))

    std_x = jnp.std(x, axis=(0, 1, 2))
    std_x = jnp.reshape(std_x, (1, 1, 1, -1)) + eps

    y = (x - mean_x) / std_x
    return gamma * y + beta


@jax.jit
def batch_pooling(inputMap, pool_size=2, stride=2):
    assert inputMap.ndim == 4, print("输入必须为4维")
    def pooling(feature_map, pool_size=pool_size, stride=stride):
        feature_map_shape = feature_map.shape
        height = feature_map_shape[0]
        width = feature_map_shape[1]
        padding_height = (round((height - pool_size + 1) / stride))
        padding_width = (round((width - pool_size + 1) / stride))

        pool_out = jnp.zeros((padding_height, padding_width))

        out_height = 0
        for r in range(0, height, stride):
            out_width = 0
            for c in range(0, width, stride):
                pool_out = pool_out.at[out_height, out_width].set(jnp.max(feature_map[r:r + pool_size, c:c + pool_size]))
                out_width = out_width + 1
                out_height = out_height + 1
        return pool_out

    image = jnp.einsum("bhwc -> bchw",inputMap)
    output_pooling = (jax.vmap(jax.vmap(pooling)))(image)
    output_pooling = jnp.einsum("bchw -> bhwc",output_pooling)

    return output_pooling




if __name__ == '__main__':

    image = jax.random.normal(jax.random.PRNGKey(17), (10, 28, 28, 64))
    output_pooling = batch_pooling(image)
    print(output_pooling.shape)