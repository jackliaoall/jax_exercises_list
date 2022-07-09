import jax.nn
import jax.numpy as jnp


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def relu(x):
    return jnp.maximum(0, x)

def softmax(y):
    y_shift = y - jnp.max(y)
    y_exp = jnp.exp(y_shift)
    y_exp_sum = jnp.sum(y_exp)
    return y_exp / y_exp_sum


if __name__ == '__main__':
    y_pred = jnp.array([[4.0, 2.0, 1.0],[4.0, 2.0, 1.0]])
    y_pred_softmax = softmax(y_pred)
    print(y_pred_softmax)
    print(jax.nn.softmax((y_pred),axis=-1))