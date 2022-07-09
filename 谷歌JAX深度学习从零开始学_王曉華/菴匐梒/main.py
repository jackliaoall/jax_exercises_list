import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


layer_sizes = [784, 512, 512, 10]
param_scale = 0.1
step_size = 0.01
num_epochs = 10
batch_size = 128
n_targets = 10
params = init_network_params(layer_sizes, random.PRNGKey(17))


def relu(x):
    return jnp.maximum(0, x)

def softmax(y):
    y_shift = y - jnp.max(y)
    y_exp = jnp.exp(y_shift)
    y_exp_sum = jnp.sum(y_exp)
    return y_exp / y_exp_sum

def predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return softmax(logits)

# batched_predict = vmap(predict, in_axes=(None, 0))
#
# random_flattened_images = random.normal(random.PRNGKey(1), (10, 28 * 28))
# batched_preds = batched_predict(params, random_flattened_images)
# print(batched_preds.shape)

random_flattened_images = random.normal(random.PRNGKey(17), (10,28 * 28))
w = random.normal(random.PRNGKey(17), (256, 784))
def pred(w,xs):
    outputs = jnp.dot(w, xs.T)
    return outputs
pred(w,random_flattened_images)
#pred(w,random_flattened_images)
jax.vmap(pred,[None,0])(w,random_flattened_images)


