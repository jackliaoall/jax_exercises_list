import os

import jax.nn
import jax.numpy as jnp
from jax import lax
from jax.nn import one_hot, relu

# def forward(w1,w2,images):
#     hiddens_1 = relu(jnp.dot(images, w1))
#     hiddens_2 = jnp.dot(hiddens_1, w2)
#     logits = jax.nn.softmax(hiddens_2)
#     return logits
#
# def loss(w1, w2, images, labels):
#   predictions = forward(w1, w2, images)
#   targets = one_hot(labels, predictions.shape[-1])
#   losses = jnp.sum(targets * predictions, axis=1)
#   return -jnp.mean(losses, axis=0)
#
# w1 = jnp.zeros((784, 512))
# w2 = jnp.zeros((512, 10))
# images = jnp.zeros((128, 784))
# labels = jnp.zeros(128, dtype=jnp.int32)
#
# print(loss(w1, w2, images, labels))

def named_predict(w1, w2, image):
    hidden = relu(lax.pdot(image, w1, ['inputs']))
    logits = lax.pdot(hidden, w2, ['hidden_1'])
    return jax.nn.softmax(logits, ['classes'])

def named_loss(w1, w2, images, labels):
    predictions = named_predict(w1, w2, images)
    targets = one_hot(labels, 10, axis=['classes'])
    losses = lax.psum(targets * predictions, ['classes'])
    return -lax.pmean(losses, ['batch'])

from jax.experimental.maps import xmap

w1 = jnp.zeros((784, 512))
w2 = jnp.zeros((512, 10))
images = jnp.zeros((128, 784))
labels = jnp.zeros(128, dtype=jnp.int32)
in_axes = [
    ['inputs', 'hidden_1'],
    ['hidden_1', 'classes'],
    ['batch', 'inputs'],
    ['batch',...]]

loss = xmap(named_loss, in_axes=in_axes, out_axes=[...])
print(loss(w1, w2, images, labels))
