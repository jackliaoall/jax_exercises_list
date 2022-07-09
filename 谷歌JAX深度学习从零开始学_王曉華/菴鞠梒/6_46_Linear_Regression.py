import jax
import jax.numpy as jnp
from typing import NamedTuple

class Params(NamedTuple):
    weight: jnp.ndarray
    bias: jnp.ndarray

def model(params:Params,xs):
    pred = params.weight * xs + params.bias
    return pred

def init(key):
  weight = jax.random.normal(key, (1,))
  bias = jax.random.normal(key + 1, (1,))
  return Params(weight, bias)

def loss(params:Params,xs,y_true):
    y_pred = model(params,xs)
    loss = (y_pred-y_true)**2
    return jnp.mean(loss)

def opt_sgd(params,xs,y_true,learn_rate = 1e-5):
    grad = jax.grad(loss)(params,xs,y_true)
    params = jax.tree_multimap(lambda par,grd:par - learn_rate *grd,params,grad)
    return params

key = jax.random.PRNGKey(17)
xs = jax.random.normal(key,(10000,1))
a = 0.929
b = 0.214
ys = a * xs + b

params = init(key)
for i in range(40000):
    params = opt_sgd(params,xs,ys)

print(params)

