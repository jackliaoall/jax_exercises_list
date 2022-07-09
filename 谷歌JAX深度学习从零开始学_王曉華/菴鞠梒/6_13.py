import jax
import time
import jax.numpy as jnp


key = jax.random.PRNGKey(17)
xs = jax.random.normal(key,(1000,))
a = 0.929
b = 0.214
ys = a * xs + b


params = jax.random.normal(key,(2,))
print(params)

@jax.jit
def model(params,x):
    a = params[0];b = params[1]
    y = a * x + b
    return y

@jax.jit
def loss_fn(params, x, y):
  prediction = model(params, x)
  return jnp.mean((prediction-y)**2)

@jax.jit
def update(params, x, y, lr=1e-3):
  return params - lr * jax.grad(loss_fn)(params, x, y)

start = time.time()
for i in range(4000):
    params = update(params, xs, ys)
    if (i+1) %500 == 0:
        loss_value = loss_fn(params,xs,ys)
        end = time.time()
        print("循环运行时间:%.12f秒" % (end - start),f"经过i轮:{i}，现在的loss值为:{loss_value}")
        start = time.time()

print(params)


