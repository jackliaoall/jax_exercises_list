from sklearn.datasets import load_iris
import jax.numpy as jnp
import jax.numpy as jnp
from jax import random,grad


data = load_iris()

iris_data = jnp.float32(data.data)					#将其转化为float类型的list
iris_target = jnp.float32(data.target)


def Dense(dense_shape = [4, 1]):
  def init_fun(input_shape = dense_shape):
    rng = random.PRNGKey(17)
    W, b = random.normal(rng, shape=input_shape), random.normal(rng, shape=(input_shape[-1],))
    return (W, b)
  def apply_fun(inputs,params):
    W, b = params
    return jnp.dot(inputs, W) + b
  return init_fun, apply_fun

init_fun, apply_fun = Dense()
params = init_fun()
def loss_linear(params, x, y):
    """loss function:
    g(x) = (f(x) - y) ** 2
    """
    preds = apply_fun(x,params)
    return jnp.mean(jnp.power(preds - y, 2.0))

learning_rate = 0.005  # 学习率
N = 1000  # 梯度下降的迭代次数

for i in range(N):
    # 计算损失
    loss = loss_linear(params,iris_data, iris_target)
    if i % 100 == 0:
        print(f'i: {i}, loss: {loss}')
    # 计算梯度
    params_grad = grad(loss_linear)(params,iris_data, iris_target)

    params = [
        # 对每个参数，加上学习率乘以负导数
        (p - g * learning_rate) for p, g in zip(params, params_grad)
    ]

print(f'i: {N}, loss: {loss}')




