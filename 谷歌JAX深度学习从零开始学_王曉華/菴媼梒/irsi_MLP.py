from sklearn.datasets import load_iris
import jax.numpy as jnp
import jax.numpy as jnp
from jax import random,grad
import jax

data = load_iris()
iris_data = jnp.float32(data.data)					#将其转化为float类型的list
iris_target = jnp.float32(data.target)

iris_data = jax.random.shuffle(random.PRNGKey(17),iris_data)
iris_target = jax.random.shuffle(random.PRNGKey(17),iris_target)

def one_hot_nojit(x, k=10, dtype=jnp.float32):
    """ Create a one-hot encoding of x of size k. """
    return jnp.array(x[:, None] == jnp.arange(k), dtype)
iris_target = one_hot_nojit(iris_target)

def Dense(dense_shape = [1, 1]):
  rng = random.PRNGKey(17)

  weight = random.normal(rng, shape=dense_shape)
  bias = random.normal(rng, shape=(dense_shape[-1],))
  params = [weight,bias]

  def apply_fun(inputs,params = params):				#apply_fun是python特性之一，称为内置函数
    W, b = params
    return jnp.dot(inputs, W) + b
  return apply_fun

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

def softmax(x, axis = -1):
  unnormalized = jnp.exp(x)
  return unnormalized / unnormalized.sum(axis, keepdims=True)

def cross_entropy(y_true,y_pred):
    y_true = jnp.array(y_true)
    y_pred = jnp.array(y_pred)
    res = -jnp.sum(y_true*jnp.log(y_pred+1e-7),axis=-1)
    return res

def mlp(x,params):
    a0, b0, a1, b1 = params

    x = Dense()(x, [a0,b0])
    x = jax.nn.tanh(x)
    x = Dense()(x, [a1,b1])
    x = softmax(x,axis=-1)
    return x

def loss_mlp(params, x, y):

    preds = mlp(x,params)
    loss_value = cross_entropy(y,preds)
    return jnp.mean(loss_value)

# 因为我们现在有两层线性层，所以有5个参数，这5个参数需要注入模型中作为起始参数
rng = random.PRNGKey(17)
a0 = random.normal(rng, shape=[4,5])
b0 = random.normal(rng, shape=(5,))

a1 = random.normal(rng, shape=[5,10])
b1 = random.normal(rng, shape=(10,))

params = [a0, b0, a1, b1]

learning_rate = 2.17e-4
for i in range(20000):
    loss = loss_mlp(params,iris_data,iris_target)
    if i % 1000 == 0:

        predict_result = mlp(iris_data, params)
        predicted_class = jnp.argmax(predict_result, axis=1)
        _iris_target = jnp.argmax(iris_target, axis=1)
        accuracy = jnp.sum(predicted_class == _iris_target) / len(_iris_target)
        print("i:",i,"loss:",loss,"accuracy:",accuracy)


    params_grad = grad(loss_mlp)(params,iris_data,iris_target)
    params = [
    (p - g * learning_rate) for p, g in zip(params, params_grad)
    ]




predict_result = mlp(iris_data, params)
predicted_class = jnp.argmax(predict_result, axis=1)
iris_target = jnp.argmax(iris_target, axis=1)
accuracy =  jnp.sum(predicted_class == iris_target)/len(iris_target)
print(accuracy)