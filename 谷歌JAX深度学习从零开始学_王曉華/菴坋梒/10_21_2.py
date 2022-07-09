import jax
import jax.numpy as jnp
from jax.experimental import sparse
key = jax.random.PRNGKey(17)
num_classes = 10                                #设置10种类别
classes_list = jnp.arange(num_classes)          #生成类别序列
x_list = []
y_list = []

for i in range(1024):
    x = jax.random.choice((key + i),classes_list,shape=(1,))[0]     #随机生成数据
    x_onehot = jax.nn.one_hot(x,num_classes=num_classes)            #转化成one_hot形式
    x_list.append(x_onehot)
    y_list.append(x)

params = [jax.random.normal(key,shape=(num_classes,1)),jax.random.normal(key,shape=(1,))]   #生成模型参数
sparsed_x = sparse.BCOO.fromdense(jnp.array(x_list))       #将数据转化成稀疏矩阵
y_list = jnp.array(y_list)


# print(sparsed_x.data)
# print(sparsed_x.indices)


def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

def y_model(params, X):
    output = (jnp.dot(X, (params[0])) + params[1])
    return sigmoid(output)

def loss(params, sparsed_x, y):
    sparsed_y_model = sparse.sparsify(y_model)
    y_hat = sparsed_y_model(params, sparsed_x)
    return -jnp.mean(y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat))

learning_rate = 1e-3

print(loss(params,sparsed_x,y_list))

for i in range(100):
    params_grad = jax.grad(loss)(params,sparsed_x,y_list)
    params = [(p - g * learning_rate) for p, g in zip(params, params_grad)]

print(loss(params,sparsed_x,y_list))



