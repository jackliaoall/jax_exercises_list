from jax import lax
import time
import jax
import jax.numpy as jnp
import conv_untils

x_train = jnp.load("../第一章/mnist_train_x.npy")
y_train = jnp.load("../第一章/mnist_train_y.npy")

x_train = lax.expand_dims(x_train,[-1])/255.
def one_hot_nojit(x, k=10, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)
y_train = one_hot_nojit(y_train)

batch_size = 312
image_channel_dimension = 1

x_test = x_train[-4096:]
y_test = y_train[-4096:]

x_train = x_train[:60000-4096]
y_train = y_train[:60000-4096]

batch_size = 320
img_shape = [batch_size,28,28,image_channel_dimension]  # shape=[N,H,W,C]
kernel_shape = [3,3,image_channel_dimension,image_channel_dimension]    #shape = [H,W,I,O]


#这里实际上就是更新kernel参数
def init_mlp_params(cnn_layers = 2):
    params = []
    key = jax.random.PRNGKey(17)
    #首先创建3层的CNN_layer使用的kernel
    for i in range(cnn_layers):
        kernel_weight = jax.random.normal(key, shape=kernel_shape) / jnp.sqrt(784)
        par_dict = dict(kernel_weight=kernel_weight)
        params.append(par_dict)
    weight = jax.random.normal(key, shape=(28*28, 10)) / jnp.sqrt(784)
    bias = jax.random.normal(key, shape=(10,)) / jnp.sqrt(784)
    par_dict = dict(weight=weight, bias=bias)
    params.append(par_dict)
    return params

params = init_mlp_params()

dn = lax.conv_dimension_numbers(img_shape,kernel_shape,('NHWC', 'HWIO', 'NHWC'))

@jax.jit
def forward(params, x):
    params_length = len(params)
    for i in range(params_length - 1):
        par = params[i]
        x = lax.conv_general_dilated(x, par["kernel_weight"], window_strides=[1, 1]
                                       , padding="SAME", dimension_numbers=dn)
        x = conv_untils.batch_normalization(x)
        x = jax.nn.selu(x)


    x = lax.reshape(x,[x.shape[0],28*28])
    x = jnp.matmul(x, params[-1]["weight"]) + params[-1]["bias"]
    x = jax.nn.softmax(x,axis=-1)
    return x

@jax.jit
def cross_entropy(y_true, y_pred):
    ce = -jnp.sum(y_true * jnp.log(jax.numpy.clip(y_pred, 1e-9, 0.999)) + (1 - y_true) * jnp.log(jax.numpy.clip(1 - y_pred, 1e-9, 0.999)), axis=1)
    return jnp.mean(ce)

@jax.jit
def loss_fun(params,xs,y_true):
    y_pred = forward(params,xs)
    return cross_entropy(y_true,y_pred)

@jax.jit
def opt_sgd(params,xs,ys,learn_rate = 1e-3):
    grads = jax.grad(loss_fun)(params,xs,ys)
    return jax.tree_multimap(lambda p, g: p - learn_rate * g, params, grads)

@jax.jit
def pred_check(params, inputs, targets):
    """ Correct predictions over a minibatch. """
    # 这里我做了修正，因为预测生成的结果是[-1,10],而输入的target也被我改成了[-1,10],
    # 所以这里需要2个jnp.argmax做一个转换。
    predict_result = forward(params, inputs)
    predicted_class = jnp.argmax(predict_result, axis=1)
    targets = jnp.argmax(targets, axis=1)
    return jnp.sum(predicted_class == targets)

start = time.time()
for i in range(200):
    batch_num = (60000-4096)//batch_size
    for j in range(batch_num):
        start = batch_size * (j)
        end = batch_size * (j + 1)

        x_batch = x_train[start:end]
        y_batch = y_train[start:end]
        params = opt_sgd(params,x_batch,y_batch)

    if (i+1) %50 == 0:
        loss_value = loss_fun(params,x_train,y_train)
        end = time.time()
        train_acc = (pred_check(params,x_test,y_test) / float(4096.))
        print("循环运行时间:%.12f秒" % (end - start),f"经过i轮:{i}，现在的loss值为:{loss_value},测试集准确率为: {train_acc}")
        start = time.time()
