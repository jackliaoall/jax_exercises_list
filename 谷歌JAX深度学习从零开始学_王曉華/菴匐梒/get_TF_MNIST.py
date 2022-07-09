import jax.numpy as jnp
import tensorflow_datasets as tfds
import jax,time


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


train_ds = tfds.load("mnist", split=tfds.Split.TRAIN, batch_size=-1)    #这里的TRAIN可以换成TEST
train_ds = tfds.as_numpy(train_ds)
train_images, train_labels = train_ds["image"], train_ds["label"]

_,hight_size,width_size,channel_dimmision = train_images.shape
num_pixels = hight_size * width_size * channel_dimmision
output_dimisions = 10

train_images = jnp.reshape(train_images,(-1,num_pixels))
train_labels = one_hot(train_labels,k = output_dimisions)


test_ds = tfds.load("mnist", split=tfds.Split.TEST, batch_size=-1)    #这里的TRAIN可以换成TEST
test_ds = tfds.as_numpy(test_ds)
test_images, test_labels = test_ds["image"], test_ds["label"]

test_images = jnp.reshape(test_images,(-1,num_pixels))
test_labels = one_hot(test_labels,k = output_dimisions)


def init_params(layer_dimisions = [num_pixels,512,256,output_dimisions]):
    key = jax.random.PRNGKey(17)
    params =  []
    for i in range(1,(len(layer_dimisions))):
        weight = jax.random.normal(key,shape=(layer_dimisions[i - 1],layer_dimisions[i]))/jnp.sqrt(num_pixels)
        bias = jax.random.normal(key,shape=(layer_dimisions[i],))/jnp.sqrt(num_pixels)
        par = {"weight":weight,"bias":bias}
        params.append(par)
    return params

def forward(params,xs):
    for par in params[:-1]:
        weight = par["weight"]
        bias = par["bias"]

        xs = jnp.dot(xs, weight) + bias
        xs = relu(xs)
    output = jnp.dot(xs, params[-1]["weight"]) + params[-1]["bias"]
    output = jax.nn.softmax(output,axis=-1)
    return output

@jax.jit
def relu(x):							#激活函数
    return jnp.maximum(0, x)

@jax.jit
def cross_entropy(y_true, y_pred):		#交叉熵函数
    ce = -jnp.sum(y_true * jnp.log(jax.numpy.clip(y_pred, 1e-9, 0.999)) + (1 - y_true) * jnp.log(jax.numpy.clip(1 - y_pred, 1e-9, 0.999)), axis=1)
    return jnp.mean(ce)

@jax.jit								#计算损失函数
def loss_fun(params,xs,y_true):
    y_pred = forward(params,xs)
    return cross_entropy(y_true,y_pred)

@jax.jit								#sgd优化函数
def opt_sgd(params,xs,y_true,learn_rate = 1e-3):
    grads = jax.grad(loss_fun)(params,xs,y_true)
    params = jax.tree_multimap(lambda p,g:p - learn_rate*g ,params,grads )
    return params

@jax.jit								#准确率计算函数
def pred_check(params, inputs, targets):
    predict_result = forward(params, inputs)
    predicted_class = jnp.argmax(predict_result, axis=1)
    targets = jnp.argmax(targets, axis=1)
    return jnp.sum(predicted_class == targets)


params = init_params()
start = time.time()
for i in range(500):
    params = opt_sgd(params,train_images,train_labels)

    if (i+1) %50 == 0:
        loss_value = loss_fun(params,test_images,test_labels)
        end = time.time()

        train_acc = (pred_check(params,test_images,test_labels) / float(10000.))
        print("循环运行时间:%.12f秒" % (end - start),f"经过i轮:{i}，现在的loss值为:{loss_value},测试集测试集准确率为: {train_acc}")
        start = time.time()









