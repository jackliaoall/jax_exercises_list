import jax
import time
import jax.numpy as jnp

layers_shape = [1, 64, 128, 1]
key = jax.random.PRNGKey(17)

def init_mlp_params(layers_shape):
    params = []
    for n_in, n_out in zip(layers_shape[:-1], layers_shape[1:]):
        weight = jax.random.normal(key,shape=(n_in,n_out))/128.
        bias = jax.random.normal(key,shape=(n_out,))/128.
        par_dict = dict(weight=weight,bias=bias)
        params.append(par_dict)
    return params

params = init_mlp_params(layers_shape)
#print(jax.tree_map(lambda x:x.shape,params))

@jax.jit
def forward(params, x):
    for par in params:
        x = jnp.matmul(x,par["weight"]) + par["bias"]
    return x

@jax.jit
def loss_fun(params,xs,y_true):
    y_pred = forward(params,xs)
    return jnp.mean((y_pred-y_true)**2)

@jax.jit
def opt_sgd(params,xs,ys,learn_rate = 1e-1):
    grads = jax.grad(loss_fun)(params,xs,ys)
    return jax.tree_multimap(lambda p, g: p - learn_rate * g, params, grads)

@jax.jit
def opt_sgd2(params,xs,ys,learn_rate = 1e-3):
    grads = jax.grad(loss_fun)(params,xs,ys)
    new_params = []
    for par,grd in zip(params,grads):
        new_weight = par["weight"]-learn_rate*grd["weight"]
        new_bias = par["bias"] - learn_rate*grd["bias"]
        par_dict = dict(weight=new_weight, bias=new_bias)
        new_params.append(par_dict)
    return new_params

key = jax.random.PRNGKey(17)
xs = jax.random.normal(key,(1000,1))
a = 0.929
b = 0.214
ys = a * xs + b

start = time.time()
for i in range(4000):
    params = opt_sgd2(params,xs,ys)
    if (i+1) %500 == 0:
        loss_value = loss_fun(params,xs,ys)
        end = time.time()
        print("循环运行时间:%.12f秒" % (end - start),f"经过i轮:{i}，现在的loss值为:{loss_value}")
        start = time.time()

xs_test = jnp.array([0.17])
print("真实的计算值：",a*xs_test+b)
print("模型拟合后的计算值：",forward(params,xs_test))


# for grd in grad_value:
#     print("-------------")
#     weight = jnp.array(grd["weight"])
#     bias = jnp.array(grd["bias"])
#
#     print("bias_shape:",weight.shape)
#     print("weight_shape:",bias.shape)

