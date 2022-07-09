import jax
import jax.numpy as jnp
from jax import grad
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental import optimizers
import gen_and_dis


def sample_latent(key, shape):
    return jax.random.normal(key, shape=shape)


key = jax.random.PRNGKey(17)

latent = sample_latent(key, shape=(100, 64))
real_shape = (-1, 28, 28, 1)

# gen_fun的处理

gen_init_random_params, gen_predict = gen_and_dis.generator()
fake_shape = (-1, 1, 1, 1)

gen_opt_init, gen_opt_update, gen_get_params = optimizers.adam(step_size=2e-4)
_, gen_init_params = gen_init_random_params(key, fake_shape)
gen_opt_state = gen_opt_init(gen_init_params)

# dic_fun的处理
dic_init_random_params, dic_predict = gen_and_dis.discriminator()
real_shape = (-1, 28, 28, 1)

dic_opt_init, dic_opt_update, dic_get_params = optimizers.adam(step_size=2e-4)
_, dic_init_params = dic_init_random_params(key, real_shape)
dic_opt_state = dic_opt_init(dic_init_params)


@jax.jit
def loss_generator(gen_params,dic_params, fake_image):
    gen_result = gen_predict(gen_params, fake_image)
    fake_result = dic_predict(dic_params,gen_result)

    fake_targets = jnp.tile(jnp.array([0,1]),[fake_image.shape[0],1])   #[0,1]代表虚假数据
    loss = jnp.mean(jnp.sum(-fake_targets * fake_result, axis=1))
    return loss

@jax.jit
def loss_discriminator(dic_params,gen_params, fake_image,real_image):
    gen_result = gen_predict(gen_params, fake_image)

    fake_result = dic_predict(dic_params,gen_result)
    real_result = dic_predict(dic_params, real_image)

    fake_targets = jnp.tile(jnp.array([0,1]),[fake_image.shape[0],1])   #[0,1]代表虚假数据
    real_targets = jnp.tile(jnp.array([1,0]),[real_image.shape[0],1])   #[1,0]代表真实数据
    loss = jnp.mean(jnp.sum(-fake_targets * fake_result, axis=1)) + jnp.mean(jnp.sum(-real_targets * real_result, axis=1))
    return loss

mnist_data = gen_and_dis.mnist_data
batch_size = 128
for i in range(1):
    batch_num = len(mnist_data)//batch_size

    for j in range(batch_num):
        start = batch_size * j
        end = batch_size * (j + 1)
        real_image = mnist_data[start:end]
        gen_params = gen_get_params(gen_opt_state)
        dic_params = dic_get_params(dic_opt_state)

        fake_image = jax.random.normal(key + j, shape=[batch_size, 1, 1, 1])
        gen_opt_state = gen_opt_update(j, grad(loss_generator)(gen_params,dic_params, fake_image), gen_opt_state)
        dic_opt_state = gen_opt_update(j, grad(loss_discriminator)(dic_params,gen_params, fake_image,real_image), dic_opt_state)

