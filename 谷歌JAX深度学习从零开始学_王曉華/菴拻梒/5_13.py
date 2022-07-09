import jax
import time

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)

key = jax.random.PRNGKey(17)
x = jax.random.normal(key,shape=[1024,1024])

start = time.time()
norm(x)
end = time.time()
print("循环运行时间:%.2f秒"%(end-start))

jit_norm = jax.jit(norm)
start = time.time()
jit_norm(x)
end = time.time()
print("循环运行时间:%.2f秒"%(end-start))


def get_negatives(x):
  return x < 0
x = jax.random.normal(key,shape=[10,10])
print(get_negatives(x).shape)

jax.jit(get_negatives)(x)