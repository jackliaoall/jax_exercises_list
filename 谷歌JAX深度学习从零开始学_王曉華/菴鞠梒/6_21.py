import jax
import jax.numpy as jnp

global_list = []

def log(x):
  global_list.append(x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

#print(jax.make_jaxpr(log)(3.0))
#print(global_list)


def pring_log(x):
    print("print_test:", x)
    x = jnp.log(x)
    print("print_test:", x)
    return x
print(jax.make_jaxpr(pring_log)(3.0))



