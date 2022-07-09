import jax
import jax.numpy as jnp
from jax import custom_vjp, custom_jvp
from jax import jit, grad, vmap
from jax import custom_vjp
import jax.numpy as jnp


# f :: a -> b
@custom_vjp
def f(x):
    return x**2


# f_fwd :: a -> (b, c)
def f_fwd(x):
    return f(x), 2*x


# f_bwd :: (c, CT b) -> CT a
def f_bwd(dot_x, y_bar):
    return (dot_x,)


f.defvjp(f_fwd, f_bwd)

# print(f(3.))
print((grad(f)(3.)))
