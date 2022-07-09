import jax.random
from jax import lax
import jax.numpy as jnp

operand = jnp.array([0.])
#print(lax.cond(True, lambda x: x + 1, lambda x: x - 1, operand))
#print(lax.cond(False, lambda x: x + 1, lambda x: x - 1, operand))

#print("---------------------------------")
def add_fun(x):
    return x + 1.

def subtraction_fun(x):
    return x - 1.
#print(lax.cond(True, add_fun, subtraction_fun, operand))
#print(lax.cond(False, add_fun, subtraction_fun, operand))

#print("---------------------------------")
x = 0
def add_fun(x):
    return x + 1.

def subtraction_fun(x):
    return x - 1.
#print(lax.cond(x > 0, add_fun, subtraction_fun, x))
#print(lax.cond(x <= 0, add_fun, subtraction_fun, x))


init_val = 0

def cond_fun(x):
    return x < 17
def body_fun(x):
    return x + 1
y = lax.while_loop(cond_fun, body_fun, init_val)
#print(y)

init_val = 0
start = 0
stop = 10
body_fun = lambda i,x: x+i
#print(lax.fori_loop(start, stop, body_fun, init_val))
print("---------------------------------")


def add_fun(i,x):
    return i+1.,x + 1.

print(lax.scan(add_fun, 0, jnp.array([1, 2, 3,4])))