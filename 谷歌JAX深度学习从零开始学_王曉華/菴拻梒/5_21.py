import jax
import jax.numpy as jnp

def impure_print_side_effect(x):
  print("实施函数计算")  # This is a side-effect
  return x


# print ("First call: ", jax.jit(impure_print_side_effect)(4.))
# print("--------------------")
#
# print ("Second call: ", jax.jit(impure_print_side_effect)(5.))
# print("--------------------")
#
# print ("Third call, different type: ", jax.jit(impure_print_side_effect)(jnp.array([5.])))

g = 0.
def impure_saves_global(x):
  global g
  g = x
  return x


# print ("First call: ", jax.jit(impure_saves_global)(4.))
# print ("Saved global: ", g)

def pure_uses_internal_state(x):
  state = dict(even=0, odd=0)
  for i in range(10):
    state['even' if i % 2 == 0 else 'odd'] += x
  return state['even'] + state['odd']

print(jax.jit(pure_uses_internal_state)(3.))
print(jax.jit(pure_uses_internal_state)(jnp.array([5.])))



import jax.numpy as jnp
import jax.lax as lax
from jax import make_jaxpr

# lax.fori_loop
array = jnp.arange(10)
print(lax.fori_loop(0, 10, lambda i,x: x+array[i], 0)) # expected result 45
iterator = iter(range(10))
print(lax.fori_loop(0, 10, lambda i,x: x+next(iterator), 0)) # unexpected result 0

# lax.scan
def func11(arr, extra):
    ones = jnp.ones(arr.shape)
    def body(carry, aelems):
        ae1, ae2 = aelems
        return (carry + ae1 * ae2 + extra, carry)
    return lax.scan(body, 0., (arr, ones))
make_jaxpr(func11)(jnp.arange(16), 5.)
# make_jaxpr(func11)(iter(range(16)), 5.) # throws error

# lax.cond
array_operand = jnp.array([0.])
lax.cond(True, lambda x: x+1, lambda x: x-1, array_operand)
iter_operand = iter(range(10))
# lax.cond(True, lambda x: next(x)+1, lambda x: next(x)-1, iter_operand) # throws error

