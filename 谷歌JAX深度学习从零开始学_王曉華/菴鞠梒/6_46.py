import jax
import jax.numpy as jnp

class Counter:

  def __init__(self):
    self.n = 0

  def count(self):
    self.n += 1
    return self.n

  def reset(self):
    self.n = 0

counter = Counter()
print(jax.make_jaxpr(counter.count)())


# for _ in range(3):
#   print(counter.count())

# fast_counter = jax.jit(counter.count)
# for i in range(3):
#   print(fast_counter())

class CounterV2:

  def __init__(self):
    pass

  def count(self,n):
    n += 1
    return n

counter = CounterV2()
n = 0
for i in range(3):
  n = counter.count(n)
  print(n)

print(jax.make_jaxpr(counter.count)(n))
