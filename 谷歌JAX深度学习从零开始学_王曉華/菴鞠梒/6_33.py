import jax

f = lambda x: x**3 + 2*x**2 - 3*x + 1

dfdx = jax.grad(f)

print(dfdx(1.))
dfdx2 = jax.grad(dfdx)
print(dfdx2(1.))

print(jax.jacfwd(dfdx)(1.))
print(jax.jacrev(f)(1.))
print(jax.jacfwd(dfdx2)(1.))