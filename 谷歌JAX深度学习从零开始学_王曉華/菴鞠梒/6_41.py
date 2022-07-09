import jax
import jax.numpy as jnp

example_trees = [
    1,
    "a",
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {'k1': 2, 'k2': (3, 4)}, 5],
    {'a': 2, 'b': (2, 3)},
    jnp.array([1, 2, 3]),
]

for pytree in example_trees:
    leaves = jax.tree_leaves(pytree)
    #print(f"{pytree}     has {len(leaves)} leaves: {leaves}")

list_of_lists = [
    [1, 2, 3],
    [1, 2],
    [ 2, 3, [4,5]]
]

#print(jax.tree_map(lambda x: x * 2, list_of_lists))

first_list = [[1, 2, 3],[1, 2, 3],[1, 2, 3]]
second_list = [[1,0, 1],[1, 1, 1],[0, 0, 0]]

print(jax.tree_multimap(lambda x, y: x + y, first_list, second_list))
