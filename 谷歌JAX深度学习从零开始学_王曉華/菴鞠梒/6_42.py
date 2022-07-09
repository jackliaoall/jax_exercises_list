import jax

def tree_transpose(list_of_trees):
  return jax.tree_multimap(lambda *xs: list(xs), *list_of_trees)

episode_steps = [dict(t=1, obs=3), dict(t=2, obs=4)]
print(tree_transpose(episode_steps))


