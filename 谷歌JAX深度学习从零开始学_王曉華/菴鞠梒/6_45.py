import jax

class MyContainer:
  def __init__(self, name, a, b, c):
    self.name = name
    self.a = a
    self.b = b
    self.c = c


def flatten_MyContainer(container:MyContainer):
  flat_contents = [container.a, container.b, container.c]
  aux_data = container.name
  return flat_contents, aux_data

def unflatten_MyContainer(aux_data: str, flat_contents: list) -> MyContainer:
  return MyContainer(aux_data, flat_contents)	#这里使用了python自动拆箱方法

jax.tree_util.register_pytree_node(MyContainer, flatten_MyContainer, unflatten_MyContainer)

print(jax.tree_leaves([
    MyContainer('xiaohua', 1, 2, 3),
    MyContainer('xiaoming', 1, 2, 3)
]))










