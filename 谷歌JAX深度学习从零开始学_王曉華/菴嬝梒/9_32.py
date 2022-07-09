import numpy as np
from typing import Any, Callable

class ArrayType:
  def __getitem__(self, idx):
    return Any

f32 = ArrayType()

x: f32[(2, 3)] = np.ones((2, 3), dtype=np.float32)
y: f32[(3, 5)] = np.ones((3, 5), dtype=np.float32)
z: f32[(2, 5)] = x.dot(y)
w: f32[(7, 1, 5)] = np.ones((7, 1, 5), dtype=np.float32)
q: f32[(7, 2, 5)] = z + w

print(q)
print(q.shape)