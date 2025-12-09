from typing import NewType, TypeVarTuple

class Array[*Shape]:

    def __init__(self, shape: tuple[*Shape]):
        self._shape: tuple[*Shape] = shape

    def get_shape(self) -> tuple[*Shape]:
        return self._shape

Height = NewType('Height', int)
Width = NewType('Width', int)
shape = (Height(480), Width(640))
x: Array[Height, Width] = Array(shape)
y = abs(x)  # Inferred type is Array[Height, Width]
z = x + x   #        ...    is Array[Height, Width]