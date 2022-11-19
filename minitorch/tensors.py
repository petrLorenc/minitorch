from __future__ import annotations # allow add typing hints for classes inside each others
from typing import List

from abc import ABC, abstractmethod


class MyObject(ABC):
    """
    Object which should be parent to all used objects
    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class Node(MyObject):

    def __init__(self, value: float, references: List[Node] = None):
        self.value = value
        self.references = references if references is not None else []

    def __call__(self, *args, **kwargs):
        return self.value

    def __add__(self, other: Node):
        return Node(value=self() + other(), references=[self, other])

    def __mul__(self, other: Node):
        return Node(value=self()*other(), references=[self, other])

    def __repr__(self):
        return f"Node({self.value}) with references to {[str(x) for x in self.references]}"

    def backward(self):
        return 0


if __name__ == '__main__':
    x = Node(1)
    y = Node(4)
    z = Node(5)

    xy = x + y
    xyz = xy * z

    print(type(xyz))
    print(xyz)

