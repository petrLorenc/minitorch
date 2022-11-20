from __future__ import annotations # allow add typing hints for classes inside each others
from typing import List
from enum import Enum
import random

from abc import ABC, abstractmethod


class OperationTypes(ABC):

    def __init__(self, value: float):
        self.value = value

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def derivative(self, node: Node):
        pass


class InputNode(OperationTypes):

    def derivative(self, node: Node):
        return 0.0

    def __str__(self):
        return " << "


class AddOperation(OperationTypes):

    # if f = 2x + 3y then
    # df/dx = 2
    # if L = 4f + 5y then
    # dL/df = 4
    # together dL/dx = dL/df * df/dx (chain rule) = 4 * 2 = 8
    def derivative(self, node: Node):
        return 1.0

    def __str__(self):
        return " + "


class ProductOperation(OperationTypes):

    def derivative(self, node: Node):
        return node.operation.value

    def __str__(self):
        return " * "


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

    def __init__(self, references: List[Node] = None, operation: OperationTypes = None):
        self.operation = operation
        self.references = references if references is not None else []
        self.gradient = 0.0

        self._id = str(round(random.random(), 5)).split(".")[1] # just for visualisation to have better idea about REF

    # overloading () operation
    def __call__(self, *args, **kwargs) -> float:
        return self.operation.value

    # overloading + operation
    def __add__(self, other: Node) -> Node:
        return Node(references=[self, other], operation=AddOperation(value=self() + other()))

    # overloading * operation
    def __mul__(self, other: Node) -> Node:
        return Node(references=[self, other], operation=ProductOperation(value=self() * other()))

    # for showing purposes
    def __repr__(self, prefix=""):
        output_repr = f"{prefix} Node({self.operation.value}, id={self._id}, grad={self.gradient}) was created by {self.operation}"
        output_repr += "\n"
        for ref in self.references:
            output_repr += prefix + ref.__repr__(prefix=prefix + " # ")
        output_repr += ""
        return output_repr

    def backward(self, current_derivative=1.):
        """
            Calculate gradient of current Node - influence of current Node on final output
            We will use chain rule and work backward
            :return:
        """
        self.gradient += current_derivative # need to sum it up - but then need something like zero_grad()

        if len(self.references) == 2:
            self.references[0].backward(self.operation.derivative(self.references[1]) * self.gradient)
            self.references[1].backward(self.operation.derivative(self.references[0]) * self.gradient)
        elif len(self.references) == 0:
            pass


if __name__ == '__main__':
    x = Node(operation=InputNode(2)) # should have grad = 3
    y = Node(operation=InputNode(3)) # should have grad = 2
    z = Node(operation=InputNode(4)) # should have grad = 1
    o = (y * x) * z + (x * z)
    print(o)
    o.backward()
    print(o)

