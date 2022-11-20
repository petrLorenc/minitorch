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
    def derivative(self, left_node: Node, right_node: Node):
        pass


class InputNode(OperationTypes):

    def derivative(self, left_node: Node, right_node: Node):
        return 0., 0.

    def __str__(self):
        return " << "


class AddOperation(OperationTypes):

    # if f = 2x + 3y then
    # df/dx = 2
    # if L = 4f + 5y then
    # dL/df = 4
    # together dL/dx = dL/df * df/dx (chain rule) = 4 * 2 = 8
    def derivative(self, left_node: Node, right_node: Node):
        return 1., 1.

    def __str__(self):
        return " + "


class SubtractOperation(OperationTypes):

    # if f = 2x + 3y then
    # df/dx = 2
    # if L = 4f + 5y then
    # dL/df = 4
    # together dL/dx = dL/df * df/dx (chain rule) = 4 * 2 = 8
    def derivative(self, left_node: Node, right_node: Node):
        return 1., -1.

    def __str__(self):
        return " - "


class ProductOperation(OperationTypes):

    def derivative(self, left_node: Node, right_node: Node):
        return right_node.operation.value, left_node.operation.value

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

    def __init__(self, references: List[Node] = None, operation: OperationTypes = None, label: str = None):
        self.operation = operation
        self.references = references if references is not None else []
        self.gradient = 0.0

        self._id = str(round(random.random(), 5)).split(".")[1] if not label else label # just for visualisation to have better idea about REF

    # overloading () operation
    def __call__(self, *args, **kwargs) -> float:
        return self.operation.value

    # overloading + operator
    def __add__(self, other: Node) -> Node:
        return Node(references=[self, other], operation=AddOperation(value=self() + other()))

    # overloading * operator
    def __mul__(self, other: Node) -> Node:
        return Node(references=[self, other], operation=ProductOperation(value=self() * other()))

    # overloading - operator # from left to right A - B = int(A).__sub__(B)
    def __sub__(self, other: Node) -> Node:
        return Node(references=[self, other], operation=SubtractOperation(value=self() - other()))

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
            partial_derivative_left, partial_derivative_right = self.operation.derivative(self.references[0], self.references[1])
            self.references[0].backward(partial_derivative_left * self.gradient)
            self.references[1].backward(partial_derivative_right * self.gradient)
        elif len(self.references) == 0:
            pass


if __name__ == '__main__':
    x1 = Node(operation=InputNode(2), label="x1")
    w1 = Node(operation=InputNode(3), label="w1")
    x2 = Node(operation=InputNode(2), label="x2")
    w2 = Node(operation=InputNode(3), label="w2")
    b = Node(operation=InputNode(4), label="b")

    o = (x1 * w1) - (x2 * w2) + b
    print(o)
    o.backward()
    print(o)

