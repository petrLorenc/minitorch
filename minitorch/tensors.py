from __future__ import annotations  # allow add typing hints for classes inside each others
from typing import List, Union, Tuple
from enum import Enum
import random

from abc import ABC, abstractmethod, abstractstaticmethod


class Operation(ABC):
    def __init__(self, value: float):
        self.value = value

    def __str__(self):
        return self.__class__.__name__


class BinaryOperation(Operation):

    @abstractmethod
    def derivative(self, left_node: Node, right_node: Node) -> Tuple[float, float]:
        pass


class AddOperation(BinaryOperation):

    # if f = 2x + 3y then
    # df/dx = 2
    # if L = 4f + 5y then
    # dL/df = 4
    # together dL/dx = dL/df * df/dx (chain rule) = 4 * 2 = 8
    def derivative(self, left_node: Node, right_node: Node):
        return 1., 1.

    def __str__(self):
        return " + "


class SubtractOperation(BinaryOperation):

    # if f = 2x + 3y then
    # df/dx = 2
    # if L = 4f + 5y then
    # dL/df = 4
    # together dL/dx = dL/df * df/dx (chain rule) = 4 * 2 = 8
    def derivative(self, left_node: Node, right_node: Node) -> Tuple[float, float]:
        return 1., -1.

    def __str__(self):
        return " - "


class ProductOperation(BinaryOperation):

    def derivative(self, left_node: Node, right_node: Node):
        return right_node.operation.value, left_node.operation.value

    def __str__(self):
        return " * "


class UnaryOperation(Operation):

    @abstractmethod
    def derivative(self, root_node: Node) -> float:
        pass

    @staticmethod
    @abstractmethod
    def apply(onto: Node, *args, **kwargs) -> Node:
        pass

    def __str__(self):
        return " UnaryOperation "


class InputNode(Operation):

    def derivative(self, _node: Node) -> float:
        return 0.

    def __str__(self):
        return " << "


class SigmoidOperation(UnaryOperation):

    @staticmethod
    def apply(onto: Node, *args, **kwargs) -> Node:
        import math
        return Node(references=[onto], operation=SigmoidOperation(value=(1 / (1 + math.exp(-onto.operation.value)))) )

    def derivative(self, _node: Node) -> float:
        return self.value * (1 - self.value)

    def __str__(self):
        return " sigmoid() "


class TanhOperation(UnaryOperation):

    @staticmethod
    def apply(onto: Node, *args, **kwargs) -> Node:
        import math
        return Node(references=[onto], operation=TanhOperation(value=math.tanh(onto.operation.value)))

    def derivative(self, _node: Node):
        return 1 - (self.value * self.value)

    def __str__(self):
        return " tanh() "


class Node:
    """
    Main object which is using for calculation and also for storing reference to other nodes. It also saves the gradient.
    """

    def __init__(self, references: List[Node] = None, operation: Operation = None, label: str = None):
        self.operation = operation
        self.references = references if references is not None else []
        self.gradient: float = 0.0

        self._id = str(round(random.random(), 5)).split(".")[
            1] if not label else label  # just for visualisation to have better idea about REF

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
        self.gradient += current_derivative  # need to sum it up - but then need something like zero_grad()

        if len(self.references) == 2:
            self.operation: BinaryOperation
            partial_derivative_left, partial_derivative_right = self.operation.derivative(self.references[0], self.references[1])
            self.references[0].backward(partial_derivative_left * self.gradient)
            self.references[1].backward(partial_derivative_right * self.gradient)
        elif len(self.references) == 1:
            self.operation : UnaryOperation
            partial_derivative = self.operation.derivative(self.references[0])
            self.references[0].backward(partial_derivative * self.gradient)
        elif len(self.references) == 0:
            pass

    def zero_grad(self):
        for node in self.references:
            node.zero_grad()
        self.gradient = 0.0


if __name__ == '__main__':
    x1 = Node(operation=InputNode(2), label="x1")
    w1 = Node(operation=InputNode(3), label="w1")
    x2 = Node(operation=InputNode(2), label="x2")
    w2 = Node(operation=InputNode(3), label="w2")
    b = Node(operation=InputNode(4), label="b")

    o = (x1 * w1) - (x2 * w2) + b
    out = TanhOperation.apply(onto=o)
    print(out)
    out.backward()
    print(out)
    print("#" * 100)

    x1 = Node(operation=InputNode(2), label="x1")
    w1 = Node(operation=InputNode(3), label="w1")
    x2 = Node(operation=InputNode(2), label="x2")
    w2 = Node(operation=InputNode(3), label="w2")
    b = Node(operation=InputNode(4), label="b")

    o = (x1 * w1) - (x2 * w2) + b
    out = SigmoidOperation.apply(onto=o)
    print(out)
    out.backward()
    print(out)