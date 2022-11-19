from __future__ import annotations # allow add typing hints for classes inside each others
from typing import List
from enum import Enum
import random

from abc import ABC, abstractmethod


class OperationTypes(Enum):
    NO_OPERATION = "INPUT"
    ADDITION = " + "
    MULTIPLICATION = " * "

    def __str__(self):
        return self.value


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

    def __init__(self, value: float, references: List[Node] = None, operation_type: OperationTypes = OperationTypes.NO_OPERATION):
        self.value = value
        self.references = references if references is not None else []
        self.operation_type = operation_type
        self.gradient = 1. # gradient for final node, others will be updated after calling backward()

        self._id = str(round(random.random(), 5)).split(".")[1] # just for visualisation to have better idea about REF

    # overloading () operation
    def __call__(self, *args, **kwargs) -> float:
        return self.value

    # overloading + operation
    def __add__(self, other: Node) -> Node:
        return Node(value=self() + other(), references=[self, other], operation_type=OperationTypes.ADDITION)

    # overloading * operation
    def __mul__(self, other: Node) -> Node:
        return Node(value=self() * other(), references=[self, other], operation_type=OperationTypes.MULTIPLICATION)

    # for showing purposes
    def __repr__(self, prefix=""):
        output_repr = f"{prefix} Node({self.value}, id={self._id}) was created by {self.operation_type}"
        output_repr += "\n"
        for ref in self.references:
            output_repr += prefix + ref.__repr__(prefix=prefix + " # ")
        output_repr += ""
        return output_repr

    def backward(self):
        pass

    def derivative(self, previous_gradient) -> float:
        """
        Calculate gradient of current Node - influence of current Node on final output
        We will use chain rule and work backward
        :return:
        """
        return


if __name__ == '__main__':
    i = Node(2)

    x = Node(3)
    y = Node(4) + i
    z = Node(5) + Node(3) + i

    xy = x + y
    xyz = xy * z

    print(xyz)

