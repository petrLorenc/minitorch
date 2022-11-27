from __future__ import annotations  # allow add typing hints for classes inside each others

import random
from typing import Type, List, Tuple, Union

from minitorch.tensors import Node, InputNode, UnaryOperation


class Neuron:

    def __init__(self, dimension: int, activation: Type[UnaryOperation]):
        self.W = [Node(label= f"w_{idx}",operation=InputNode(random.random())) for idx in range(dimension)]
        self.b = Node(label="b",operation=InputNode(random.random()))
        self.activation = activation

    def __call__(self, *args, **kwargs) -> Node:
        assert isinstance(args[0], Node)
        assert len(args) == len(self.W)

        # f = W*x + b
        output_node: Node = self.W[0] * args[0]
        for w_nth, x_nth in zip(self.W[1:], args[1:]):
            output_node += w_nth * x_nth
        output_node += self.b

        # activation function
        return self.activation.apply(output_node)

    def _get_grads(self):
        return [x.gradient for x in self.W], self.b.gradient

    def train_step(self, learning_rate: float):
        w_grads, b_grads = self._get_grads()
        for node, w_grad in zip(self.W, w_grads):
            node.operation.value += w_grad * learning_rate
        self.b.operation.value += b_grads * learning_rate

    def __str__(self):
        output_str = ""
        for w in self.W:
            output_str += str(w)
        output_str += str(self.b)
        return output_str

# class NeuralNetwork():
#
#     def __init__(self, dimensions: List[Tuple]):
#         self.layers = layers
#
#     def __call__(self, *args, **kwargs):
#         assert isinstance(args[0], Node)
#         assert len(args) == len(self.W)


if __name__ == '__main__':
    from minitorch.tensors import TanhOperation
    random.seed(42)

    layer = Neuron(dimension=3, activation=TanhOperation)
    inputs = [
        Node(operation=InputNode(1.0), label="x_1"),
        Node(operation=InputNode(0.0), label="x_2"),
        Node(operation=InputNode(-1.0), label="x_3")
    ]
    train_label = Node(operation=InputNode(1.0))
    print(layer)
    for epoch in range(100):
        layer_output = layer(*inputs)
        loss = layer_output - train_label
        print(loss.operation.value)
        loss.backward()
        # print(loss)
        layer.train_step(0.01)
        # print(loss)
        loss.zero_grad()
        # print(loss)
    print(layer)
