from minitorch.tensors import Node, InputNode


def test_linear_function():
    x = Node(operation=InputNode(2)) # should have grad = 3
    w = Node(operation=InputNode(3)) # should have grad = 2
    b = Node(operation=InputNode(4)) # should have grad = 1
    o = (x * w) + b
    o.backward()
    assert x.gradient == 3
    assert w.gradient == 2
    assert b.gradient == 1


def test_complex_function():
    x = Node(operation=InputNode(2)) # should have grad = 3
    y = Node(operation=InputNode(3)) # should have grad = 2
    z = Node(operation=InputNode(4)) # should have grad = 1
    o = (x * y) + (x * z) + (y * x) * z
    o.backward()
    assert x.gradient == 19
    assert y.gradient == 10
    assert z.gradient == 8
