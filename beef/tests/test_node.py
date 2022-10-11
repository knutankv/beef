import unittest
from ..fe.node import Node

class TestThreeDimNode(unittest.TestCase):
    _node = Node(1,[0,1,2])
    assert _node.label == 1

if __name__ == "__main__":
    unittest.main()