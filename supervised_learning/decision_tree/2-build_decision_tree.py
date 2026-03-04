#!/usr/bin/env python3
"""
Module implementing a printable Decision Tree structure.

Adds string representation (__str__) functionality
to display the tree hierarchy in a readable format.
"""

import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 is_root=False, depth=0):
        """
        Initializes a Node instance.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def left_child_add_prefix(self, text):
        """
        Adds formatting prefix for left subtree.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |  " + line + "\n"
        return new_text.rstrip("\n")

    def right_child_add_prefix(self, text):
        """
        Adds formatting prefix for right subtree.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "       " + line + "\n"
        return new_text.rstrip("\n")

    def __str__(self):
        """
        Returns string representation of the node and its subtree.
        """
        if self.is_root:
            node_repr = (
                f"root [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )
        else:
            node_repr = (
                f"-> node [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )

        left_text = self.left_child.__str__()
        right_text = self.right_child.__str__()

        left_block = self.left_child_add_prefix(left_text)
        right_block = self.right_child_add_prefix(right_text)

        return node_repr + "\n" + left_block + "\n" + right_block


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.
    """

    def __init__(self, value, depth=None):
        """
        Initializes a Leaf instance.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """
        Returns string representation of a leaf.
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """
    Represents a Decision Tree.
    """

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random",
                 root=None):
        """
        Initializes a Decision_Tree instance.
        """
        self.rng = np.random.default_rng(seed)

        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self):
        """
        Returns string representation of the tree.
        """
        return self.root.__str__()
