#!/usr/bin/env python3
"""
Module implementing the basic structure of a Decision Tree.

This module defines three classes:
- Node: internal node of a decision tree
- Leaf: terminal node containing a prediction value
- Decision_Tree: wrapper class representing the full tree

This file implements the logic to compute the maximum depth
of a decision tree.
"""

import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.

    Attributes:
        feature (int): Index of the feature used for splitting.
        threshold (float): Threshold value for the split.
        left_child (Node): Left subtree.
        right_child (Node): Right subtree.
        is_leaf (bool): Whether the node is a leaf.
        is_root (bool): Whether the node is the root.
        sub_population (array-like): Subset of data at this node.
        depth (int): Depth of the node in the tree.
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

    def max_depth_below(self):
        """
        Computes the maximum depth of the subtree
        rooted at this node.

        Returns:
            int: Maximum depth among all descendants.
        """
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.

    Attributes:
        value: Prediction value stored in the leaf.
        depth (int): Depth of the leaf in the tree.
    """

    def __init__(self, value, depth=None):
        """
        Initializes a Leaf instance.

        Args:
            value: Prediction value.
            depth (int): Depth of the leaf.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf.

        Returns:
            int: Depth of the leaf.
        """
        return self.depth


class Decision_Tree:
    """
    Represents a Decision Tree classifier.

    Attributes:
        rng (Generator): Random number generator.
        root (Node): Root node of the tree.
        explanatory (array-like): Feature data.
        target (array-like): Target labels.
        max_depth (int): Maximum allowed depth.
        min_pop (int): Minimum population to split.
        split_criterion (str): Criterion for splitting.
        predict (callable): Prediction method (to be defined later).
    """

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random",
                 root=None):
        """
        Initializes a Decision_Tree instance.

        Args:
            max_depth (int): Maximum depth allowed.
            min_pop (int): Minimum population per node.
            seed (int): Random seed.
            split_criterion (str): Splitting criterion.
            root (Node): Optional pre-built root node.
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

    def depth(self):
        """
        Returns the maximum depth of the decision tree.

        Returns:
            int: Maximum depth of the tree.
        """
        return self.root.max_depth_below()
