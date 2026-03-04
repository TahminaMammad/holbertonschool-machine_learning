#!/usr/bin/env python3
"""
Module implementing a basic Decision Tree structure.

This module defines:
- Node: internal node of a decision tree
- Leaf: terminal node
- Decision_Tree: wrapper class

This version adds functionality to count:
- Total number of nodes
- Number of leaves only
"""

import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.

    Attributes:
        feature (int): Feature index used for splitting.
        threshold (float): Threshold value for the split.
        left_child (Node): Left subtree.
        right_child (Node): Right subtree.
        is_leaf (bool): Whether the node is a leaf.
        is_root (bool): Whether the node is the root.
        sub_population (array-like): Subset of data.
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
            int: Maximum depth among descendants.
        """
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """
        Counts nodes in the subtree rooted at this node.

        Args:
            only_leaves (bool): If True, counts only leaves.
                                If False, counts all nodes.

        Returns:
            int: Number of nodes (or leaves) in subtree.
        """
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves
        )
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )

        if only_leaves:
            return left_count + right_count

        return 1 + left_count + right_count


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.

    Attributes:
        value: Prediction value stored in the leaf.
        depth (int): Depth of the leaf.
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

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the leaf node.

        Args:
            only_leaves (bool): Ignored (always returns 1).

        Returns:
            int: 1
        """
        return 1


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
        split_criterion (str): Splitting strategy.
        predict (callable): Prediction method (to be defined).
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
            int: Maximum depth.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts nodes in the tree.

        Args:
            only_leaves (bool): If True, counts only leaves.
                                If False, counts all nodes.

        Returns:
            int: Number of nodes (or leaves).
        """
        return self.root.count_nodes_below(
            only_leaves=only_leaves
        )
