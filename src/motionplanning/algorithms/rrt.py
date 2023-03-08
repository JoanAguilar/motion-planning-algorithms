"""Implementation for the RRT algorithm."""

import bisect
from typing import Callable, Hashable, TypeVar

from motionplanning.graphs import Tree

T = TypeVar('T', bound=Hashable)


class MaxIterError(Exception):
    """Raised when the maximum number of iterations is reached."""


def rrt(
        init_node: T,
        sample_free: Callable[[], T],
        cost: Callable[[T, T], float],
        goal: Callable[[T], bool],
        steer: Callable[[T, T], T],
        check_path: Callable[[T, T], bool],
        max_iter: int
        ) -> tuple[Tree[T], T]:
    """
    Run the RRT (Rapidly-exploring Random Trees) algorithm.

    If the RRT algorithm runs successfully, it generates a tree graph
    with one node in the specified goal region. One can then examine
    the tree in order to find the path connecting the starting node
    with the node in the goal region. The implementation follows the
    one outlined in [1], see there for more information.

    Args:
        init_node: Initial node of the tree. This node will always be
            present in the returned tree.
        sample_free: Function to sample a node from the free space.
        cost: Function to compute the cost between two nodes.
        goal: Function that checks if a node is in the goal region.
            "True" means the node is in the goal region.
        steer: Function to generate a node from one node towards
            another node. ``steer(v0, v1)`` generates a new node,
            ``v2``, which is on the path from ``v0`` towards ``v1``.
            In general, it is expected that ``v2`` will be closer (or
            as close) to ``v0`` than ``v1``.
        check_path: Function to check if the path between two nodes
            is feasible (obstacle-free). "True" means the path is
            feasible.
        max_iter: Maximum number of iterations to run the algorithm.

    Returns:
        The generated tree, and the node in the goal region.

    Raises:
        MaxIterError: if the maximum number of iterations is reached
            and no node in the goal region has been found.

    [1] KARAMAN, Sertac; FRAZZOLI, Emilio. Sampling-based algorithms
    for optimal motion planning. _The international journal of_
    _robotics research,_ 2011, 30.7: 846-894.
    """

    def nearest(n1, nodes):
        # Return the node nearest to ``n1`` in ``nodes`` in terms of
        # cost.
        nodes_cost_sorted: list[tuple[T, float]] = []
        for n2 in nodes:
            c12 = cost(n1, n2)
            bisect.insort(nodes_cost_sorted, (n2, c12), key=lambda x: x[1])
        return nodes_cost_sorted[0][0]

    tree = Tree[T](init_node)
    if goal(init_node):
        return tree, init_node
    for _ in range(max_iter):
        v = sample_free()
        near_v = nearest(v, tree.nodes)
        new_v = steer(near_v, v)
        if check_path(near_v, new_v):
            tree.add_node(new_v, near_v, cost(near_v, new_v))
            if goal(new_v):
                return tree, new_v

    raise MaxIterError(
        f"Could not find a node in the goal region after {max_iter} "
        "iterations.")
