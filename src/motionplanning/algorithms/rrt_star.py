"""Implementations for the RRT* and k-RRT* algorithms."""

import bisect
from typing import Callable, Iterable, TypeVar

import numpy as np

from motionplanning.algorithms import MaxIterError
from motionplanning.graphs import Tree

T = TypeVar('T')


def rrt_star(
        init_node: T,
        sample_free: Callable[[], T],
        cost: Callable[[T, T], float],
        goal: Callable[[T], bool],
        steer: Callable[[T, T], T],
        check_path: Callable[[T, T], bool],
        gamma_rrt: float,
        d: int,
        steer_cost: float,
        max_iters: int,
        return_fast: bool = False
        ) -> tuple[Tree[T], T]:
    """
    Run the RRT* (RRT-star) algorithm.

    If the RRT* algorithm runs successfully, it generates a tree graph
    with one node in the specified goal region. One can then examine
    the tree in order to find the path connecting the starting node
    with the node in the goal region. RRT* differs from RRT in that the
    tree generated by RRT* connects nodes through a minimum-cost path.
    The implementation follows the one outlined in [1], see there for
    more information.

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
            as close) to ``v0`` in terms of cost than ``v1``.
        check_path: Function to check if the path between two nodes
            is feasible (obstacle-free). "True" means the path is
            feasible.
        gamma_rrt: One of the three parameters (the others being ``d``
            and ``steer_cost``) used to compute the connection radius.
            The connection radius is computed as
            ``min(gamma_rrt * (log(len(V)) / len(V)) ** (1 / d), steer_cost)``,
            where ``len(V)`` is the number of nodes present in the
            computed graph at the current iteration.
        d: Dimensionality of the (free) space.
        steer_cost: Maximum cost between the node returned by ``steer``
            and its first argument. That is, ``steer_cost`` is the
            minimum value that satisfies
            ``steer_cost >= dist(v0, steer(v0, v1))`` for any value of
            ``v0`` and ``v1`` in the free space.
        max_iters: Maximum number of iterations to run the algorithm.
        return_fast: If set to True, the algorithm returns once a node
            in the goal region has been found; if set to False, the
            algorithm runs until the maximum number of iterations is
            reached.

    Returns:
        The generated tree, and the node in the goal region.

    Raises:
        MaxIterError: if the maximum number of iterations is reached
            and no node in the goal region has been found.

    [1] KARAMAN, Sertac; FRAZZOLI, Emilio. Sampling-based algorithms
    for optimal motion planning. _The international journal of_
    _robotics research,_ 2011, 30.7: 846-894.
    """

    def nearest(n1: T, nodes: Iterable[T]) -> T:
        # Return the node nearest to ``n1`` in ``nodes`` in terms of
        # cost.
        nodes_cost_sorted: list[tuple[T, float]] = []
        for n2 in nodes:
            c12 = cost(n1, n2)
            bisect.insort(nodes_cost_sorted, (n2, c12), key=lambda x: x[1])
        return nodes_cost_sorted[0][0]

    def near(n1: T, nodes: Iterable[T], near_cost: float) -> list[T]:
        # Return a list of nodes "near" ``n1`` in terms of cost. These
        # are the nodes with a cost less or equal to ``near_cost`` from
        # ``n1``. ``n1`` may be included in the list. The returned list
        # is sorted by cost.
        nodes_cost_sorted: list[tuple[T, float]] = []
        for n2 in nodes:
            c12 = cost(n1, n2)
            if c12 <= near_cost:
                bisect.insort(nodes_cost_sorted, (n2, c12), key=lambda x: x[1])
        return [node for node, _ in nodes_cost_sorted]

    def root_cost(v, tree: Tree[T]) -> float:
        # Return the cost of reaching node ``v`` from the root node
        # (``init_node``) in ``tree``.
        _, c = tree.path(init_node, v)
        return c

    tree = Tree[T](init_node)
    if goal(init_node):
        goal_node = init_node
    else:
        goal_node = None
    if return_fast and goal_node is not None:
        return tree, goal_node

    for _ in range(max_iters):
        nodes = tree.nodes
        v = sample_free()
        nearest_v = nearest(v, nodes)
        new_v = steer(nearest_v, v)
        if check_path(nearest_v, new_v):
            n_nodes = len(nodes)
            r = min(
                gamma_rrt * (np.log(n_nodes) / n_nodes) ** (1 / d),
                steer_cost)
            near_v = near(new_v, nodes, r)

            # Connect along a minimum-cost path.
            min_v = nearest_v
            min_c = root_cost(nearest_v, tree) + cost(nearest_v, new_v)
            for u in near_v:
                c = root_cost(u, tree) + cost(u, new_v)
                if check_path(u, new_v) and c < min_c:
                    min_v = u
                    min_c = c
            tree.add_node(new_v, min_v, cost(min_v, new_v))

            # Rewire the tree.
            for u in near_v:
                c = root_cost(new_v, tree) + cost(new_v, u)
                if check_path(new_v, u) and c < root_cost(u, tree):
                    tree.add_node(u, new_v, cost(new_v, u))

            if goal(new_v):
                goal_node = new_v
            if return_fast and goal_node is not None:
                return tree, goal_node

    if goal_node is not None:
        return tree, goal_node

    raise MaxIterError(
        f"Could not find a node in the goal region after {max_iters} "
        "iterations.")


def krrt_star(
        init_node: T,
        sample_free: Callable[[], T],
        cost: Callable[[T, T], float],
        goal: Callable[[T], bool],
        steer: Callable[[T, T], T],
        check_path: Callable[[T, T], bool],
        max_iters: int,
        k_rrt: float = 2 * np.e,
        return_fast: bool = False
        ) -> tuple[Tree[T], T]:
    """
    Run the k-RRT* (k-nearest RRT*) algorithm.

    If the k-RRT* algorithm runs successfully, it generates a tree
    graph with one node in the specified goal region. One can then
    examine the tree in order to find the path connecting the starting
    node with the node in the goal region. k-RRT* follows a similar
    procedure to RRT*, with the major difference that RRT* considers
    node connections based on a cost threshold, while k-RRT* considers
    connections with the ``k`` nearest nodes. The implementation
    follows the one outlined in [1], see there for more information.

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
            as close) to ``v0`` in terms of cost than ``v1``.
        check_path: Function to check if the path between two nodes
            is feasible (obstacle-free). "True" means the path is
            feasible.
        max_iters: Maximum number of iterations to run the algorithm.
        k_rrt: The parameter used to compute how many nodes are
            considered for connection. The number of nodes considered
            is computed as ``round(k_rrt * np.log(len(V)))``, where
            ``len(V)`` is the number of nodes in the computed graph at
            the current iteration.
        return_fast: If set to True, the algorithm returns once a node
            in the goal region has been found; if set to False, the
            algorithm runs until the maximum number of iterations is
            reached.

    Returns:
        The generated tree, and the node in the goal region.

    Raises:
        MaxIterError: if the maximum number of iterations is reached
            and no node in the goal region has been found.

    [1] KARAMAN, Sertac; FRAZZOLI, Emilio. Sampling-based algorithms
    for optimal motion planning. _The international journal of_
    _robotics research,_ 2011, 30.7: 846-894.
    """

    def k_nearest(n1: T, nodes: Iterable[T], k: int) -> list[T]:
        # Return the ``k`` nearest nodes to ``n1`` from ``nodes`` in
        # terms of cost.
        nodes_cost_sorted: list[tuple[T, float]] = []
        for n2 in nodes:
            c12 = cost(n1, n2)
            bisect.insort(nodes_cost_sorted, (n2, c12), key=lambda x: x[1])
        nodes_sorted = []
        for i in range(min(k, len(nodes_cost_sorted))):
            nodes_sorted.append(nodes_cost_sorted[i][0])
        return nodes_sorted

    def nearest(n1: T, nodes: Iterable[T]) -> T:
        # Return the node nearest to ``n1`` in ``nodes`` in terms of
        # cost.
        return k_nearest(n1, nodes, 1)[0]

    def root_cost(v, tree: Tree[T]) -> float:
        # Return the cost of reaching node ``v`` from the root node
        # (``init_node``) in ``tree``.
        _, c = tree.path(init_node, v)
        return c

    tree = Tree[T](init_node)
    if goal(init_node):
        goal_node = init_node
    else:
        goal_node = None
    if return_fast and goal_node is not None:
        return tree, goal_node

    for _ in range(max_iters):
        nodes = tree.nodes
        v = sample_free()
        nearest_v = nearest(v, nodes)
        new_v = steer(nearest_v, v)
        if check_path(nearest_v, new_v):
            n_nodes = len(nodes)
            k = round(k_rrt * np.log(n_nodes))
            near_v = k_nearest(new_v, nodes, k)

            # Connect along a minimum-cost path.
            min_v = nearest_v
            min_c = root_cost(nearest_v, tree) + cost(nearest_v, new_v)
            for u in near_v:
                c = root_cost(u, tree) + cost(u, new_v)
                if check_path(u, new_v) and c < min_c:
                    min_v = u
                    min_c = c
            tree.add_node(new_v, min_v, cost(min_v, new_v))

            # Rewire the tree.
            for u in near_v:
                c = root_cost(new_v, tree) + cost(new_v, u)
                if check_path(new_v, u) and c < root_cost(u, tree):
                    tree.add_node(u, new_v, cost(new_v, u))

            if goal(new_v):
                goal_node = new_v
            if return_fast and goal_node is not None:
                return tree, goal_node

    if goal_node is not None:
        return tree, goal_node

    raise MaxIterError(
        f"Could not find a node in the goal region after {max_iters} "
        "iterations.")
