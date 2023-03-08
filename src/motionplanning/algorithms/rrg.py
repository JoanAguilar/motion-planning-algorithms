"""Implementation for the RRG algorithm."""

import bisect
from typing import Callable, Iterable, Optional, TypeVar

import numpy as np

from motionplanning.graphs import Graph

T = TypeVar('T')


def rrg(
        sample_free: Callable[[], T],
        cost: Callable[[T, T], float],
        steer: Callable[[T, T], T],
        check_path: Callable[[T, T], bool],
        gamma_rrg: float,
        d: int,
        steer_cost: float,
        iters: int,
        init_nodes: Optional[Iterable[T]] = None
        ) -> Graph[T]:
    """
    Run the RRG (Rapidly-exploring Random Graph) algorithm.

    The RRG algorithm generates a graph (roadmap) spanning the free
    space that can be used for multiple queries. RRG follows a similar
    procedure to RRT, with the major difference that RRG allows cycles
    in the returned graph. For the same same sampling sequence, the
    graphs returned by RRG and RRT will contain the same nodes, with
    the edges returned by RRT being a subset of the ones returned by
    RRG. The implementation follows the one outlined in [1], see there
    for more information.

    Args:
        sample_free: Function to sample a node from the free space.
        cost: Function to compute the cost between two nodes.
        steer: Function to generate a node from one node towards
            another node. ``steer(v0, v1)`` generates a new node,
            ``v2``, which is on the path from ``v0`` towards ``v1``.
            In general, it is expected that ``v2`` will be closer (or
            as close) to ``v0`` than ``v1``.
        check_path: Function to check if the path between two nodes
            is feasible (obstacle-free). "True" means the path is
            feasible.
        gamma_rrg: One of the three parameters (the others being ``d``
            and ``steer_cost``) used to compute the connection radius.
            The connection radius is computed as
            ``min(gamma_rrg * (log(len(V)) / len(V)) ** (1 / d), steer_cost)``,
            where ``len(V)`` is the number of nodes present in the
            computed graph at the current iteration. [1] recommends
            using a value of ``gamma_rrg`` such that
            ``gamma_rrg > 2 * (1 + 1 / d) ** (1 / d) * (mu / zeta) ** (1 / d)``
            where ``mu`` is the Lebesgue measure (that is, the volume)
            of the free space, and ``zeta`` is the volume of the unit
            ball in the ``d``-dimensional Euclidean space.
        d: Dimensionality of the (free) space.
        steer_cost: Maximum cost between the node returned by ``steer``
            and its first argument. That is, ``steer_cost`` is the
            minimum value that satisfies
            ``steer_cost >= dist(v0, steer(v0, v1))`` for any value of
            ``v0`` and ``v1`` in the free space.
        iters: Number of iterations to run the algorithm. The returned
            roadmap will contain ``iters + len(init_nodes)`` nodes.
        init_nodes: Nodes added to the roadmap during initialization.
            These nodes will always be present in the returned roadmap.

    Returns:
        A Graph containing the generated roadmap.

    [1] KARAMAN, Sertac; FRAZZOLI, Emilio. Sampling-based algorithms
    for optimal motion planning. _The international journal of_
    _robotics research,_ 2011, 30.7: 846-894.
    """
    if init_nodes is None:
        init_nodes = []

    def nearest(n1: T, nodes: Iterable[T]) -> T:
        # Return the node nearest to ``n1`` in ``nodes`` in terms of
        # cost.
        nodes_cost_sorted: list[tuple[T, float]] = []
        for n2 in nodes:
            c12 = cost(n1, n2)
            bisect.insort(nodes_cost_sorted, (n2, c12), key=lambda x: x[1])
        return nodes_cost_sorted[0][0]

    def near(n1: T, nodes: Iterable[T], near_cost: float) -> list[T]:
        # Return a list of nodes "near" ``n1``. These are the nodes
        # with a cost less or equal to ``near_cost`` from ``n1``.
        # ``n1`` is included in the list. The returned list is sorted
        # by cost.
        nodes_cost_sorted: list[tuple[T, float]] = []
        for n2 in nodes:
            c12 = cost(n1, n2)
            if c12 <= near_cost:
                bisect.insort(nodes_cost_sorted, (n2, c12), key=lambda x: x[1])
        return [node for node, _ in nodes_cost_sorted]

    graph = Graph[T]()
    for node in init_nodes:
        graph.add_node(node)

    for _ in range(iters):
        nodes = graph.nodes
        v = sample_free()
        nearest_v = nearest(v, nodes)
        new_v = steer(nearest_v, v)
        if check_path(nearest_v, new_v):
            n_nodes = len(nodes)
            r = min(
                gamma_rrg * (np.log(n_nodes) / n_nodes) ** (1 / d),
                steer_cost)
            near_v = near(new_v, nodes, r)
            graph.add_node(new_v)
            graph.add_edge(
                nearest_v,
                new_v,
                cost(nearest_v, new_v),
                twoway=True)
            for u in near_v:
                if check_path(u, new_v):
                    graph.add_edge(u, new_v, cost(u, new_v), twoway=True)
    return graph
