"""Implementations of the PRM, sPRM, and k-sPRM algorithms."""

import bisect
from typing import Callable, Hashable, Iterable, Optional, TypeVar

from motionplanning.graphs import Graph

T = TypeVar('T', bound=Hashable)


def prm(
        sample_free: Callable[[], T],
        cost: Callable[[T, T], float],
        check_path: Callable[[T, T], bool],
        near_cost: float,
        iters: int,
        init_nodes: Optional[list[T]] = None
        ) -> Graph[T]:
    """
    Run the PRM (Probabilistic RoadMaps) algorithm.

    The PRM algorithm generates a graph (roadmap) spanning the free
    space that can be used for multiple queries. The generated graph is
    a "forest", a tree of trees. The implementation follows the one
    outlined in [1], see there for more information.

    Args:
        sample_free: Function to sample a node from the free space.
        cost: Function to compute the cost between two nodes.
        check_path: Function to check if the path between two nodes
            is feasible (obstacle-free). "True" means the path is
            feasible.
        near_cost: Cost threshold at which nodes are no longer
            considered "near" each other.
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

    def near(n1: T, nodes: Iterable[T]) -> list[T]:
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
        v = sample_free()
        near_v = near(v, graph.nodes)
        graph.add_node(v)
        for u in near_v:
            if not graph.connected(v, u) and check_path(v, u):
                graph.add_edge(v, u, cost(v, u), twoway=True)
    return graph


def sprm(
        sample_free: Callable[[], T],
        cost: Callable[[T, T], float],
        check_path: Callable[[T, T], bool],
        near_cost: float,
        iters: int,
        init_nodes: Optional[list[T]] = None
        ) -> Graph[T]:
    """
    Run the sPRM ("simplified" PRM) algorithm.

    The sPRM algorithm generates a graph (roadmap) spanning the free
    space that can be used for multiple queries. The major difference
    with PRM is that sPRM allows connections between nodes that are
    already part of the same connected component, thus, the generated
    graph is not a "forest" (a tree of trees). The implementation
    follows the one outlined in [1], see there for more information.

    Args:
        sample_free: Function to sample a node from the free space.
        cost: Function to compute the distance between two nodes.
        check_path: Function to check that the path between two nodes
            is feasible (obstacle-free). "True" means the path is
            feasible
        near_cost: Cost threshold at which nodes are no longer
            considered "near" each other.
        iters: Number of iterations to run the algorithm. The resulting
            graph will contain ``iters + len(init_nodes)`` nodes.
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

    def near(n1: T, nodes: Iterable[T]) -> list[T]:
        # Return a list of nodes "near" ``n1``. These are the nodes
        # with a cost less or equal to ``near_cost`` from ``n1``.
        # ``n1`` is not included in the list. The returned list is
        # sorted by cost.
        nodes_cost_sorted: list[tuple[T, float]] = []
        for n2 in nodes:
            c12 = cost(n1, n2)
            if n1 != n2 and c12 <= near_cost:
                bisect.insort(nodes_cost_sorted, (n2, c12), key=lambda x: x[1])
        return [node for node, _ in nodes_cost_sorted]

    graph = Graph[T]()
    for node in init_nodes:
        graph.add_node(node)

    for _ in range(iters):
        graph.add_node(sample_free())
    nodes = graph.nodes
    for v in nodes:
        near_v = near(v, nodes)
        for u in near_v:
            if check_path(v, u):
                graph.add_edge(v, u, cost(v, u), twoway=True)
    return graph


def ksprm(
        sample_free: Callable[[], T],
        cost: Callable[[T, T], float],
        check_path: Callable[[T, T], bool],
        iters: int,
        init_nodes: Optional[list[T]] = None,
        k: int = 15
        ) -> Graph[T]:
    """
    Run the k-sPRM (k-nearest sPRM) algorithm.

    The k-sPRM algorithm generates a graph (roadmap) spanning the free
    space that can be used for multiple queries. The major difference
    with sPRM is that rather than attempting connections between nodes
    under a cost threshold, k-sPRM attemps connections with the ``k``
    neareast nodes (in terms of cost). The implementation follows the
    one outlined in [1], see there for more information.

    Args:
        sample_free: Function to sample a node from the free space.
        cost: Function to compute the cost between two nodes.
        check_path: Function to check that the path between two nodes
            is feasible (obstacle-free). "True" means the path is
            feasible
        iters: Number of iterations to run the algorithm. The resulting
            graph will contain ``iters + len(init_nodes)`` nodes.
        k: The number of nearest nodes to consider for connection. [2]
            reports 15 as a typical value.
        init_nodes: Nodes added to the roadmap during initialization.
            These nodes will always be present in the returned roadmap.

    Returns:
        A Graph containing the generated roadmap.

    [1] KARAMAN, Sertac; FRAZZOLI, Emilio. Sampling-based algorithms
    for optimal motion planning. _The international journal of_
    _robotics research,_ 2011, 30.7: 846-894.

    [2] LAVALLE, Steven M. _Planning algorithms_. Cambridge university
    press, 2006.
    """
    if init_nodes is None:
        init_nodes = []

    def near(n1: T, nodes: Iterable[T]) -> list[T]:
        # Return the ``k`` nearest nodes to ``n1`` from ``nodes``.
        nodes_cost_sorted: list[tuple[T, float]] = []
        for n2 in nodes:
            c12 = cost(n1, n2)
            bisect.insort(nodes_cost_sorted, (n2, c12), key=lambda x: x[1])
        nodes_sorted = []
        for i in range(min(k, len(nodes_cost_sorted))):
            nodes_sorted.append(nodes_cost_sorted[i][0])
        return nodes_sorted

    graph = Graph[T]()
    for node in init_nodes:
        graph.add_node(node)

    for _ in range(iters):
        graph.add_node(sample_free())
    nodes = graph.nodes
    for v in nodes:
        near_v = near(v, nodes)
        for u in near_v:
            if check_path(v, u):
                graph.add_edge(v, u, cost(v, u), twoway=True)
    return graph
