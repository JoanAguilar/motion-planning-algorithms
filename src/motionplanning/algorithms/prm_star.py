"""Implementation for the PRM* and k-PRM* algorithms."""

from typing import Callable, Hashable, Optional, TypeVar

import numpy as np

from motionplanning.algorithms import ksprm, sprm
from motionplanning.graphs import Graph

T = TypeVar('T', bound=Hashable)


def prm_star(
        sample_free: Callable[[], T],
        cost: Callable[[T, T], float],
        check_path: Callable[[T, T], bool],
        gamma_prm: float,
        d: int,
        iters: int,
        init_nodes: Optional[list[T]] = None
        ) -> Graph[T]:
    """
    Run the PRM* (PRM-star) algorithm.

    The PRM* algorithm generates a graph (roadmap) spanning the free
    space that can be used for multiple queries. The generated graph is
    a "forest", a tree of trees. The main difference between PRM* and
    sPRM is that the former connects nodes based on a threshold that
    depends on the parameter ``gamma_prm``, the dimensionality of the
    space ``d``, and the number of iterations/samples ``iter``; while
    the latter uses the provided threshold. The implementation follows
    the one outlined in [1], see there for more information.

    Args:
        sample_free: Function to sample a node from the free space.
        cost: Function to compute the cost between two nodes.
        check_path: Function to check if the path between two nodes
            is feasible (obstacle-free). "True" means the path is
            feasible.
        gamma_prm: One of the three parameters (the others being ``d``
            and ``iters``) used to compute the connection radius. The
            connection radius is computed as
            ``gamma_prm * (log(iters) / iters) ** (1 / d)``.
            [1] recommends using a value of ``gamma_prm`` such that
            ``gamma_prm > 2 * (1 + 1 / d) ** (1 / d) * (mu / zeta) ** (1 / d)``
            where ``mu`` is the Lebesgue measure (that is, the volume)
            of the free space, and ``zeta`` is the volume of the unit
            ball in the ``d``-dimensional Euclidean space.
        d: Dimensionality of the (free) space.
        iters: Number of iterations to run the algorithm. Corresponds
            to the number samples used by the algorithm. The returned
            graph will contain ``iters + len(init_nodes)`` nodes.
        init_nodes: Nodes added to the roadmap during initialization.
            These nodes will always be present in the returned roadmap.

    Returns:
        A Graph containing the generated roadmap.

    [1] KARAMAN, Sertac; FRAZZOLI, Emilio. Sampling-based algorithms
    for optimal motion planning. _The international journal of_
    _robotics research,_ 2011, 30.7: 846-894.
    """
    near_cost = gamma_prm * (np.log(iters) / iters) ** (1 / d)
    return sprm(
        sample_free,
        cost,
        check_path,
        near_cost,
        iters,
        init_nodes=init_nodes)


def kprm_star(
        sample_free: Callable[[], T],
        cost: Callable[[T, T], float],
        check_path: Callable[[T, T], bool],
        iters: int,
        init_nodes: Optional[list[T]] = None,
        k_prm: float = 2 * np.e
        ) -> Graph[T]:
    """
    Run the k-PRM* (k-nearest PRM-star) algorithm.

    The k-PRM* algorithm generates a graph (roadmap) spanning the free
    space that can be used for multiple queries. The major difference
    between k-PRM* and k-sPRM is that the former considers for
    connection a number of nodes that depends on the parameter
    ``k_prm``, and the number of samples ``iters``; while the latter
    considers a given number. The implementation follows the one
    outlined in [1], see there for more information.

    Args:
        sample_free: Function to sample a node from the free space.
        cost: Function to compute the cost between two nodes.
        check_path: Function to check that the path between two nodes
            is feasible (obstacle-free). "True" means the path is
            feasible
        iters: Number of iterations to run the algorithm. Corresponds
            to the number samples used by the algorithm. The returned
            graph will contain ``iters + len(init_nodes)`` nodes.
        init_nodes: Nodes added to the roadmap during initialization.
            These nodes will always be present in the returned roadmap.
        k_prm: One of the two parameters (the other one being
            ``iters``) used to control how many nodes are considered
            for connection. The number of nodes considered is computed
            as ``round(k_prm * log(iters))``.  [1] recommends using a
            value of ``k_prm`` such that ``k_prm > e * (1 + 1 / d)``,
            and notes that ``k_prm = 2 * e`` _is a valid choice for_
            _all problem instances_.

    Returns:
        A Graph containing the generated roadmap.

    [1] KARAMAN, Sertac; FRAZZOLI, Emilio. Sampling-based algorithms
    for optimal motion planning. _The international journal of_
    _robotics research,_ 2011, 30.7: 846-894.
    """
    k = round(k_prm * np.log(iters))
    return ksprm(
        sample_free,
        cost,
        check_path,
        iters,
        init_nodes=init_nodes,
        k=k)
