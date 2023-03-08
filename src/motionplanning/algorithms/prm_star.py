"""Implementation for the PRM* algorithm."""

from typing import Callable, Hashable, Optional, TypeVar

import numpy as np

from motionplanning.algorithms import sprm
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
