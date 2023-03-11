"""Contains the ``Graph`` class, used in multiple algorithms."""

import bisect
from typing import Callable, Generic, Hashable, Iterator, Optional, TypeVar

from motionplanning.graphs import NodeError, PathError

T = TypeVar('T', bound=Hashable)


class Graph(Generic[T]):
    """
    A directed graph with nodes of type ``T``.

    Graph edges have an associated cost (which is specified when adding
    an edge).

    The type ``T`` must be hashable.
    """

    def __init__(self) -> None:
        """Initialize the graph with no nodes or edges."""
        self._edges: dict[T, dict[T, float]] = {}

    def add_node(self, node: T) -> None:
        """Add ``node`` to the graph."""
        if node not in self._edges:
            self._edges[node] = {}

    def add_edge(
            self,
            start_node: T,
            end_node: T,
            cost: float,
            twoway: bool = False
            ) -> None:
        """
        Add edge to the graph.

        If any of the nodes are not in the graph, they will be added.
        If the edge was already in the graph, its associated cost will
        be updated.

        Args:
            start_node: Origin node of the edge.
            end_node: Destination node of the edge.
            cost: Cost associated with the edge.
            twoway: If set to True, the reverse edge (from ``end_node``
                to ``start_node``) is also added, with the same cost.
        """
        if start_node not in self._edges:
            self._edges[start_node] = {}
        if end_node not in self._edges:
            self._edges[end_node] = {}
        self._edges[start_node][end_node] = cost

        if twoway:
            self._edges[end_node][start_node] = cost

    def remove_node(self, node: T) -> None:
        """
        Remove ``node`` from the graph.

        Edges associated with the node are also removed.

        Raises:
            NodeError: if ``node`` is not in the graph.
        """
        try:
            del self._edges[node]
        except KeyError:
            raise NodeError(
                f"Cannot remove node ({node}) because it is not in the graph.")
        for n in self._edges:
            try:
                del self._edges[n][node]
            except KeyError:
                pass

    def remove_edge(self, start_node: T, end_node: T) -> None:
        """
        Remove edge from the graph.

        Only edges will be removed when calling this method, nodes will
        remain in the graph, even if completely disconnected.

        Args:
            start_node: Origin node of the edge.
            end_node: Destination node of the edge.

        Raises:
            NodeError: if either ``start_node`` or ``end_node`` are not
                in the graph.
            PathError: if there is no connection between ``start_node``
                and ``end_node``.
        """
        try:
            del self._edges[start_node][end_node]
        except KeyError:
            if start_node not in self._edges:
                raise NodeError(
                    f"Start node ({start_node}) is not in the graph.")
            elif end_node not in self._edges:
                raise NodeError(f"End node ({end_node}) is not in the graph.")
            else:
                raise PathError(
                    f"No connection between the start node ({start_node}) and "
                    f"the end node ({end_node}) exists.")

    def connected(self, n1: T, n2: T) -> bool:
        """
        Check if the nodes ``n1`` and ``n2`` are connected.

        The implementation performs a breadth-first search starting at
        ``n1``, it returns once ``n2`` is found or once all the nodes
        connected to ``n1`` have been explored.

        Returns:
            True if the nodes are connected, False otherwise.

        Raises:
            NodeError: if either ``n1`` or ``n2`` are not in the graph.
        """
        if n1 not in self._edges:
            raise NodeError(f"The node 'n1' ({n1}) is not in the graph.")
        if n2 not in self._edges:
            raise NodeError(f"The node 'n2' ({n2}) is not in the graph.")

        n1_group = set()
        n1_group.add(n1)
        explore_group: set[T] = set()
        explore_group.update(self._edges[n1])
        while len(explore_group) > 0:
            if n2 in n1_group or n2 in explore_group:
                return True
            n = explore_group.pop()
            n1_group.add(n)
            new_explore_nodes = [
                ne for ne in self._edges[n] if ne not in n1_group]
            explore_group.update(new_explore_nodes)
        return False

    def a_star_path(
            self,
            start_node: T,
            goal_node: T,
            heuristic: Callable[[T, T], float]
            ) -> tuple[list[T], float]:
        """
        Find the optimal path between two nodes.

        The implementation uses the A* (A-star) algorithm to find the
        optimal (minimum-cost) path ``start_node`` to ``goal_node``.
        The algorithm is guaranteed to return the optimal path if:

        1. The associated cost for all edges in the graph is positive.

        2. The used heuristic is _admissible_.

        Args:
            start_node: Start node.
            goal_node: Goal node.
            heuristic: Heuristic used by the A* algorithm.

        Returns:
            A tuple with the optimal path as a list of nodes and the
            cost associated with the path.

        Raises:
            NodeError: if either ``start_node`` or ``goal_node`` are
                not in the graph.
            PathError: if a path between ``start_node`` and
                ``goal_node`` cannot be found.
        """
        if start_node not in self._edges:
            raise NodeError(
                f"The starting node ({start_node}) is not in the graph.")
        if goal_node not in self._edges:
            raise NodeError(
                f"The goal node ({goal_node}) is not in the graph.")

        def generate_path(
                explored: dict[T, tuple[Optional[T], float]],
                node: T
                ) -> tuple[list[T], float]:
            # Generate a path from the starting node to ``node``. All
            # the nodes in the path must be in ``explored``. Return the
            # path as a list of nodes, and the cost associated with the
            # path.
            path = [node]
            parent = explored[node][0]
            cost: float = 0
            while parent is not None:
                path.append(parent)
                cost += self._edges[parent][node]
                node = parent
                parent = explored[node][0]
            path.reverse()
            return path, cost

        S = TypeVar('S', bound=Hashable)

        class Frontier(Generic[S]):
            # The ``Frontier`` class is used internally to represent
            # the _frontier_ nodes (sometimes also referred as the
            # _open set_).
            #
            # The class maintains an ordered list and a dictionary with
            # the nodes, its parent nodes, and the expected optimal
            # cost to traverse the graph from ``start_node`` to
            # ``goal_node`` passing by the node in question. The list
            # is sorted based on this cost. The class also provides
            # methods to add, get, and pop nodes; together with some
            # sequence and container magic methods. Maintaining both
            # data structures (a list and a dictionary) allows for
            # efficient access by index and by value (respectively), at
            # the expense of memory usage.

            def __init__(self) -> None:
                self._node_list: list[tuple[T, Optional[T], float]] = []
                self._node_dict: dict[T, tuple[Optional[T], float]] = {}

            def __setitem__(
                    self,
                    node: T,
                    data: tuple[Optional[T], float]
                    ) -> None:
                # Add the node to the frontier. ``data`` must contain
                # the parent and the cost associated with the node. If
                # the node already exists, its associated values are
                # updated.
                try:
                    self.pop(node)
                except KeyError:
                    pass
                parent, cost = data
                bisect.insort(
                    self._node_list,
                    (node, parent, cost),
                    key=lambda x: x[2])
                self._node_dict[node] = (parent, cost)

            def pop_by_index(self, n: int) -> tuple[T, Optional[T], float]:
                # Pop the node at position ``n`` in the list and return
                # itself, its parent, and its associated cost.
                node, parent, cost = self._node_list.pop(n)
                del self._node_dict[node]
                return node, parent, cost

            def get_by_index(self, n: int) -> tuple[T, Optional[T], float]:
                # Get the node at position ``n`` in the list, returns
                # the node itself, its parent, and its associated cost.
                return self._node_list[n]

            def pop(self, node: T) -> tuple[Optional[T], float]:
                # Pop the node, returns its parent and the node's
                # associated cost.
                for i, data in enumerate(self._node_list):
                    n2, parent, cost = data
                    if n2 == node:
                        del self._node_list[i]
                        del self._node_dict[node]
                        return parent, cost
                raise KeyError

            def __getitem__(self, node: T) -> tuple[Optional[T], float]:
                # Get the parent and the associated cost of the node.
                return self._node_dict[node]

            def __iter__(self) -> Iterator[T]:
                for node, _, _ in self._node_list:
                    yield node

            def __len__(self) -> int:
                return len(self._node_list)

            def __contains__(self, node: T) -> bool:
                return node in self._node_dict

        # Initialize explored and frontier nodes.
        explored = {}
        frontier = Frontier[T]()
        frontier[start_node] = (None, heuristic(start_node, goal_node))

        while len(frontier) > 0:
            node, parent, cost = frontier.pop_by_index(0)
            explored[node] = (parent, cost)

            if node == goal_node:
                # An optimal path has been found.
                return generate_path(explored, goal_node)

            # Update explored and frontier nodes.
            for new_node in self._edges[node]:
                new_cost = (
                    cost - heuristic(node, goal_node) +
                    self._edges[node][new_node] +
                    heuristic(new_node, goal_node))
                if new_node in explored and new_cost < explored[new_node][1]:
                    del explored[new_node]
                    frontier[new_node] = (node, new_cost)
                elif new_node in frontier and new_cost < frontier[new_node][1]:
                    frontier[new_node] = (node, new_cost)
                elif new_node not in explored and new_node not in frontier:
                    frontier[new_node] = (node, new_cost)

        raise PathError(
            f"Cannot find a path between nodes ({start_node}) and "
            f"({goal_node}).")

    def direct_connections(self, node: T) -> list[T]:
        """
        Return the list of nodes with an edge from ``node``.

        Raises:
            NodeError: if ``node`` is not in the graph.
        """
        if node not in self._edges:
            raise NodeError(f"Node ({node}) is not in the graph.")
        return list(self._edges[node].keys())

    @property
    def nodes(self) -> set[T]:
        """Return a set of all the nodes in the graph."""
        return set(self._edges)

    @property
    def edges(self) -> dict[tuple[T, T], float]:
        """
        Return the cost of all the edges in the graph.

        Returns:
            A dictionary, the keys are tuples of nodes, representing
            each of the edges of the graph; the values are the cost
            associated with each edge.
        """
        edges_dict = {}
        for n1 in self._edges:
            for n2 in self._edges[n1]:
                edges_dict[(n1, n2)] = self._edges[n1][n2]
        return edges_dict

    def __contains__(self, node: T) -> bool:
        """Return true if ``node`` is in the graph."""
        return node in self._edges
