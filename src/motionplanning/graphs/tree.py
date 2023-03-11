"""Contains the ``Tree`` class, used in multiple algorithms."""

from typing import Generic, Hashable, Optional, TypeVar

from motionplanning.graphs import Graph, NodeError, PathError

T = TypeVar('T', bound=Hashable)


class Tree(Generic[T]):
    """
    A fully connected tree with nodes of type ``T``.

    Tree edges have an associated cost (which is specified when adding
    a node).

    The type ``T`` must be hashable.
    """

    def __init__(self, root: T) -> None:
        """
        Initialize the tree with a single root node.

        Args:
            root: The root node.
        """
        # The implementation keeps track of connections using
        # ``parents``, which stores connections from children to
        # parents and the cost associated with the connection.
        self._parents: dict[T, Optional[tuple[T, float]]] = {root: None}

    def add_node(self, node: T, parent: T, cost: float) -> None:
        """
        Add a node to the tree.

        Note that if ``node`` is already in the tree, the connection
        with its parent and its associated cost will be updated (the
        tree will be "rewired").

        Args:
            node: Node to add.
            parent: Parent of the node to add.
            cost: Cost associated with the ``parent``-to-``node``
            connection.

        Raises:
            NodeError: if ``parent`` is not in the tree.
        """
        if parent not in self._parents:
            raise NodeError(
                f"Parent node ({parent}) is not in the tree.")

        self._parents[node] = (parent, cost)

    def path(self, start_node: T, end_node: T) -> tuple[list[T], float]:
        """
        Find the path from ``start_node`` to ``end_node``.

        Note that this method only searches for paths that go in the
        root-to-leaves direction (that is, for this method to return a
        path, ``start_node`` must be closer to the root than
        ``end_node``).

        Returns:
            A tuple with the path as a list of nodes and the cost
            associated with the path.

        Raises:
            NodeError: if either ``start_node`` or ``end_node`` are not
                in the tree.
            PathError: if ``start_node`` and ``end_node`` are in the
                tree but a path from ``start_node`` to ``end_node``
                does not exist.
        """
        if start_node not in self._parents:
            raise NodeError(
                f"The start node ({start_node}) is not in the graph.")
        if end_node not in self._parents:
            raise NodeError(f"The end node ({end_node}) is not in the graph.")

        node = end_node
        path = [end_node]
        path_cost: float = 0
        while True:
            if node == start_node:
                path.reverse()
                return path, path_cost
            node_cost = self._parents[node]
            if node_cost is None:
                raise PathError(
                    f"No path exists between the start node ({start_node}) "
                    f"and the end node ({end_node}).")
            node, cost = node_cost
            path.append(node)
            path_cost += cost

    def as_graph(self) -> Graph[T]:
        """Return the tree as a Graph."""
        graph = Graph[T]()
        edges = self.edges
        for edge, cost in edges.items():
            graph.add_edge(edge[0], edge[1], cost)
        return graph

    @property
    def nodes(self) -> set[T]:
        """Return a set of all the nodes in the tree."""
        return set(self._parents)

    @property
    def edges(self) -> dict[tuple[T, T], float]:
        """
        Return the cost of all the edges in the tree.

        Returns:
            A dictionary, the keys are tuples of nodes, representing
            each of the edges of the tree, from parent to children; the
            values are the cost associated with each edge.
        """
        edges = {}
        for child in self._parents:
            parent_cost = self._parents[child]
            if parent_cost is not None:
                parent, cost = parent_cost
                edges[(parent, child)] = cost
        return edges

    def __contains__(self, node: T) -> bool:
        """Return true if ``node`` is in the tree."""
        return node in self._parents
