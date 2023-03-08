"""Errors used by graphs and trees."""


class NodeError(Exception):
    """Raised when a node is not on the graph."""


class PathError(Exception):
    """Raised when a path between nodes cannot be found."""
