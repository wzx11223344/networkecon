"""
Graph data structure and random network generators.

Provides a lightweight Graph class backed by a NumPy adjacency matrix,
plus classical random graph models used in network economics.

Theory
------
- Erdos-Renyi G(n, p): Each edge exists independently with probability p.
- Barabasi-Albert: Preferential attachment yields scale-free degree distribution P(k) ~ k^{-3}.
- Watts-Strogatz: Rewiring a ring lattice introduces small-world properties.
- Stochastic Block Model (SBM): Partition nodes into blocks with block-level connection probabilities.
- Core-Periphery: Dense core plus sparse periphery, common in financial and trade networks.
"""

import numpy as np
from collections import deque


class Graph:
    """Undirected graph with optional node/edge attributes.

    The adjacency matrix A is symmetric (A_{ij} = A_{ji}), with A_{ij} = 1
    indicating an edge between nodes i and j.

    Parameters
    ----------
    adjacency : np.ndarray (n, n)
        Symmetric adjacency matrix.

    Attributes
    ----------
    n : int
        Number of nodes.
    m : int
        Number of edges (undirected).
    adj : np.ndarray (n, n)
        Adjacency matrix.
    node_attrs : dict
        Optional node-level attributes.
    edge_attrs : dict
        Optional edge-level attributes (keyed by (i, j) tuples with i < j).
    """

    def __init__(self, adjacency):
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Adjacency must be a square matrix.")
        self.adj = adjacency.copy()
        self.n = self.adj.shape[0]
        self.m = int(np.sum(self.adj > 0) // 2)
        self.node_attrs = {}
        self.edge_attrs = {}

    # ---- properties -------------------------------------------------------

    def density(self):
        r"""Return graph density.

        .. math::
            d = \\frac{2m}{n(n-1)}
        """
        if self.n < 2:
            return 0.0
        return 2.0 * self.m / (self.n * (self.n - 1))

    def degrees(self):
        """Return degree of each node (1D array of length n)."""
        return np.sum(self.adj > 0, axis=1)

    def degree_distribution(self):
        """Return empirical degree distribution.

        Returns
        -------
        k : np.ndarray
            Unique degree values.
        pk : np.ndarray
            Fraction of nodes with each degree.
        """
        deg = self.degrees()
        unique, counts = np.unique(deg, return_counts=True)
        return unique, counts / self.n

    def neighbors(self, node):
        """Return list of neighbor indices for a given node."""
        return list(np.where(self.adj[node] > 0)[0])

    def clustering_coefficient(self):
        r"""Average clustering coefficient (transitivity).

        .. math::
            C = \\frac{1}{n}\\sum_i \\frac{2t_i}{k_i(k_i-1)}

        where t_i is the number of triangles through node i.
        """
        c = np.zeros(self.n)
        for i in range(self.n):
            nb = self.neighbors(i)
            k = len(nb)
            if k < 2:
                c[i] = 0.0
                continue
            sub = self.adj[np.ix_(nb, nb)]
            tri = np.sum(sub > 0) / 2
            c[i] = 2.0 * tri / (k * (k - 1))
        return float(np.mean(c))

    def average_path_length(self):
        """Average shortest-path length over all connected node pairs.

        Uses BFS from each node.  Returns inf if the graph is disconnected.
        """
        total_dist = 0
        reachable = 0
        for i in range(self.n):
            dist = self._bfs_distances(i)
            finite = dist[dist < np.inf]
            reachable += len(finite) - 1  # exclude self
            total_dist += np.sum(finite) - 0.0  # d(i,i)=0
        if reachable == 0:
            return float('inf')
        return total_dist / reachable

    def _bfs_distances(self, source):
        dist = np.full(self.n, np.inf)
        dist[source] = 0.0
        q = deque([source])
        while q:
            u = q.popleft()
            for v in self.neighbors(u):
                if np.isinf(dist[v]):
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def shortest_paths(self):
        """All-pairs shortest path matrix (BFS). Returns (n, n) float array."""
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            D[i] = self._bfs_distances(i)
        return D

    def copy(self):
        """Return a deep copy of the graph."""
        g = Graph(self.adj)
        g.node_attrs = {k: v.copy() if hasattr(v, 'copy') else v
                        for k, v in self.node_attrs.items()}
        g.edge_attrs = dict(self.edge_attrs)
        return g

    def __repr__(self):
        return f"Graph(n={self.n}, m={self.m}, density={self.density():.4f})"


# =====================================================================
# Random graph generators
# =====================================================================

def er_random_graph(n, p, seed=None):
    r"""Erdos-Renyi G(n, p) random graph.

    Each of the :math:`\\binom{n}{2}` possible edges exists independently with
    probability *p*.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Edge probability.
    seed : int, optional
        Random seed.

    Returns
    -------
    Graph
    """
    rng = np.random.default_rng(seed)
    # upper triangle only, then symmetrize
    A = np.triu(rng.random((n, n)) < p, k=1)
    A = A + A.T
    return Graph(A)


def barabasi_albert(n, m, seed=None):
    r"""Barabasi-Albert preferential attachment network.

    Nodes arrive one by one; each new node connects to *m* existing nodes with
    probability proportional to their current degree.

    This yields a scale-free degree distribution :math:`P(k) \\propto k^{-3}`.

    Parameters
    ----------
    n : int
        Final number of nodes.
    m : int
        Number of edges per new node (must be <= starting clique size).
    seed : int, optional
        Random seed.

    Returns
    -------
    Graph
    """
    if m < 1 or m >= n:
        raise ValueError("Require 1 <= m < n")
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=int)
    # start with a clique of size m+1
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            A[i, j] = A[j, i] = 1
    # track list of edge endpoints for sampling
    endpoints = []
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            endpoints.extend([i, j])
    for new_node in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            candidate = endpoints[rng.integers(0, len(endpoints))]
            if candidate != new_node and candidate not in targets:
                targets.add(candidate)
        for t in targets:
            A[new_node, t] = A[t, new_node] = 1
            endpoints.extend([new_node, t, t, new_node])
    return Graph(A)


def watts_strogatz(n, k, p, seed=None):
    r"""Watts-Strogatz small-world network.

    Start with a ring lattice where each node connects to its *k* nearest
    neighbours, then rewire each edge with probability *p*.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node is connected to *k* nearest neighbours (must be even).
    p : float
        Rewiring probability.
    seed : int, optional
        Random seed.

    Returns
    -------
    Graph
    """
    if k % 2 != 0:
        raise ValueError("k must be even.")
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=int)
    # ring lattice
    half_k = k // 2
    for i in range(n):
        for j in range(1, half_k + 1):
            A[i, (i + j) % n] = 1
            A[(i + j) % n, i] = 1
    # rewire
    for i in range(n):
        for j in range(i + 1, i + half_k + 1):
            nb = j % n
            if A[i, nb] and rng.random() < p:
                A[i, nb] = A[nb, i] = 0
                while True:
                    new_target = rng.integers(0, n)
                    if new_target != i and A[i, new_target] == 0:
                        A[i, new_target] = A[new_target, i] = 1
                        break
    return Graph(A)


def stochastic_block_model(n_blocks, block_sizes, pref_matrix, seed=None):
    r"""Stochastic Block Model (SBM).

    Nodes are partitioned into *n_blocks* communities.  An edge between a node
    in block *r* and a node in block *s* exists with probability
    :math:`P_{rs}`, given by the preference matrix.

    Parameters
    ----------
    n_blocks : int
        Number of blocks.
    block_sizes : list of int
        Size of each block.
    pref_matrix : np.ndarray (n_blocks, n_blocks)
        Block-level connection probabilities.
    seed : int, optional
        Random seed.

    Returns
    -------
    graph : Graph
    labels : np.ndarray
        Block label for each node.
    """
    rng = np.random.default_rng(seed)
    sizes = np.asarray(block_sizes, dtype=int)
    n = int(np.sum(sizes))
    labels = np.repeat(np.arange(n_blocks), sizes)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = labels[i], labels[j]
            if rng.random() < pref_matrix[bi, bj]:
                A[i, j] = A[j, i] = 1
    return Graph(A), labels


def core_periphery(n_core, n_periphery, p_core, p_peri, p_cross, seed=None):
    r"""Core-periphery network.

    Core nodes are densely connected among themselves; periphery nodes connect
    sparsely to the core and among themselves.

    Parameters
    ----------
    n_core : int
        Number of core nodes.
    n_periphery : int
        Number of periphery nodes.
    p_core : float
        Edge probability within the core.
    p_peri : float
        Edge probability within the periphery.
    p_cross : float
        Edge probability between core and periphery.
    seed : int, optional
        Random seed.

    Returns
    -------
    Graph
    """
    rng = np.random.default_rng(seed)
    n = n_core + n_periphery
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if i < n_core and j < n_core:
                prob = p_core
            elif i >= n_core and j >= n_core:
                prob = p_peri
            else:
                prob = p_cross
            if rng.random() < prob:
                A[i, j] = A[j, i] = 1
    return Graph(A)
