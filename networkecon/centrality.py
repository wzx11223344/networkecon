"""
Centrality measures for network analysis.

Centrality quantifies the structural importance of each node in a network.
This module implements eight standard measures and one decomposition:

- **Degree**: local connectivity
- **Betweenness** (Brandes 2001): fraction of shortest paths passing through a node
- **Eigenvector** (Bonacich 1987): power iteration on the adjacency matrix
- **PageRank** (Brin & Page 1998): random-walk with teleportation
- **Katz** (1953): weighted sum of walks of all lengths
- **Closeness** (Sabidussi 1966): reciprocal of mean distance to all others
- **HITS / Hubs & Authorities** (Kleinberg 1999): mutual reinforcement
- **k-core decomposition** (Seidman 1983): iterative peeling of low-degree nodes

References
----------
* Brandes, U. (2001). A faster algorithm for betweenness centrality.
  *Journal of Mathematical Sociology*, 25(2), 163-177.
* Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford.
"""

import numpy as np
from collections import deque


# =====================================================================
# Degree centrality
# =====================================================================

def degree_centrality(graph):
    r"""Normalized degree centrality.

    .. math::
        C_D(v) = \\frac{\\deg(v)}{n-1}

    Parameters
    ----------
    graph : Graph

    Returns
    -------
    np.ndarray (n,)
    """
    deg = graph.degrees().astype(float)
    if graph.n > 1:
        deg /= (graph.n - 1)
    return deg


# =====================================================================
# Betweenness centrality  (Brandes 2001)
# =====================================================================

def betweenness_centrality(graph, normalized=True):
    r"""Betweenness centrality via Brandes' algorithm.

    Betweenness of node *v* is the fraction of all-pairs shortest paths that
    pass through *v*:

    .. math::
        C_B(v) = \\sum_{s \\neq v \\neq t} \\frac{\\sigma_{st}(v)}{\\sigma_{st}}

    Uses BFS from each source (unweighted Brandes).

    Parameters
    ----------
    graph : Graph
    normalized : bool
        If True, divide by :math:`(n-1)(n-2)/2` (directed) or
        :math:`(n-1)(n-2)` (undirected).

    Returns
    -------
    np.ndarray (n,)
    """
    n = graph.n
    CB = np.zeros(n)
    for s in range(n):
        # --- single-source shortest paths ---
        S = []                 # stack of nodes in order of non-increasing distance
        P = [set() for _ in range(n)]  # predecessors on shortest paths from s
        sigma = np.zeros(n); sigma[s] = 1
        dist = np.full(n, -1, dtype=int); dist[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            S.append(v)
            for w in graph.neighbors(v):
                if dist[w] < 0:          # first discovery
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:  # shortest path via v
                    sigma[w] += sigma[v]
                    P[w].add(v)
        # --- accumulation ---
        delta = np.zeros(n)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                CB[w] += delta[w]
    if normalized and n > 2:
        CB /= ((n - 1) * (n - 2) / 2)
    return CB


# =====================================================================
# Eigenvector centrality  (Bonacich)
# =====================================================================

def eigenvector_centrality(graph, tol=1e-6, max_iter=1000):
    r"""Eigenvector centrality via the power iteration method.

    A node is central if it is connected to other central nodes:

    .. math::
        \\lambda x = A x

    Parameters
    ----------
    graph : Graph
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    np.ndarray (n,)
    """
    n = graph.n
    x = np.ones(n) / n
    for _ in range(max_iter):
        x_next = graph.adj @ x
        norm = np.linalg.norm(x_next)
        if norm < 1e-15:
            return x_next
        x_next /= norm
        if np.linalg.norm(x_next - x) < tol:
            return x_next
        x = x_next
    return x


# =====================================================================
# PageRank  (Brin & Page)
# =====================================================================

def pagerank(graph, alpha=0.85, tol=1e-6, max_iter=1000):
    r"""Google PageRank.

    Stationary distribution of a random walk that, with probability
    :math:`\\alpha`, follows an outgoing edge, and with probability
    :math:`1-\\alpha` jumps to a random node:

    .. math::
        x = \\alpha A D^{-1} x + (1-\\alpha) \\frac{\\mathbb{1}}{n}

    Parameters
    ----------
    graph : Graph
    alpha : float
        Damping factor (0 < alpha < 1).
    tol : float
        Convergence tolerance.
    max_iter : int

    Returns
    -------
    np.ndarray (n,)
    """
    n = graph.n
    deg = graph.degrees().astype(float)
    deg[deg == 0] = 1.0  # avoid division by zero (dangling nodes)
    M = graph.adj / deg[:, None]  # column-stochastic transition matrix

    x = np.ones(n) / n
    teleport = (1 - alpha) / n
    for _ in range(max_iter):
        x_next = alpha * (M.T @ x) + teleport
        if np.linalg.norm(x_next - x, 1) < tol:
            return x_next
        x = x_next
    return x


# =====================================================================
# Katz centrality
# =====================================================================

def katz_centrality(graph, alpha=0.1, beta=1.0, tol=1e-6, max_iter=1000):
    r"""Katz centrality.

    Counts walks of all lengths, with factor :math:`\\alpha^k` for walks of
    length *k*:

    .. math::
        x = \\alpha A x + \\beta \\mathbb{1}
        \\Rightarrow x = \\beta (I - \\alpha A)^{-1} \\mathbb{1}

    Convergence requires :math:`\\alpha < 1/\\lambda_{\\max}(A)`.

    Parameters
    ----------
    graph : Graph
    alpha : float
        Attenuation factor.
    beta : float
        Exogenous weight.

    Returns
    -------
    np.ndarray (n,)
    """
    n = graph.n
    x = np.ones(n)
    b = np.full(n, beta)
    for _ in range(max_iter):
        x_next = alpha * (graph.adj @ x) + b
        if np.linalg.norm(x_next - x) < tol:
            return x_next
        x = x_next
    return x


# =====================================================================
# Closeness centrality
# =====================================================================

def closeness_centrality(graph):
    r"""Closeness centrality.

    .. math::
        C_C(v) = \\frac{n-1}{\\sum_{u \\neq v} d(v,u)}

    where :math:`d(v,u)` is the shortest-path distance.

    Parameters
    ----------
    graph : Graph

    Returns
    -------
    np.ndarray (n,)
    """
    n = graph.n
    D = graph.shortest_paths()
    cc = np.zeros(n)
    for i in range(n):
        finite = D[i][D[i] < np.inf]
        total = np.sum(finite) - 0.0  # exclude d(i,i)=0
        if total > 0:
            cc[i] = (len(finite) - 1) / total
    return cc


# =====================================================================
# HITS -- Hubs and Authorities  (Kleinberg 1999)
# =====================================================================

def hubs_authorities(graph, tol=1e-6, max_iter=1000):
    r"""HITS (Hyperlink-Induced Topic Search) algorithm.

    Authority scores come from the principal eigenvector of :math:`A^T A`;
    hub scores from the principal eigenvector of :math:`A A^T`.

    Parameters
    ----------
    graph : Graph
    tol : float
    max_iter : int

    Returns
    -------
    hubs : np.ndarray (n,)
    authorities : np.ndarray (n,)
    """
    n = graph.n
    # authority: power iteration on A^T A
    auth = np.ones(n) / n
    for _ in range(max_iter):
        auth_next = graph.adj.T @ (graph.adj @ auth)
        auth_next /= np.linalg.norm(auth_next) or 1.0
        if np.linalg.norm(auth_next - auth) < tol:
            auth = auth_next
            break
        auth = auth_next
    # hubs: from authorities
    hubs = graph.adj @ auth
    hubs /= np.linalg.norm(hubs) or 1.0
    return hubs, auth


# =====================================================================
# k-core decomposition  (Seidman 1983)
# =====================================================================

def core_number(graph):
    r"""k-core decomposition.

    The *k*-core of a graph is the maximal subgraph in which every node has
    degree at least *k* within the subgraph.  The core number of a node is the
    largest *k* for which it belongs to the *k*-core.

    Algorithm: iteratively remove the node with smallest current degree,
    recording its core-ness as that degree.

    Parameters
    ----------
    graph : Graph

    Returns
    -------
    np.ndarray (n,)
        Core number of each node.
    """
    n = graph.n
    deg = graph.degrees().copy()
    core = np.zeros(n, dtype=int)
    # bin-sort for O(n+m) efficiency
    max_deg = int(np.max(deg))
    bins = [[] for _ in range(max_deg + 1)]
    for v in range(n):
        bins[deg[v]].append(v)

    removed = np.zeros(n, dtype=bool)
    for d in range(max_deg + 1):
        while bins[d]:
            v = bins[d].pop()
            if removed[v]:
                continue
            core[v] = d
            removed[v] = True
            for u in graph.neighbors(v):
                if not removed[u]:
                    old_deg = deg[u]
                    deg[u] -= 1
                    if deg[u] < old_deg:
                        bins[old_deg].remove(u)
                        bins[deg[u]].append(u)
    return core
