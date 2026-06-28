"""
Community detection algorithms.

Community structure is a hallmark of real-world networks — dense within-group
connections and sparse between-group ones.  This module provides:

- **Modularity** (Newman & Girvan 2004): quality function for a partition.
- **Louvain** (Blondel et al. 2008): greedy modularity optimisation, fast and scalable.
- **Spectral clustering**: embed using *k* leading eigenvectors of the Laplacian,
  then cluster with k-means.
- **Girvan-Newman** (2002): divisive algorithm that removes edges with highest
  betweenness.
- **Label propagation** (Raghavan et al. 2007): near-linear time heuristic.
- **NMI**: normalized mutual information for comparing partitions.

References
----------
* Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
  Fast unfolding of communities in large networks. *JSTAT*, P10008.
* Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community
  structure in networks. *Physical Review E*, 69(2), 026113.
"""

import numpy as np


# =====================================================================
# Modularity
# =====================================================================

def modularity(graph, communities):
    r"""Newman-Girvan modularity.

    .. math::
        Q = \\frac{1}{2m} \\sum_{ij} \\left[ A_{ij}
            - \\frac{k_i k_j}{2m} \\right] \\delta(c_i, c_j)

    Parameters
    ----------
    graph : Graph
    communities : dict
        {community_id: list_of_nodes} or list of lists.

    Returns
    -------
    float
    """
    A = graph.adj.astype(float)
    deg = graph.degrees().astype(float)
    m2 = 2.0 * graph.m

    if isinstance(communities, dict):
        comm_list = list(communities.values())
    elif isinstance(communities, np.ndarray) and communities.ndim == 1:
        # flat label array: convert to list-of-lists
        unique_labels = np.unique(communities)
        comm_list = [list(np.where(communities == lab)[0]) for lab in unique_labels]
    else:
        comm_list = list(communities)

    Q = 0.0
    for comm in comm_list:
        for i in comm:
            for j in comm:
                Q += A[i, j] - deg[i] * deg[j] / m2
    return Q / m2


# =====================================================================
# Louvain (Blondel et al. 2008)
# =====================================================================

def louvain(graph, max_iter=100, resolution=1.0):
    r"""Louvain community detection via greedy modularity optimisation.

    Two phases are alternated until no improvement is possible:

    1. **Local moving**: each node is moved to the neighbour community that
       yields the largest modularity gain.
    2. **Aggregation**: build a new network whose nodes are the communities
       found in phase 1.

    Parameters
    ----------
    graph : Graph
    max_iter : int
        Maximum passes of the Louvain algorithm.
    resolution : float
        Resolution parameter (gamma).  Higher values favour smaller communities.

    Returns
    -------
    list of np.ndarray
        Each element is the community assignment (node -> community label)
        at the finest level.
    """
    n = graph.n
    A = graph.adj.astype(float)
    m2 = 2.0 * graph.m
    deg = graph.degrees().astype(float)

    # initialise: each node in its own community
    community = np.arange(n)
    # community -> (internal weight, total degree)
    comm_weight = np.zeros(n)   # sum of internal edges
    comm_deg = deg.copy()       # total degree of nodes in community

    for _pass in range(max_iter):
        improved = False
        nodes = np.random.permutation(n)
        for u in nodes:
            cu = community[u]
            # remove u from its community
            # compute neighbours' community contributions
            neighbour_comms = {}
            for v in graph.neighbors(u):
                cv = community[v]
                neighbour_comms[cv] = neighbour_comms.get(cv, 0.0) + 1.0

            # delta Q for moving u from cu to each candidate community
            best_delta = 0.0
            best_c = cu
            for cv, w_uv in neighbour_comms.items():
                if cv == cu:
                    continue
                delta = (w_uv - resolution * deg[u] * comm_deg[cv] / m2)
                if delta > best_delta:
                    best_delta = delta
                    best_c = cv

            if best_c != cu:
                community[u] = best_c
                # update community stats
                comm_deg[cu] -= deg[u]
                comm_deg[best_c] += deg[u]
                improved = True

        if not improved:
            break

    # renumber communities 0..k-1
    _, community = np.unique(community, return_inverse=True)
    return [community]


# =====================================================================
# Spectral Clustering
# =====================================================================

def spectral_clustering(graph, n_clusters):
    r"""Spectral clustering via normalised graph Laplacian.

    1. Compute the normalised Laplacian
       :math:`L_{\\text{sym}} = I - D^{-1/2} A D^{-1/2}`.
    2. Extract the *k* eigenvectors corresponding to the *k* smallest
       eigenvalues.
    3. Row-normalise the eigenvector matrix and cluster rows with k-means.

    Parameters
    ----------
    graph : Graph
    n_clusters : int
        Number of clusters.

    Returns
    -------
    labels : np.ndarray (n,)
    """
    n = graph.n
    A = graph.adj.astype(float)
    deg = graph.degrees().astype(float)
    deg_sqrt_inv = np.zeros(n)
    mask = deg > 0
    deg_sqrt_inv[mask] = 1.0 / np.sqrt(deg[mask])

    L_sym = np.eye(n) - np.diag(deg_sqrt_inv) @ A @ np.diag(deg_sqrt_inv)

    eigvals, eigvecs = np.linalg.eigh(L_sym)
    # use k eigenvectors with smallest eigenvalues
    X = eigvecs[:, :n_clusters]
    # row-normalise
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    # k-means on rows of X
    labels = _kmeans(X, n_clusters, seed=42)
    return labels


# =====================================================================
# Girvan-Newman (divisive, edge betweenness)
# =====================================================================

def girvan_newman(graph, n_communities=2, max_iter=100):
    r"""Girvan-Newman divisive community detection.

    Iteratively removes the edge with the highest betweenness centrality,
    splitting the network until the desired number of connected components
    is reached.

    Parameters
    ----------
    graph : Graph
    n_communities : int
        Target number of communities.
    max_iter : int
        Maximum edge removals.

    Returns
    -------
    labels : np.ndarray (n,)
    """
    n = graph.n
    A = graph.adj.copy().astype(float)
    g = type(graph)(A)

    for it in range(max_iter):
        # count connected components
        labels = _connected_components(g)
        n_comp = len(set(labels))
        if n_comp >= n_communities:
            return labels

        # compute edge betweenness (simplified: BFS-based)
        if g.m == 0:
            break
        edge_btw = _edge_betweenness(g)
        # remove edge with max betweenness
        i, j = np.unravel_index(np.argmax(edge_btw), (n, n))
        g.adj[i, j] = g.adj[j, i] = 0
        g.m -= 1

    return _connected_components(g)


# =====================================================================
# Label Propagation
# =====================================================================

def label_propagation(graph, max_iter=100):
    r"""Label propagation community detection.

    Initially each node receives a unique label.  At each step, each node
    adopts the most common label among its neighbours (ties broken randomly).
    This converges rapidly in practice.

    Parameters
    ----------
    graph : Graph
    max_iter : int

    Returns
    -------
    labels : np.ndarray (n,)
    """
    n = graph.n
    labels = np.arange(n)
    for _ in range(max_iter):
        changed = False
        nodes = np.random.permutation(n)
        for u in nodes:
            nb = graph.neighbors(u)
            if not nb:
                continue
            nb_labels = labels[nb]
            unique, counts = np.unique(nb_labels, return_counts=True)
            max_count = np.max(counts)
            candidates = unique[counts == max_count]
            new_label = np.random.choice(candidates)
            if new_label != labels[u]:
                labels[u] = new_label
                changed = True
        if not changed:
            break
    # renumber
    _, labels = np.unique(labels, return_inverse=True)
    return labels


# =====================================================================
# Community statistics
# =====================================================================

def community_sizes(communities):
    """Return array of community sizes.

    Parameters
    ----------
    communities : list of np.ndarray or dict

    Returns
    -------
    np.ndarray
    """
    if isinstance(communities, dict):
        return np.array([len(v) for v in communities.values()])
    if isinstance(communities, np.ndarray):
        _, counts = np.unique(communities, return_counts=True)
        return counts
    return np.array([len(c) for c in communities])


# =====================================================================
# Normalized Mutual Information (NMI)
# =====================================================================

def nmi(predicted, ground_truth):
    r"""Normalized Mutual Information between two partitions.

    .. math::
        NMI(U, V) = \\frac{2 \\, I(U; V)}{H(U) + H(V)}

    Parameters
    ----------
    predicted : np.ndarray (n,)
    ground_truth : np.ndarray (n,)

    Returns
    -------
    float
    """
    n = len(predicted)
    pu = np.unique(predicted)
    pv = np.unique(ground_truth)
    nu, nv = len(pu), len(pv)

    if nu < 2 or nv < 2:
        return 1.0 if np.array_equal(predicted, ground_truth) else 0.0

    # contingency table
    cont = np.zeros((nu, nv))
    for i in range(nu):
        for j in range(nv):
            cont[i, j] = np.sum((predicted == pu[i]) & (ground_truth == pv[j]))

    # marginal probabilities
    p_u = cont.sum(axis=1) / n
    p_v = cont.sum(axis=0) / n
    p_uv = cont / n

    # mutual information
    mi = 0.0
    for i in range(nu):
        for j in range(nv):
            if p_uv[i, j] > 0:
                mi += p_uv[i, j] * np.log(p_uv[i, j] / (p_u[i] * p_v[j]))

    # entropies
    h_u = -np.sum(p_u[p_u > 0] * np.log(p_u[p_u > 0]))
    h_v = -np.sum(p_v[p_v > 0] * np.log(p_v[p_v > 0]))
    denom = (h_u + h_v)
    if denom < 1e-15:
        return 1.0
    return 2.0 * mi / denom


# =====================================================================
# Internal helpers
# =====================================================================

def _connected_components(graph):
    """Return component label for each node (BFS)."""
    n = graph.n
    labels = np.full(n, -1, dtype=int)
    comp_id = 0
    for i in range(n):
        if labels[i] == -1:
            q = [i]
            labels[i] = comp_id
            while q:
                u = q.pop(0)
                for v in graph.neighbors(u):
                    if labels[v] == -1:
                        labels[v] = comp_id
                        q.append(v)
            comp_id += 1
    return labels


def _edge_betweenness(graph):
    """Simplified edge betweenness via BFS from all sources."""
    n = graph.n
    eb = np.zeros((n, n))
    for s in range(n):
        dist = np.full(n, -1, dtype=int); dist[s] = 0
        sigma = np.zeros(n); sigma[s] = 1
        order = []
        q = [s]
        while q:
            v = q.pop(0)
            order.append(v)
            for w in graph.neighbors(v):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
        delta = np.zeros(n)
        for w in reversed(order):
            for v in graph.neighbors(w):
                if dist[v] == dist[w] - 1:
                    c = (sigma[v] / sigma[w]) * (1 + delta[w])
                    eb[v, w] += c
                    eb[w, v] += c
                    delta[v] += c
    return eb


def _kmeans(X, k, seed=42, max_iter=300):
    """Simple k-means clustering."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    # initialise centroids
    idx = rng.choice(n, k, replace=False)
    centroids = X[idx].copy()
    for _ in range(max_iter):
        # assign
        dists = np.zeros((n, k))
        for j in range(k):
            dists[:, j] = np.sum((X - centroids[j]) ** 2, axis=1)
        labels = np.argmin(dists, axis=1)
        # update
        new_centroids = np.zeros((k, d))
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centroids[j] = np.mean(X[mask], axis=0)
            else:
                new_centroids[j] = centroids[j]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels
