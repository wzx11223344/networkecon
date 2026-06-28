"""
Network visualization using Matplotlib.

Provides ready-to-use plotting functions for network graphs, degree
distributions, and adjacency matrices.  Layout algorithms compute 2D
positions for meaningful and aesthetically pleasing node placement.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# =====================================================================
# Layout algorithms
# =====================================================================

def spring_layout(graph, iterations=50, k=None, seed=None):
    r"""Fruchterman-Reingold force-directed layout.

    Nodes repel each other like charged particles; edges act as springs
    pulling connected nodes together.  Minimises the energy

    .. math::
        E = \\sum_i \\sum_{j>i} \\frac{k^2}{\\|p_i - p_j\\|}
            - \\sum_{(i,j)\\in E} \\frac{\\|p_i - p_j\\|^2}{k}

    Parameters
    ----------
    graph : Graph
    iterations : int
        Number of iterations.
    k : float, optional
        Optimal distance between nodes. Default: sqrt(area / n).
    seed : int, optional

    Returns
    -------
    pos : np.ndarray (n, 2)
    """
    n = graph.n
    rng = np.random.default_rng(seed)
    area = n * 1.0
    if k is None:
        k = np.sqrt(area / n)
    t = np.sqrt(area) * 0.1   # initial temperature
    dt = t / (iterations + 1)

    pos = rng.uniform(-1, 1, (n, 2)) * np.sqrt(area) * 0.5
    disp = np.zeros((n, 2))

    for _ in range(iterations):
        disp.fill(0)
        # repulsion
        for i in range(n):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1)
            dist = np.maximum(dist, 1e-6)
            rep = (k ** 2) / (dist ** 2)
            disp[i] += np.sum(delta / dist[:, None] * rep[:, None], axis=0)
        # attraction
        for i in range(n):
            for j in graph.neighbors(i):
                if j > i:
                    delta = pos[j] - pos[i]
                    dist = np.linalg.norm(delta)
                    if dist > 0:
                        disp[i] += delta * (dist / k)
                        disp[j] -= delta * (dist / k)
        # update positions
        for i in range(n):
            d_norm = np.linalg.norm(disp[i])
            if d_norm > 0:
                pos[i] += (disp[i] / d_norm) * min(d_norm, t)
        # cool
        t -= dt

    # centre
    pos -= np.mean(pos, axis=0)
    return pos


def circular_layout(graph):
    """Place nodes evenly on a circle.

    Returns
    -------
    pos : np.ndarray (n, 2)
    """
    n = graph.n
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angles), np.sin(angles)])


def random_layout(graph, seed=None):
    """Random node positions in the unit square.

    Returns
    -------
    pos : np.ndarray (n, 2)
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (graph.n, 2))


# =====================================================================
# Plotting
# =====================================================================

def plot_network(graph, pos=None, node_color=None, node_size=None,
                 edge_alpha=0.3, title=None, ax=None, cmap='viridis',
                 colorbar_label=None):
    """Draw the full network.

    Parameters
    ----------
    graph : Graph
    pos : np.ndarray (n, 2), optional
        Node positions.  Uses spring_layout if None.
    node_color : array-like, optional
        Node-level values for colour mapping.
    node_size : array-like, optional
        Node-level sizes.
    edge_alpha : float
        Edge transparency.
    title : str, optional
    ax : matplotlib Axes, optional
    cmap : str
    colorbar_label : str, optional
    """
    if pos is None:
        pos = spring_layout(graph, seed=42)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    n = graph.n
    A = graph.adj

    # edges
    for i in range(n):
        for j in graph.neighbors(i):
            if j > i:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        'k-', alpha=edge_alpha, linewidth=0.5)

    # nodes
    if node_size is None:
        deg = graph.degrees().astype(float)
        node_size = 20 + 80 * (deg - deg.min()) / (deg.max() - deg.min() + 1)
    if node_color is None:
        node_color = 'steelblue'

    sc = ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c=node_color,
                    cmap=cmap, edgecolors='white', linewidth=0.5, zorder=5)

    if not isinstance(node_color, str):
        cbar = plt.colorbar(sc, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title)
    return ax


def plot_degree_distribution(graph, log_scale=True, ax=None, title=None):
    """Plot the degree distribution P(k).

    Parameters
    ----------
    graph : Graph
    log_scale : bool
        Use log-log axes (recommended for scale-free networks).
    ax : matplotlib Axes, optional
    title : str
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    k, pk = graph.degree_distribution()
    ax.scatter(k, pk, s=30, color='steelblue', edgecolors='white', linewidth=0.5)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('Degree k')
    ax.set_ylabel('P(k)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Degree Distribution')
    return ax


def plot_adjacency_matrix(graph, ax=None, title=None):
    """Plot the adjacency matrix as a heatmap.

    Parameters
    ----------
    graph : Graph
    ax : matplotlib Axes, optional
    title : str
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(graph.adj, cmap='binary', interpolation='none', aspect='equal')
    ax.set_xlabel('Node')
    ax.set_ylabel('Node')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Adjacency Matrix')
    plt.colorbar(im, ax=ax, label='Edge')
    return ax
