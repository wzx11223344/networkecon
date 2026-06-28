"""
Diffusion and contagion models on networks.

Network diffusion captures how behaviour, information, or disease spreads
through social and economic connections.  This module implements:

Epidemic models
---------------
- **SIR** (Kermack & McKendrick 1927): Susceptible -> Infected -> Recovered
- **SIS**: Susceptible -> Infected -> Susceptible (no immunity)

Influence models
----------------
- **Independent Cascade** (Goldenberg, Libai & Muller 2001): each newly
  activated node gets one chance to activate each neighbour.
- **Linear Threshold** (Granovetter 1978): a node becomes active when the
  fraction of active neighbours exceeds a threshold.

Theory
------
The SIR epidemic threshold on a network is

.. math::
    \\frac{\\beta}{\\gamma} > \\frac{1}{\\lambda_{\\max}(A)}

where :math:`\\lambda_{\\max}(A)` is the largest eigenvalue of the adjacency
matrix (Pastor-Satorras & Vespignani 2001).
"""

import numpy as np


# =====================================================================
# SIR Model
# =====================================================================

class SIRModel:
    r"""Susceptible-Infected-Recovered compartmental model on a network.

    Parameters
    ----------
    graph : Graph
    beta : float
        Transmission rate per S-I edge.
    gamma : float
        Recovery rate.
    """

    def __init__(self, graph, beta, gamma):
        self.graph = graph
        self.beta = beta
        self.gamma = gamma

    def step(self, state):
        r"""Advance one time-step.

        Each susceptible neighbour of an infected node becomes infected with
        probability :math:`\beta` per edge.  Each infected node recovers with
        probability :math:`\gamma`.

        Parameters
        ----------
        state : np.ndarray (n,)
            0=Susceptible, 1=Infected, 2=Recovered

        Returns
        -------
        np.ndarray
            New state vector.
        """
        n = self.graph.n
        new_state = state.copy()
        for i in range(n):
            if state[i] == 1:  # infected
                if np.random.random() < self.gamma:
                    new_state[i] = 2  # recover
            elif state[i] == 0:  # susceptible
                nb_infected = np.sum(state[self.graph.neighbors(i)] == 1)
                if nb_infected > 0:
                    prob = 1 - (1 - self.beta) ** nb_infected
                    if np.random.random() < prob:
                        new_state[i] = 1
        return new_state


def simulate_sir(graph, beta, gamma, initial_infected, max_time=100):
    r"""Simulate an SIR epidemic on a network.

    Parameters
    ----------
    graph : Graph
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    initial_infected : list of int
        Initially infected node indices.
    max_time : int
        Maximum simulation steps.

    Returns
    -------
    t : np.ndarray
        Time steps (integers).
    S : np.ndarray
        Number of susceptible at each time.
    I : np.ndarray
        Number of infected.
    R : np.ndarray
        Number of recovered.
    """
    n = graph.n
    state = np.zeros(n, dtype=int)
    for idx in initial_infected:
        state[idx] = 1
    model = SIRModel(graph, beta, gamma)
    S, I, R = [], [], []
    for t_step in range(max_time + 1):
        S.append(np.sum(state == 0))
        I.append(np.sum(state == 1))
        R.append(np.sum(state == 2))
        if I[-1] == 0:
            break
        state = model.step(state)
    return (np.arange(len(S)), np.array(S), np.array(I), np.array(R))


def epidemic_threshold(graph):
    r"""Compute the theoretical epidemic threshold.

    Returns :math:`1 / \lambda_{\max}(A)`.

    Parameters
    ----------
    graph : Graph

    Returns
    -------
    float
    """
    eigvals = np.linalg.eigvalsh(graph.adj.astype(float))
    lambda_max = np.max(np.abs(eigvals))
    if lambda_max < 1e-15:
        return float('inf')
    return 1.0 / lambda_max


# =====================================================================
# SIS Model
# =====================================================================

class SISModel:
    r"""Susceptible-Infected-Susceptible model on a network.

    Infected nodes recover with probability :math:`\gamma` and immediately
    become susceptible again (no immunity).

    Parameters
    ----------
    graph : Graph
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    """

    def __init__(self, graph, beta, gamma):
        self.graph = graph
        self.beta = beta
        self.gamma = gamma

    def step(self, state):
        n = self.graph.n
        new_state = state.copy()
        for i in range(n):
            if state[i] == 1:
                if np.random.random() < self.gamma:
                    new_state[i] = 0
            elif state[i] == 0:
                nb_infected = np.sum(state[self.graph.neighbors(i)] == 1)
                if nb_infected > 0:
                    prob = 1 - (1 - self.beta) ** nb_infected
                    if np.random.random() < prob:
                        new_state[i] = 1
        return new_state


# =====================================================================
# Independent Cascade
# =====================================================================

def independent_cascade(graph, seed_nodes, p):
    r"""Independent Cascade (IC) model for influence spread.

    Each newly activated node *u* gets a single chance to activate each
    inactive neighbour *v*.  Activation succeeds independently with
    probability *p*.

    Parameters
    ----------
    graph : Graph
    seed_nodes : list of int
        Initially active nodes.
    p : float
        Activation probability per edge.

    Returns
    -------
    active : np.ndarray (bool)
        Final activation state of each node.
    n_activated : int
        Total number of activated nodes.
    """
    n = graph.n
    active = np.zeros(n, dtype=bool)
    newly_active = list(seed_nodes)
    for s in seed_nodes:
        active[s] = True

    while newly_active:
        u = newly_active.pop(0)
        for v in graph.neighbors(u):
            if not active[v] and np.random.random() < p:
                active[v] = True
                newly_active.append(v)
    return active, int(np.sum(active))


# =====================================================================
# Linear Threshold
# =====================================================================

def linear_threshold(graph, thresholds, seed_nodes):
    r"""Linear Threshold (LT) model.

    Each node *v* has a threshold :math:`\theta_v`.  Node *v* becomes active
    when the fraction of its active neighbours exceeds :math:`\theta_v`.

    Parameters
    ----------
    graph : Graph
    thresholds : np.ndarray (n,)
        Threshold for each node, typically drawn from U[0,1].
    seed_nodes : list of int
        Initially active nodes.

    Returns
    -------
    active : np.ndarray (bool)
    n_activated : int
    """
    n = graph.n
    active = np.zeros(n, dtype=bool)
    for s in seed_nodes:
        active[s] = True
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if active[v]:
                continue
            nb = graph.neighbors(v)
            if len(nb) == 0:
                continue
            frac_active = np.mean(active[nb])
            if frac_active >= thresholds[v]:
                active[v] = True
                changed = True
    return active, int(np.sum(active))


# =====================================================================
# Tipping point & cascade simulation
# =====================================================================

def tipping_point(graph, seed_fraction=0.01, p=0.1, n_trials=50):
    r"""Estimate the tipping point fraction of seeds needed for a global cascade.

    Incrementally increases the seed fraction and records whether a cascade
    reaches a macroscopic size.  Uses the Independent Cascade model.

    Parameters
    ----------
    graph : Graph
    seed_fraction : float
        Starting fraction of seed nodes.
    p : float
        IC activation probability.
    n_trials : int
        Monte Carlo trials per fraction.

    Returns
    -------
    fractions : list of float
        Seed fractions tested.
    cascade_sizes : list of float
        Average cascade size for each fraction.
    """
    n = graph.n
    fractions = []
    sizes = []
    for frac in np.linspace(seed_fraction, 0.5, 20):
        total_size = 0
        for _ in range(n_trials):
            n_seeds = max(1, int(frac * n))
            seeds = list(np.random.choice(n, n_seeds, replace=False))
            _, activated = independent_cascade(graph, seeds, p)
            total_size += activated
        fractions.append(frac)
        sizes.append(total_size / (n_trials * n))
    return fractions, sizes


def simulate_cascade(graph, model='IC', p=0.1, n_simulations=100):
    r"""Run multiple cascade simulations from random single seeds.

    Parameters
    ----------
    graph : Graph
    model : str
        'IC' (Independent Cascade) or 'LT' (Linear Threshold).
    p : float
        Activation probability for IC (ignored for LT).
    n_simulations : int
        Number of random-seed runs.

    Returns
    -------
    sizes : np.ndarray
        Cascade sizes (fraction of nodes) for each simulation.
    """
    n = graph.n
    sizes = np.zeros(n_simulations)
    for sim in range(n_simulations):
        seed = [np.random.randint(0, n)]
        if model == 'IC':
            _, activated = independent_cascade(graph, seed, p)
        else:
            thresholds = np.random.random(n)
            _, activated = linear_threshold(graph, thresholds, seed)
        sizes[sim] = activated / n
    return sizes
