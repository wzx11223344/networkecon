"""
NetworkEcon — Network Economics Toolkit
========================================

Graph theory, centrality measures, diffusion models, peer effects estimation,
community detection, and network visualization.
"""

from networkecon.graph import (
    Graph,
    er_random_graph,
    barabasi_albert,
    watts_strogatz,
    stochastic_block_model,
    core_periphery,
)

from networkecon.centrality import (
    degree_centrality,
    betweenness_centrality,
    eigenvector_centrality,
    pagerank,
    katz_centrality,
    closeness_centrality,
    hubs_authorities,
    core_number,
)

from networkecon.diffusion import (
    SIRModel,
    SISModel,
    simulate_sir,
    epidemic_threshold,
    independent_cascade,
    linear_threshold,
    tipping_point,
    simulate_cascade,
)

from networkecon.peer_effects import (
    linear_in_means,
    peer_effect_2sls,
    spatial_autoregressive,
    peer_influence_test,
    reflection_problem_explanation,
)

from networkecon.community import (
    modularity,
    louvain,
    spectral_clustering,
    girvan_newman,
    label_propagation,
    community_sizes,
    nmi,
)

from networkecon.visualization import (
    plot_network,
    plot_degree_distribution,
    plot_adjacency_matrix,
    spring_layout,
    circular_layout,
    random_layout,
)

__version__ = "0.1.0"
__all__ = [
    # graph
    "Graph", "er_random_graph", "barabasi_albert", "watts_strogatz",
    "stochastic_block_model", "core_periphery",
    # centrality
    "degree_centrality", "betweenness_centrality", "eigenvector_centrality",
    "pagerank", "katz_centrality", "closeness_centrality",
    "hubs_authorities", "core_number",
    # diffusion
    "SIRModel", "SISModel", "simulate_sir", "epidemic_threshold",
    "independent_cascade", "linear_threshold", "tipping_point",
    "simulate_cascade",
    # peer_effects
    "linear_in_means", "peer_effect_2sls", "spatial_autoregressive",
    "peer_influence_test", "reflection_problem_explanation",
    # community
    "modularity", "louvain", "spectral_clustering", "girvan_newman",
    "label_propagation", "community_sizes", "nmi",
    # visualization
    "plot_network", "plot_degree_distribution", "plot_adjacency_matrix",
    "spring_layout", "circular_layout", "random_layout",
]
