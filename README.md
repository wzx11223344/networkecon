# NetworkEcon

**Network Economics Toolkit** — A pure-Python (NumPy) library for network economics research and teaching.

Graph theory, centrality measures, diffusion models, peer effects estimation, community detection, and network visualization — all in one package.

## Features

| Module | Description |
|--------|-------------|
| `graph` | Graph data structure, random graph generators (Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, SBM, Core-Periphery) |
| `centrality` | Degree, Betweenness (Brandes), Eigenvector (power iteration), PageRank, Katz, Closeness, HITS, k-core |
| `diffusion` | SIR/SIS epidemic models, Independent Cascade, Linear Threshold, cascade simulation |
| `peer_effects` | Linear-in-means, spatial autoregressive (SAR), 2SLS identification (Bramoulle et al. 2009), Manski reflection problem |
| `community` | Louvain, Spectral Clustering, Girvan-Newman, Label Propagation, modularity, NMI |
| `visualization` | Network plots, degree distributions, adjacency matrix heatmaps, layout algorithms |

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import networkecon as nxe

# Generate a Barabasi-Albert network
g = nxe.barabasi_albert(n=100, m=3)

# Compute centrality
pr = nxe.pagerank(g)

# Simulate SIR epidemic
t, S, I, R = nxe.simulate_sir(g, beta=0.3, gamma=0.1, initial_infected=[0])

# Detect communities
communities = nxe.louvain(g)

# Visualize
nxe.plot_network(g, node_color=pr)
```

Run `python examples/demo.py` for a complete demonstration.

## Requirements

- Python >= 3.8
- NumPy >= 1.20
- Matplotlib >= 3.4 (for visualization)

## License

MIT License. See [LICENSE](LICENSE) for details.
