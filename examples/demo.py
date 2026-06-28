#!/usr/bin/env python
"""
NetworkEcon Demo — End-to-end demonstration of all modules.

Generates networks, computes centrality, simulates diffusion,
estimates peer effects, detects communities, and visualises results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

import networkecon as nxe

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_graphs():
    """1. Generate random networks and print properties."""
    print("=" * 60)
    print("1. GRAPH GENERATION")
    print("=" * 60)

    # Erdos-Renyi
    g_er = nxe.er_random_graph(n=100, p=0.1, seed=42)
    print(f"[ER]   {g_er}")
    print(f"       clustering = {g_er.clustering_coefficient():.4f}")
    print(f"       avg path   = {g_er.average_path_length():.2f}")

    # Barabasi-Albert
    g_ba = nxe.barabasi_albert(n=100, m=3, seed=42)
    print(f"[BA]   {g_ba}")
    print(f"       clustering = {g_ba.clustering_coefficient():.4f}")
    print(f"       avg path   = {g_ba.average_path_length():.2f}")

    # Watts-Strogatz
    g_ws = nxe.watts_strogatz(n=100, k=6, p=0.1, seed=42)
    print(f"[WS]   {g_ws}")
    print(f"       clustering = {g_ws.clustering_coefficient():.4f}")
    print(f"       avg path   = {g_ws.average_path_length():.2f}")

    # Stochastic Block Model
    pmat = np.array([[0.3, 0.05],
                     [0.05, 0.3]])
    g_sbm, sbm_labels = nxe.stochastic_block_model(2, [50, 50], pmat, seed=42)
    print(f"[SBM]  {g_sbm}, modularity = {nxe.modularity(g_sbm, sbm_labels):.4f}")

    # Core-Periphery
    g_cp = nxe.core_periphery(20, 80, p_core=0.5, p_peri=0.02, p_cross=0.1, seed=42)
    print(f"[C-P]  {g_cp}")

    # Plot degree distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    nxe.plot_degree_distribution(g_er, ax=axes[0], title='ER G(100, 0.1)')
    nxe.plot_degree_distribution(g_ba, ax=axes[1], title='BA(100, 3)')
    nxe.plot_degree_distribution(g_ws, ax=axes[2], title='WS(100, 6, 0.1)')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'degree_distributions.png'), dpi=150)
    plt.close(fig)
    print("       -> saved degree_distributions.png")

    return g_ba, g_sbm, sbm_labels


def demo_centrality(g):
    """2. Compute all centrality measures."""
    print("\n" + "=" * 60)
    print("2. CENTRALITY MEASURES")
    print("=" * 60)

    deg = nxe.degree_centrality(g)
    btw = nxe.betweenness_centrality(g)
    eig = nxe.eigenvector_centrality(g)
    pr = nxe.pagerank(g, alpha=0.85)
    katz = nxe.katz_centrality(g, alpha=0.01)
    clo = nxe.closeness_centrality(g)
    hubs, auth = nxe.hubs_authorities(g)
    cores = nxe.core_number(g)

    print(f"  Degree       max = {deg.max():.4f},  mean = {deg.mean():.4f}")
    print(f"  Betweenness  max = {btw.max():.4f},  mean = {btw.mean():.4f}")
    print(f"  Eigenvector  max = {eig.max():.4f},  mean = {eig.mean():.4f}")
    print(f"  PageRank     max = {pr.max():.4f},   mean = {pr.mean():.4f}")
    print(f"  Katz         max = {katz.max():.4f},  mean = {katz.mean():.4f}")
    print(f"  Closeness    max = {clo.max():.4f},  mean = {clo.mean():.4f}")
    print(f"  Hubs         max = {hubs.max():.4f},  mean = {hubs.mean():.4f}")
    print(f"  Authorities  max = {auth.max():.4f},  mean = {auth.mean():.4f}")
    print(f"  Core number  max = {cores.max()},     mean = {cores.mean():.2f}")

    # Visualise with PageRank colouring
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    nxe.plot_network(g, node_color=pr, node_size=pr*800+10,
                     title='PageRank', ax=axes[0], cmap='plasma',
                     colorbar_label='PageRank')
    nxe.plot_network(g, node_color=btw, node_size=btw*3000+10,
                     title='Betweenness', ax=axes[1], cmap='plasma',
                     colorbar_label='Betweenness')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'centrality.png'), dpi=150)
    plt.close(fig)
    print("       -> saved centrality.png")


def demo_diffusion(g):
    """3. Simulate SIR epidemic and cascade."""
    print("\n" + "=" * 60)
    print("3. DIFFUSION MODELS")
    print("=" * 60)

    # SIR
    beta, gamma = 0.3, 0.1
    t, S, I, R = nxe.simulate_sir(g, beta, gamma, initial_infected=[0], max_time=50)
    print(f"  SIR (beta={beta}, gamma={gamma}): peak I= {I.max()} ({100*I.max()/g.n:.1f}%)")
    print(f"  Final recovered: {R[-1]} ({100*R[-1]/g.n:.1f}%)")
    threshold = nxe.epidemic_threshold(g)
    print(f"  Epidemic threshold 1/lambda_max = {threshold:.4f}")

    # SIR curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, S, 'b-', label='Susceptible', linewidth=2)
    ax.plot(t, I, 'r-', label='Infected', linewidth=2)
    ax.plot(t, R, 'g-', label='Recovered', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.set_title(f'SIR Epidemic on BA(100,3), $\\beta$={beta}, $\\gamma$={gamma}')
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, 'sir_epidemic.png'), dpi=150)
    plt.close(fig)

    # Independent Cascade
    sizes = nxe.simulate_cascade(g, model='IC', p=0.1, n_simulations=200)
    print(f"  IC (p=0.1): mean cascade size = {sizes.mean():.3f}, "
          f"max = {sizes.max():.3f}")

    # Histogram of cascade sizes
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sizes, bins=20, color='steelblue', edgecolor='white')
    ax.set_xlabel('Cascade size (fraction)')
    ax.set_ylabel('Frequency')
    ax.set_title('Independent Cascade Size Distribution')
    fig.savefig(os.path.join(OUTPUT_DIR, 'cascade_sizes.png'), dpi=150)
    plt.close(fig)

    # Tipping point
    fracs, csizes = nxe.tipping_point(g, p=0.15, n_trials=30)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fracs, csizes, 'o-', color='steelblue')
    ax.set_xlabel('Seed fraction')
    ax.set_ylabel('Avg. cascade size (fraction)')
    ax.set_title('Tipping Point Analysis')
    fig.savefig(os.path.join(OUTPUT_DIR, 'tipping_point.png'), dpi=150)
    plt.close(fig)
    print("       -> saved sir_epidemic.png, cascade_sizes.png, tipping_point.png")


def demo_peer_effects(g):
    """4. Estimate peer effects via 2SLS."""
    print("\n" + "=" * 60)
    print("4. PEER EFFECTS ESTIMATION")
    print("=" * 60)

    n = g.n
    A = g.adj.astype(float)
    deg = g.degrees().astype(float)
    deg[deg == 0] = 1
    G = A / deg[:, None]

    # Generate synthetic data with true peer effects
    X = np.random.randn(n, 2)
    true_beta = 0.3
    epsilon = 0.1 * np.random.randn(n)
    # y = alpha + beta*Gy + X*gamma + epsilon
    # Solve reduced form: y = (I - beta*G)^{-1} (alpha + X*gamma + epsilon)
    gamma_true = np.array([0.5, -0.3])
    alpha_true = 1.0
    rhs = alpha_true + X @ gamma_true + epsilon
    y = np.linalg.solve(np.eye(n) - true_beta * G, rhs)

    # OLS (biased)
    ols_res = nxe.linear_in_means(y, A, X=X)
    print(f"  True beta = {true_beta}")
    print(f"  OLS  beta = {ols_res['beta']:.4f}  (biased)")

    # 2SLS (Bramoulle et al. 2009)
    iv_res = nxe.peer_effect_2sls(y, A, X=X)
    print(f"  2SLS beta = {iv_res['beta']:.4f}  (first-stage R^2 = {iv_res['first_stage_r2']:.4f})")

    # Peer influence test
    test_res = nxe.peer_influence_test(y, A, n_permutations=200)
    print(f"  Peer influence test: observed beta = {test_res['observed_beta']:.4f}, "
          f"p = {test_res['p_value']:.4f}")

    # Reflection problem
    print(f"\n  Reflection Problem:")
    print(f"  {nxe.reflection_problem_explanation()[:200]}...")


def demo_community(g_sbm, sbm_labels):
    """5. Detect communities and compare with ground truth."""
    print("\n" + "=" * 60)
    print("5. COMMUNITY DETECTION")
    print("=" * 60)

    # Louvain
    lv = nxe.louvain(g_sbm)[0]
    print(f"  Louvain: {len(np.unique(lv))} communities, "
          f"NMI = {nxe.nmi(lv, sbm_labels):.4f}")

    # Spectral
    sp = nxe.spectral_clustering(g_sbm, n_clusters=2)
    print(f"  Spectral: {len(np.unique(sp))} clusters, "
          f"NMI = {nxe.nmi(sp, sbm_labels):.4f}")

    # Label propagation
    lp = nxe.label_propagation(g_sbm)
    print(f"  LabelProp: {len(np.unique(lp))} communities, "
          f"NMI = {nxe.nmi(lp, sbm_labels):.4f}")

    # Community sizes
    print(f"  Community sizes: {nxe.community_sizes(lv)}")

    # Visualise
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    nxe.plot_network(g_sbm, node_color=sbm_labels, title='Ground Truth SBM',
                     ax=axes[0], cmap='Set1')
    nxe.plot_network(g_sbm, node_color=lv, title='Louvain',
                     ax=axes[1], cmap='Set2')
    nxe.plot_network(g_sbm, node_color=sp, title='Spectral',
                     ax=axes[2], cmap='Set2')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'communities.png'), dpi=150)
    plt.close(fig)
    print("       -> saved communities.png")


def demo_visualization(g):
    """6. Additional visualizations."""
    print("\n" + "=" * 60)
    print("6. VISUALIZATION")
    print("=" * 60)

    # Adjacency matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    nxe.plot_adjacency_matrix(g, ax=ax, title=f'Adjacency Matrix ({g.n} nodes)')
    fig.savefig(os.path.join(OUTPUT_DIR, 'adjacency.png'), dpi=150)
    plt.close(fig)

    # Different layouts
    layouts = {
        'spring': nxe.spring_layout(g, seed=42),
        'circular': nxe.circular_layout(g),
        'random': nxe.random_layout(g, seed=42),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, pos) in zip(axes, layouts.items()):
        nxe.plot_network(g, pos=pos, title=f'{name} layout', ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'layouts.png'), dpi=150)
    plt.close(fig)
    print("       -> saved adjacency.png, layouts.png")


def main():
    print("NetworkEcon Demo")
    print("================")

    g_ba, g_sbm, sbm_labels = demo_graphs()
    demo_centrality(g_ba)
    demo_diffusion(g_ba)
    demo_peer_effects(g_ba)
    demo_community(g_sbm, sbm_labels)
    demo_visualization(g_ba)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("Demo complete.")


if __name__ == '__main__':
    main()
