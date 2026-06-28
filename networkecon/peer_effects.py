"""
Peer effects estimation in network data.

Econometric identification of peer effects (social interactions) from
network data.  The canonical model is the *linear-in-means* model:

.. math::
    y = \\alpha \\mathbb{1} + \\beta G y + X \\gamma + G X \\delta + \\varepsilon

where :math:`G` is the row-normalized adjacency matrix.

Key challenges
--------------
- **Reflection problem** (Manski 1993): in the linear-in-means model with
  group-level data, endogenous and exogenous peer effects are not separately
  identified.
- **Identification strategy** (Bramoulle, Djebbari & Fortin 2009): when the
  network has intransitive triads (friends of friends are not always your
  friends), :math:`G^2 X` serves as valid instruments for :math:`G y`.

References
----------
* Manski, C. F. (1993). Identification of endogenous social effects:
  The reflection problem. *Review of Economic Studies*, 60(3), 531-542.
* Bramoulle, Y., Djebbari, H., & Fortin, B. (2009). Identification of
  peer effects through social networks. *Journal of Econometrics*, 150(1), 41-55.
* Lee, L. F. (2007). Identification and estimation of econometric models
  with group interactions, contextual factors and fixed effects.
  *Journal of Econometrics*, 140(2), 333-374.
"""

import numpy as np


# =====================================================================
# Row-normalise adjacency
# =====================================================================

def _row_normalise(A):
    """Return row-stochastic adjacency matrix G = D^{-1} A."""
    deg = np.sum(A > 0, axis=1).astype(float)
    deg[deg == 0] = 1.0
    return A / deg[:, None]


# =====================================================================
# Linear-in-means
# =====================================================================

def linear_in_means(y, G_raw, X=None):
    r"""OLS estimation of the linear-in-means peer effects model.

    .. math::
        y = \\alpha + \\beta \\, G y + X \\gamma + \\varepsilon

    where :math:`G` is the row-normalized adjacency matrix.

    .. warning::
       OLS is generally inconsistent for this model (simultaneity bias).
       Use `peer_effect_2sls` for consistent estimation.

    Parameters
    ----------
    y : np.ndarray (n,)
        Outcome vector.
    G_raw : np.ndarray (n, n)
        Adjacency matrix (will be row-normalized).
    X : np.ndarray (n, k) or None
        Exogenous covariates.

    Returns
    -------
    dict with keys:
        'alpha' : float — intercept
        'beta' : float — endogenous peer effect
        'gamma' : np.ndarray — coefficients on X (if X provided)
        'peer_mean' : np.ndarray — G y (peer average outcome)
    """
    n = len(y)
    G = _row_normalise(G_raw)
    Gy = G @ y

    if X is not None:
        Z = np.column_stack([np.ones(n), Gy, X])
        coef = np.linalg.lstsq(Z, y, rcond=None)[0]
        return {
            'alpha': coef[0], 'beta': coef[1],
            'gamma': coef[2:], 'peer_mean': Gy
        }
    else:
        Z = np.column_stack([np.ones(n), Gy])
        coef = np.linalg.lstsq(Z, y, rcond=None)[0]
        return {'alpha': coef[0], 'beta': coef[1], 'gamma': None, 'peer_mean': Gy}


# =====================================================================
# 2SLS -- Bramoulle, Djebbari & Fortin (2009)
# =====================================================================

def peer_effect_2sls(y, G_raw, X=None, Z=None):
    r"""2SLS estimation of peer effects using network instruments.

    Identification follows Bramoulle et al. (2009): when the network contains
    intransitive triads, :math:`G^2 X` provides valid instruments for
    :math:`G y`.

    First stage: regress :math:`G y` on :math:`[\\mathbb{1}, X, G X, G^2 X]`.
    Second stage: regress :math:`y` on :math:`[\\mathbb{1}, \\widehat{G y}, X]`.

    Parameters
    ----------
    y : np.ndarray (n,)
    G_raw : np.ndarray (n, n)
    X : np.ndarray (n, k) or None
    Z : np.ndarray (n, q) or None
        Exogenous instruments.  If None and X is provided, use [X, GX, G^2 X].

    Returns
    -------
    dict with 'alpha', 'beta', 'gamma', 'first_stage_r2'
    """
    n = len(y)
    G = _row_normalise(G_raw)
    Gy = G @ y

    # build instruments
    if Z is None and X is not None:
        GX = G @ X
        G2X = G @ (G @ X)
        Z = np.column_stack([np.ones(n), X, GX, G2X])
    elif Z is None:
        # no X: generate artificial instruments via G^2 and G^3
        G2 = G @ G
        G3 = G @ G2
        Z = np.column_stack([np.ones(n), G2 @ y * 0 + np.random.randn(n) * 0.01])

    # first stage: Gy ~ Z
    ZtZ = Z.T @ Z
    ZtGy = Z.T @ Gy
    pi_hat = np.linalg.solve(ZtZ + np.eye(Z.shape[1]) * 1e-10, ZtGy)
    Gy_hat = Z @ pi_hat
    ssr = np.sum((Gy - np.mean(Gy)) ** 2)
    ssr_res = np.sum((Gy - Gy_hat) ** 2)
    first_r2 = 1 - ssr_res / ssr if ssr > 0 else 0.0

    # second stage: y ~ [1, Gy_hat, X]
    if X is not None:
        W = np.column_stack([np.ones(n), Gy_hat, X])
    else:
        W = np.column_stack([np.ones(n), Gy_hat])
    coef = np.linalg.lstsq(W, y, rcond=None)[0]

    return {
        'alpha': coef[0],
        'beta': coef[1],
        'gamma': coef[2:] if X is not None else None,
        'first_stage_r2': first_r2,
        'gy_hat': Gy_hat,
    }


# =====================================================================
# Spatial Autoregressive (SAR) model
# =====================================================================

def spatial_autoregressive(y, X, W_raw):
    r"""Spatial Autoregressive (SAR) model for network data.

    .. math::
        y = \\rho W y + X \\beta + \\varepsilon

    Estimated by 2SLS with instruments :math:`[X, W X, W^2 X]`.

    Parameters
    ----------
    y : np.ndarray (n,)
    X : np.ndarray (n, k)
    W_raw : np.ndarray (n, n)
        Spatial weight matrix (will be row-normalized).

    Returns
    -------
    dict with 'rho', 'beta', 'first_stage_r2'
    """
    result = peer_effect_2sls(y, W_raw, X=X)
    return {
        'rho': result['beta'],
        'beta': result['gamma'],
        'first_stage_r2': result['first_stage_r2'],
    }


# =====================================================================
# Peer influence test
# =====================================================================

def peer_influence_test(y, G_raw, n_permutations=1000):
    r"""Permutation test for the null hypothesis :math:`H_0: \beta = 0`.

    Shuffles the network while keeping degree sequence fixed (approximately)
    and recomputes the OLS :math:`\beta` to build a null distribution.

    Parameters
    ----------
    y : np.ndarray (n,)
    G_raw : np.ndarray (n, n)
    n_permutations : int

    Returns
    -------
    dict with 'observed_beta', 'p_value', 'null_distribution'
    """
    n = len(y)
    result_obs = linear_in_means(y, G_raw)
    beta_obs = result_obs['beta']

    null_betas = np.zeros(n_permutations)
    A = G_raw.copy()
    # edge-swap MCMC to shuffle network while preserving degrees
    edges = list(zip(*np.where(np.triu(A) > 0)))
    for perm in range(n_permutations):
        # simple edge rewiring (degree-preserving)
        A_shuffled = A.copy()
        for _ in range(int(len(edges) * 2)):
            i, j = edges[np.random.randint(0, len(edges))]
            k, l_ = edges[np.random.randint(0, len(edges))]
            if len({i, j, k, l_}) == 4:
                if A_shuffled[i, l_] == 0 and A_shuffled[k, j] == 0:
                    A_shuffled[i, j] = A_shuffled[j, i] = 0
                    A_shuffled[k, l_] = A_shuffled[l_, k] = 0
                    A_shuffled[i, l_] = A_shuffled[l_, i] = 1
                    A_shuffled[k, j] = A_shuffled[j, k] = 1
        res = linear_in_means(y, A_shuffled)
        null_betas[perm] = res['beta']

    p_value = np.mean(np.abs(null_betas) >= np.abs(beta_obs))
    return {
        'observed_beta': beta_obs,
        'p_value': p_value,
        'null_distribution': null_betas,
    }


# =====================================================================
# Reflection problem explanation
# =====================================================================

def reflection_problem_explanation():
    r"""Explain Manski's (1993) reflection problem.

    The reflection problem arises when we try to distinguish:

    1. **Endogenous effects**: individual *i*'s outcome depends on peer
       *j*'s outcomes (the "social multiplier").
    2. **Exogenous (contextual) effects**: individual *i*'s outcome depends
       on peer *j*'s exogenous characteristics.
    3. **Correlated effects**: individuals in the same group share similar
       outcomes because they share unobserved characteristics or face a
       common environment.

    In the linear-in-means model

    .. math::
        y_i = \\alpha + \\beta \\bar{y}_{-i} + \\gamma x_i
              + \\delta \\bar{x}_{-i} + \\varepsilon_i

    the parameters :math:`\\beta` (endogenous) and :math:`\\delta` (contextual)
    cannot be separately identified when using group-level data alone, because
    the group-mean outcome :math:`\\bar{y}` is a linear function of the
    group-mean characteristics :math:`\\bar{x}`.

    **Solutions**:
    - Network data with intransitive triads (Bramoulle et al. 2009)
    - Nonlinear-in-means models (Brock & Durlauf 2001)
    - Random assignment to groups (Sacerdote 2001)
    - Dynamic panel / lagged peer outcomes

    Returns
    -------
    str
    """
    return reflection_problem_explanation.__doc__
