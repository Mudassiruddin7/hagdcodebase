"""Hierarchical decomposition of an attribution graph (paper Section III-D, Definition 1).

A multi-resolution sequence H = (G^(0), ..., G^(R)) is built bottom-up.  At level r the
vertices of G^(r) are partitioned by spectral clustering on the smallest eigenvectors of
the normalised Laplacian (Eq. 6)

        L^(r) = I - D^{-1/2} A^(r) D^{-1/2},

and every cluster becomes a supernode of G^(r+1).  With branching factor b each level has
about 1/b of the vertices of the one below, so the depth is R = O(log_b n) (Theorem 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Level:
    n: int
    adj: sp.csr_matrix  # symmetric, zero diagonal, aggregated attribution magnitude
    members: list[np.ndarray]  # leaf (level-0) vertices contained in each vertex
    intra: np.ndarray  # (n,) total edge weight inside each vertex


@dataclass
class Hierarchy:
    levels: list[Level]
    parent: list[np.ndarray]  # parent[r][v] = vertex of level r+1 containing vertex v of level r

    @property
    def depth(self) -> int:
        return len(self.levels) - 1

    def children(self, level: int, v: int) -> np.ndarray:
        """Vertices of ``level - 1`` merged into vertex ``v`` of ``level``."""
        return np.nonzero(self.parent[level - 1] == v)[0]


def normalized_laplacian(adj: sp.spmatrix) -> sp.csr_matrix:
    """Eq. 6.  Isolated vertices get a zero row/column in D^{-1/2} A D^{-1/2}."""
    adj = sp.csr_matrix(adj, dtype=np.float64)
    deg = np.asarray(adj.sum(1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = deg[deg > 0] ** -0.5
    d = sp.diags(inv)
    return sp.identity(adj.shape[0], format="csr") - d @ adj @ d


def spectral_embedding(adj: sp.spmatrix, dim: int) -> np.ndarray:
    """Rows = vertices, columns = the ``dim`` eigenvectors of L with the smallest eigenvalues."""
    n = adj.shape[0]
    dim = int(max(1, min(dim, n - 1)))
    lap = normalized_laplacian(adj)
    if n <= 2500 or dim >= n - 2:
        import scipy.linalg

        _, vecs = scipy.linalg.eigh(lap.toarray(), subset_by_index=[0, dim - 1])
    else:
        import scipy.sparse.linalg

        # smallest eigenvalues of L  <=>  largest of (2I - L); the shift keeps eigsh well-conditioned
        shifted = 2.0 * sp.identity(n, format="csr") - lap
        try:
            _, vecs = scipy.sparse.linalg.eigsh(shifted, k=dim, which="LA", tol=1e-4)
        except Exception:  # noqa: BLE001  (ARPACK non-convergence)
            _, vecs = scipy.sparse.linalg.lobpcg(shifted, np.random.default_rng(0).normal(size=(n, dim)), largest=True, maxiter=200)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def _kmeans(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    k = int(max(1, min(k, len(x))))
    if k == 1:
        return np.zeros(len(x), dtype=np.int64)
    return KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(x).astype(np.int64)


def spectral_partition(adj: sp.spmatrix, n_clusters: int, embed_dim: int, max_size: int, seed: int = 0,
                       order_hint: np.ndarray | None = None) -> np.ndarray:
    """Cluster label per vertex.  Isolated vertices are grouped by ``order_hint`` (e.g. layer)."""
    adj = sp.csr_matrix(adj)
    n = adj.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    deg = np.asarray(adj.sum(1)).ravel()
    conn = np.nonzero(deg > 0)[0]
    iso = np.nonzero(deg <= 0)[0]
    next_label = 0
    if len(conn) > 0:
        frac = len(conn) / n
        k = int(max(1, round(n_clusters * frac)))
        sub = adj[conn][:, conn]
        emb = spectral_embedding(sub, min(embed_dim, max(1, k)))
        lab = _kmeans(emb, k, seed)
        # split clusters that grew far beyond the branching factor
        out = np.zeros(len(conn), dtype=np.int64)
        for c in np.unique(lab):
            idx = np.nonzero(lab == c)[0]
            if len(idx) > max_size:
                parts = _kmeans(emb[idx], int(np.ceil(len(idx) / max(1, max_size // 2))), seed + 1)
                for p in np.unique(parts):
                    out[idx[parts == p]] = next_label
                    next_label += 1
            else:
                out[idx] = next_label
                next_label += 1
        labels[conn] = out
    if len(iso) > 0:
        order = iso if order_hint is None else iso[np.argsort(order_hint[iso], kind="stable")]
        size = max(1, int(round(n / max(1, n_clusters))))
        for start in range(0, len(order), size):
            labels[order[start:start + size]] = next_label
            next_label += 1
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64)


def build_hierarchy(n: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray, branching: int = 4,
                    embed_dim: int = 16, max_cluster_factor: int = 4, seed: int = 0,
                    order_hint: np.ndarray | None = None) -> Hierarchy:
    """Bottom-up agglomerative coarsening until at most ``branching`` supernodes remain."""
    b = max(2, int(branching))
    a = sp.coo_matrix((np.abs(w).astype(np.float64), (src, dst)), shape=(n, n)).tocsr()
    a = a + a.T
    a.setdiag(0)
    a.eliminate_zeros()
    levels = [Level(n, a.tocsr(), [np.array([i]) for i in range(n)], np.zeros(n))]
    parents: list[np.ndarray] = []
    hint = None if order_hint is None else np.asarray(order_hint, dtype=np.float64)
    r = 0
    while levels[-1].n > b:
        cur = levels[-1]
        target = int(np.ceil(cur.n / b))
        labels = spectral_partition(cur.adj, target, embed_dim, max_cluster_factor * b, seed + r, hint)
        k = int(labels.max()) + 1
        if k >= cur.n:  # no coarsening happened: merge consecutive vertices to guarantee progress
            order = np.arange(cur.n) if hint is None else np.argsort(hint, kind="stable")
            labels = np.empty(cur.n, dtype=np.int64)
            labels[order] = np.arange(cur.n) // b
            k = int(labels.max()) + 1
        p = sp.csr_matrix((np.ones(cur.n), (np.arange(cur.n), labels)), shape=(cur.n, k))
        agg = (p.T @ cur.adj @ p).tocsr()
        intra = agg.diagonal() / 2.0 + np.asarray(p.T @ cur.intra).ravel()
        agg.setdiag(0)
        agg.eliminate_zeros()
        members = [np.concatenate([cur.members[v] for v in np.nonzero(labels == c)[0]]) for c in range(k)]
        levels.append(Level(k, agg, members, intra))
        parents.append(labels)
        if hint is not None:
            hint = np.array([hint[labels == c].mean() for c in range(k)])
        r += 1
        if r > 64:
            break
    return Hierarchy(levels, parents)
