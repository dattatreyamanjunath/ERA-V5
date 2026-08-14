"""
Reversible Kronecker Embedding (RKE)
====================================

A drop-in replacement for nn.Embedding with the same complexity profile as
Kronecker Embeddings (arXiv:2605.29459) plus *exact* closed-form invertibility:
given e = encode(bytes), decode(e) returns the original bytes with no error.

Construction
------------
    e(b) = flat( T @ Z(b) ) @ Q,      Z(b)[p,:] = g_{b_p} / sqrt(L)

  G in R^{256 x s}   codebook, rows unit-norm and pairwise distinct
  T in R^{P x P}     invertible circulant position mixer, T = F* diag(w) F, w > 0
  Q in O(d)          orthogonal output mixer (product of Householder reflections)
  d = P * s

Invertibility (exact, unconditional):
  Q orthogonal    -> undo by Q^T
  T invertible    -> undo by T^-1
  equal-norm rows -> L recovered from slot norms
  distinct rows   -> b_p = argmax_v <Z[p,:], g_v>   is exact by Cauchy-Schwarz

Robustness: decoding is provably exact for any perturbation with
  ||eta||_2 < r (1 - mu) / (2 sqrt(L) ||T^-1||_2),
  mu = max_{u!=v} <g_u,g_v>/r^2   (coherence; Welch bound sqrt((256-s)/(255 s)))

Training: keep the guarantees as hard invariants --
  G  -> renormalize rows after every step (Stiefel-like product of spheres)
  w  -> w_k = w_min + softplus(theta_k)   (hard conditioning floor)
  Q  -> parameterize as a product of Householder reflections (exactly orthogonal)
Optionally add a coherence penalty  lambda * ||off-diag(G G^T)||_F^2  to push mu
toward the Welch bound and widen the decoding margin.
"""

import numpy as np

N_BYTE = 256


# ----------------------------------------------------------------- components
def build_codebook(s, n=N_BYTE, iters=3000, lr=0.5, seed=0):
    """n unit vectors in R^s with (approximately) minimal coherence."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, s))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    for _ in range(iters):
        M = G @ G.T
        np.fill_diagonal(M, 0.0)
        A = np.abs(M)
        W = np.exp(60.0 * (A - A.max()))
        W /= W.sum()
        G -= lr * 2.0 * ((W * np.sign(M)) @ G)
        G /= np.linalg.norm(G, axis=1, keepdims=True)
    M = G @ G.T
    np.fill_diagonal(M, 0.0)
    return G, float(np.abs(M).max())


def welch_bound(n=N_BYTE, s=128):
    return float(np.sqrt((n - s) / ((n - 1) * s)))


def build_T(P, chi=8.0, width=0.55):
    """Symmetric real circulant with spectrum in [1/chi, 1].

    The position-similarity profile is rho = IDFT(w**2): rho(delta) is the
    cosine between a token and the same token shifted by delta. chi is the
    exact condition number, so it is also the exact factor by which the
    decoding margin shrinks. Set chi = 1 to recover hard position slots.
    """
    k = np.arange(P)
    kk = np.minimum(k, P - k)
    w = np.exp(-((kk / (width * P / 2)) ** 2))
    w = np.maximum(w / w.max(), 1.0 / chi)
    F = np.fft.fft(np.eye(P), axis=0) / np.sqrt(P)
    return np.real(F.conj().T @ np.diag(w) @ F), w


def build_Q(d, m=64, seed=0):
    """Exactly orthogonal matrix as a product of m Householder reflections."""
    rng = np.random.default_rng(seed)
    Q = np.eye(d)
    for _ in range(m):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        Q -= 2.0 * np.outer(Q @ v, v)
    return Q


# ----------------------------------------------------------------- the module
class ReversibleKroneckerEmbedding:
    def __init__(self, d_model, max_bytes=32, chi=8.0, n_householder=64, seed=0):
        if d_model % max_bytes:
            raise ValueError("d_model must be divisible by max_bytes")
        self.d, self.P = d_model, max_bytes
        self.s = d_model // max_bytes
        self.G, self.mu = build_codebook(self.s, seed=seed)
        self.T, self.w = build_T(max_bytes, chi)
        self.Tinv = np.linalg.inv(self.T)
        self.Q = build_Q(d_model, n_householder, seed=seed)

    # -- diagnostics ---------------------------------------------------------
    @property
    def margin(self):
        """Guaranteed-exact perturbation radius at worst-case length P."""
        return (1.0 - self.mu) / (2.0 * np.sqrt(self.P) * self.w.max() / self.w.min())

    @property
    def rho(self):
        r = np.real(np.fft.ifft(self.w ** 2))
        return r / r[0]

    def n_params(self, n_householder=64):
        return N_BYTE * self.s + self.P + n_householder * self.d

    # -- forward / inverse ---------------------------------------------------
    def encode(self, tokens):
        """tokens: list of byte tuples (len <= P). -> (B, d) float32"""
        B = len(tokens)
        Z = np.zeros((B, self.P, self.s), np.float32)
        for i, b in enumerate(tokens):
            L = len(b)
            if not 1 <= L <= self.P:
                raise ValueError(f"token length {L} outside [1, {self.P}]")
            Z[i, :L, :] = self.G[list(b)] / np.sqrt(L)
        return (np.einsum("pq,bqs->bps", self.T, Z).reshape(B, -1) @ self.Q).astype(np.float32)

    def decode(self, E, chunk=2000):
        """(B, d) -> list of byte tuples. Exact left inverse of encode."""
        out = []
        for st in range(0, E.shape[0], chunk):
            Ec = E[st:st + chunk]
            B = Ec.shape[0]
            Z = (Ec @ self.Q.T).reshape(B, self.P, self.s)
            Z = np.einsum("pq,bqs->bps", self.Tinv, Z)
            norms = np.linalg.norm(Z, axis=2)
            L = (norms > 0.5 * norms.max(axis=1, keepdims=True)).sum(axis=1)
            best = np.einsum("bps,vs->bpv", Z, self.G).argmax(axis=2)
            out += [tuple(best[i, :L[i]]) for i in range(B)]
        return out


# ----------------------------------------------------------------- self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for d in (768, 2048, 4096):
        m = ReversibleKroneckerEmbedding(d)
        toks = [tuple(rng.integers(0, 256, size=int(l)))
                for l in rng.integers(1, m.P + 1, size=5000)]
        ok = sum(a == b for a, b in zip(toks, m.decode(m.encode(toks))))
        print(f"d={d:>5}  s={m.s:>3}  mu={m.mu:.4f} "
              f"(Welch {welch_bound(s=m.s):.4f})  margin={m.margin:.5f}  "
              f"round-trip {ok}/{len(toks)}  params={m.n_params()/1e6:.4f}M")
