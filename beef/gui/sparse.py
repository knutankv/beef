"""Sparse assembly and solvers for large beef models (BEEF GUI).

beef assembles dense global matrices and imposes constraints through a dense
null-space basis. That is fine for small and medium models, but
becomes slow and memory hungry once a model has many thousand DOFs (and the
state-space run_eig doubles the problem size). This module reuses beef's own element
and feature matrices, assembles them sparsely and imposes constraints by DOF
elimination, giving the same results (to round-off) at a fraction of the cost.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu


def assemble_sparse(eldef):
    """Global sparse K, C, M (csr) from beef elements + features (unconstrained)."""
    n = int(eldef.ndofs)
    rows, cols, kv, mv, bad = [], [], [], [], []
    for el in eldef.elements:
        gd = np.asarray(el.global_dofs, dtype=int)
        ke = np.asarray(el.get_k(), dtype=float); me = np.asarray(el.get_m(), dtype=float)
        if getattr(eldef, 'include_linear_kg', False):     # linearized geometric stiffness from el.N0 (as beef)
            ke = ke + np.asarray(el.get_kg_axial(), dtype=float)
        if not (np.all(np.isfinite(ke)) and np.all(np.isfinite(me))):
            bad.append(el.label)
        r, c = np.meshgrid(gd, gd, indexing='ij')
        rows.append(r.ravel()); cols.append(c.ravel()); kv.append(ke.ravel()); mv.append(me.ravel())
    if bad:
        raise ValueError(f'Non-finite element matrices for elements {bad[:20]}'
                         + (' ...' if len(bad) > 20 else '') + ' - check that section e2 is not parallel to these elements.')
    R = np.concatenate(rows); Cc = np.concatenate(cols)
    K = sp.coo_matrix((np.concatenate(kv), (R, Cc)), shape=(n, n)).tocsr()
    M = sp.coo_matrix((np.concatenate(mv), (R, Cc)), shape=(n, n)).tocsr()
    C = sp.csr_matrix((n, n))
    for f in eldef.features:
        gd = np.hstack([eldef.get_global_dofs([nl])[np.asarray(f.dofs, dtype=int)] for nl in f.node_labels]).astype(int)
        r, c = np.meshgrid(gd, gd, indexing='ij')
        add = sp.coo_matrix((np.asarray(f.global_matrix, dtype=float).ravel(), (r.ravel(), c.ravel())), shape=(n, n)).tocsr()
        if f.type == 'k': K = K + add
        elif f.type == 'm': M = M + add
        elif f.type == 'c': C = C + add
    return K, C, M


def constraint_map(eldef):
    """Sparse L (n x n_free) with u = L q satisfying all beef constraints.

    Node-to-ground constraints remove DOFs; node-to-node constraints make DOFs
    equal (the same kinematics as beef's compatibility matrix)."""
    n = int(eldef.ndofs)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    fixed = np.zeros(n, bool)
    pairs = eldef.dof_pairs
    if pairs is not None:
        for a, b in pairs:
            if b is None:
                fixed[int(a)] = True
            else:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb: parent[rb] = ra
    roots = np.array([find(i) for i in range(n)], dtype=int)
    fixed_root = np.zeros(n, bool); fixed_root[roots[fixed]] = True
    free_roots = np.unique(roots[~fixed_root[roots]])
    col = -np.ones(n, int); col[free_roots] = np.arange(len(free_roots))
    keep = ~fixed_root[roots]
    rows = np.arange(n)[keep]
    return sp.csr_matrix((np.ones(rows.size), (rows, col[roots[keep]])), shape=(n, len(free_roots)))


def _subspace_iteration(K, M, k, sigma, tol=1e-10, maxit=300):
    """Fallback eigen-solver (shift-invert subspace iteration) if ARPACK is unavailable."""
    lu = splu((K - sigma*M).tocsc())
    n = K.shape[0]; p = min(n, max(2*k, k + 8))
    X = np.random.default_rng(0).standard_normal((n, p))
    lam_old = None
    for _ in range(maxit):
        Y = lu.solve(M @ X)
        Kr = Y.T @ (K @ Y); Mr = Y.T @ (M @ Y)
        Kr = 0.5*(Kr + Kr.T); Mr = 0.5*(Mr + Mr.T)
        from scipy.linalg import eigh
        lam, Q = eigh(Kr, Mr)
        X = Y @ Q
        if lam_old is not None and np.max(np.abs(lam[:k] - lam_old[:k])/np.maximum(np.abs(lam[:k]), 1e-30)) < tol:
            break
        lam_old = lam
    return lam[:k], X[:, :k]


def eig_sparse(eldef, n_modes=20, f_shift=1e-3):
    """Undamped natural modes (shift-invert Lanczos). Returns (freq [Hz], phi [n x n_modes], has_damping)."""
    K, C, M = assemble_sparse(eldef)
    L = constraint_map(eldef)
    Kr = (L.T @ K @ L).tocsc(); Mr = (L.T @ M @ L).tocsc()
    Kr = 0.5*(Kr + Kr.T); Mr = 0.5*(Mr + Mr.T)
    k = int(max(1, min(n_modes, Kr.shape[0] - 2)))
    sigma = -(2*np.pi*f_shift)**2           # just below zero -> K - sigma*M is positive definite
    try:
        from scipy.sparse.linalg import eigsh
        lam, q = eigsh(Kr, k=k, M=Mr, sigma=sigma, which='LM')
    except Exception:
        lam, q = _subspace_iteration(Kr, Mr, k, sigma)
    order = np.argsort(lam); lam = lam[order]; q = q[:, order]
    freq = np.sqrt(np.abs(lam))/(2*np.pi)
    phi = L @ q
    phi = phi/np.max(np.abs(phi), axis=0)    # same normalisation as beef (max |value| = 1)
    return freq, phi, C.nnz > 0


def global_forces(eldef, forces, t):
    """Global load vector at time t (as beef's Analysis.get_global_forces)."""
    f = np.zeros(int(eldef.ndofs))
    for force in forces:
        ix = np.hstack([eldef.node_dof_lookup(nl, dof_ix=dix) for nl, dix in zip(force.node_labels, force.dof_ix)]).flatten()
        f[ix] += force.evaluate(t)
    return f


def lin_static_sparse(eldef, f):
    """Linear static displacements u (full DOF vector) and residual K u - f (reactions at constrained DOFs)."""
    K, C, M = assemble_sparse(eldef)
    L = constraint_map(eldef)
    q = spsolve((L.T @ K @ L).tocsc(), L.T @ f)
    u = L @ q
    return u, K @ u - f
