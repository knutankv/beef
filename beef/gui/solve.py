"""Run analyses for the GUI and post-process the results into JSON-able dicts."""
import time

import numpy as np

from . import model as gm
from . import sparse as ss
from .session import UserError

SPARSE_KINDS = ('eig', 'lin_static', 'buckling')
MAX_RESULT_VALUES = 4_000_000     # cap on DOFs x time samples sent to the browser


def build(m, assemble=True):
    """Execute the generated model code; returns (eldef, forces, namespace)."""
    import beef  # noqa: F401
    from beef import fe
    ns = {'np': np, 'fe': fe}
    exec(compile(gm.model_code(m, assemble=assemble), '<beef-gui model>', 'exec'), ns)
    exec(compile(gm.forces_code(m, select=gm.force_selection(m)), '<beef-gui forces>', 'exec'), ns)
    exec(compile(gm.element_equivalent_code(m), '<beef-gui element loads>', 'exec'), ns)
    return ns['eldef'], ns['forces'], ns


def fixed_end(feq_list, select=None, t=None, tmax=1.0):
    """Sum of the element equivalent load vectors active for a load selection at time t:
    {element label: local vector}. Ramps are scaled by t/tmax (like fe.Force with a scalar t); base loads
    (select='base') are at full value."""
    out = {}
    for case, mode, feq in feq_list or []:
        if select is not None and case != select:
            continue
        fac = 1.0
        if mode == 'ramp' and select != 'base' and t is not None:
            fac = min(max(float(t)/float(tmax or 1.0), 0.0), 1.0)
        for lbl, v in feq.items():
            out[lbl] = out.get(lbl, 0.0) + fac*np.asarray(v, dtype=float)
    return out


def run_base(m, eldef, solver, feq_list=None):
    """Static base state. Returns (eldef in base state, u_base, base element results)."""
    from beef import fe
    mode = gm.base_mode(m)
    print(f"Static base state: {'nonlinear' if mode == 'static' else 'linear'} static under the 'static base' loads", flush=True)
    if mode == 'lin_static' and solver == 'sparse':    # same as model.base_code, with sparse assembly
        ns = {'np': np, 'fe': fe, 'eldef': eldef}      # element loads refer to eldef
        exec(compile(gm.forces_code(m, 'base_forces', 'base'), '<beef-gui base forces>', 'exec'), ns)
        u_b, _ = ss.lin_static_sparse(eldef, ss.global_forces(eldef, ns['base_forces'], 1.0))
        is2 = eldef.domain == '2d'
        for el in eldef.elements:
            q = el.get_local_kd() @ el.tmat @ u_b[np.asarray(el.global_dofs, dtype=int)]
            el.N0 = (q[3] - q[0])/2 if is2 else (q[6] - q[0])/2
        eldef.include_linear_kg = True
    else:
        ns = {'np': np, 'fe': fe, 'eldef': eldef}
        exec(compile(gm.base_code(m), '<beef-gui base state>', 'exec'), ns)
        eldef, u_b = ns['eldef'], np.asarray(ns['u_base'], dtype=float)
    if not np.all(np.isfinite(u_b)):
        raise UserError('The static base analysis did not converge (non-finite displacements). '
                        'Increase the number of base load steps or itmax, or reduce the base loads.')
    E_b = element_results(eldef, u_b, nonlinear=(mode == 'static'),    # leaves eldef in the base state
                          fixed=fixed_end(feq_list, 'base'))
    Nmax = max(abs(v) for pair in E_b['values']['N'] for v in pair)
    print(f"Base state: max |u| = {np.max(np.abs(u_b)):.4g}, max |N| = {Nmax:.4g}", flush=True)
    return eldef, u_b, E_b


def _add_results(Ea, Eb):
    """Element results Ea + Eb (same elements/keys)."""
    out = dict(Ea); out['values'] = {k: [[a0 + b0, a1 + b1] for (a0, a1), (b0, b1) in zip(Ea['values'][k], Eb['values'][k])]
                                     for k in Ea['keys']}
    return out


def choose_solver(m, solver='auto'):
    kind = m.get('analysis', {}).get('kind', 'lin_static')
    if solver == 'sparse':
        if kind not in SPARSE_KINDS:
            raise UserError(f'The sparse solver supports modal and linear static analysis, not {kind!r}.')
        return 'sparse'
    if solver == 'beef':
        return 'beef'
    return 'sparse' if (gm.is_large(m) and kind in SPARSE_KINDS) else 'beef'


# ------------------------------------------------------------------ element section forces
def _local_axes(el):
    T = np.asarray(el.tmat)
    if el.domain == '3d':
        return T[0, :3], T[1, :3], T[2, :3]
    e1 = np.r_[T[0, :2], 0.0]
    return e1, np.array([-e1[1], e1[0], 0.0]), np.array([0.0, 0.0, 1.0])


def element_results(eldef, u, nonlinear=False, fixed=None):
    """Section forces at both element ends, and local axes, for displacement vector u (all DOFs).

    3D keys: N, Qy, Qz, Mx (torsion), My, Mz; 2D keys: N, Q, M. Values are [end 1, end 2]
    per element, in beef's sign convention (the element averages equal beef's N, My, ... properties).
    fixed: {element label: local equivalent load vector} of distributed element loads; the end forces are then
    k u - f_eq (beef itself only knows the resulting nodal loads)."""
    # Linear: section forces k_local*T*u_e with the UNDEFORMED transformation (beef's deform_linear moves the
    # nodes and update_linear rebuilds T from the deformed geometry; 3D update_linear also sets no forces).
    # Nonlinear: beef's corotational state after deform().
    eldef.reset_deformation()
    if nonlinear:
        eldef.deform(u)
    u = np.asarray(u, dtype=float)
    is3 = eldef.domain == '3d'
    keys = ['N', 'Qy', 'Qz', 'Mx', 'My', 'Mz'] if is3 else ['N', 'Q', 'M']
    out = {k: [] for k in keys}
    e2s, e3s = [], []
    for el in eldef.elements:
        f = np.asarray(fixed[el.label], dtype=float) if (fixed and el.label in fixed) else None
        if is3:
            if nonlinear:
                q = np.asarray(el.q_loc, dtype=float)
            else:
                q = el.get_local_kd() @ el.tmat @ u[np.asarray(el.global_dofs, dtype=int)]
            if f is not None:
                q = q - f
            for k, i in zip(keys, range(6)):
                out[k].append([float(-q[i]), float(q[i + 6])])
        else:
            if nonlinear:                       # corotational: t = [N, M (mean), -Q L/2]
                t = np.asarray(el.t, dtype=float)
                N, Q, M1, M2 = t[0], -2*t[2]/el.L, t[1] - t[2], t[1] + t[2]
                ends = np.array([N, N, Q, Q, M1, M2])
            else:                               # q = [N1, V1, M1, N2, V2, M2] end forces (local)
                q = el.get_local_kd() @ el.tmat @ u[np.asarray(el.global_dofs, dtype=int)]
                ends = np.array([-q[0], q[3], -q[1], q[4], -q[2], q[5]])
            if f is not None:                   # end values of k u - f_eq
                ends = ends + np.array([f[0], -f[3], f[1], -f[4], f[2], -f[5]])
            out['N'].append([float(ends[0]), float(ends[1])])
            out['Q'].append([float(ends[2]), float(ends[3])])
            out['M'].append([float(ends[4]), float(ends[5])])
        _, e2, e3 = _local_axes(el)
        e2s.append([float(v) for v in e2]); e3s.append([float(v) for v in e3])
    return {'labels': [int(el.label) for el in eldef.elements], 'keys': keys, 'values': out, 'e2': e2s, 'e3': e3s}


def _clean(v):
    v = float(v)
    return v if np.isfinite(v) else None


def _reactions(eldef, R, node_labels, node_gdofs):
    cd = eldef.constrained_dofs
    if cd is None:
        return []
    inv = {g: (node_labels[i], j) for i, gd in enumerate(node_gdofs) for j, g in enumerate(gd)}
    out = []
    for g in cd:
        nl, j = inv.get(int(g), (None, None))
        out.append({'node': nl, 'dof': j, 'value': _clean(R[int(g)])})
    return out


# ------------------------------------------------------------------ main entry
def run(sess, m, solver='auto'):
    """Build and solve the GUI model. Runs on the job thread; prints go to the job log."""
    import beef
    from beef import fe
    a = m.get('analysis', {})
    kind = a.get('kind', 'lin_static')
    solver = choose_solver(m, solver)
    t0 = time.time()
    print(f"beef {beef.__version__}: {kind} analysis, {len(m['nodes'])} nodes, {len(m['elements'])} elements, "
          f"{gm.model_dofs(m)} DOFs, solver: {solver}", flush=True)
    base = gm.base_mode(m)
    # a nonlinear static base needs beef's (dense) run_static, also when the following modal analysis is sparse
    eldef, forces, bns = build(m, assemble=(solver == 'beef' or base == 'static'))
    feq_list = bns.get('element_feq', [])
    sel = gm.force_selection(m)                 # which loads the main analysis uses
    tmax = gm.num(a.get('tmax'), 1.0)
    print(f"Model built in {time.time()-t0:.2f} s", flush=True)
    u_base = E_base = None
    if base != 'none':
        tb = time.time()
        eldef, u_base, E_base = run_base(m, eldef, solver, feq_list)
        print(f"Base state computed in {time.time()-tb:.2f} s", flush=True)
    node_labels = [int(n.label) for n in eldef.nodes]
    node_gdofs = [[int(d) for d in n.global_dofs] for n in eldef.nodes]
    out = {'kind': kind, 'solver': solver, 'domain': eldef.domain, 'nodeLabels': node_labels,
           'node_gdofs': node_gdofs, 'ndofPerNode': int(eldef.nodes[0].ndofs), 'ndofs': int(eldef.ndofs),
           'beef_version': beef.__version__}
    if base != 'none':
        out['base'] = {'mode': base, 'max_u': float(np.max(np.abs(u_base))), 'elements': E_base,
                       'u': [float(v) for v in u_base]}
    analysis = None
    t1 = time.time()

    if solver == 'sparse':
        if kind == 'eig':
            freq, phi, has_damp = ss.eig_sparse(eldef, n_modes=gm.n_modes(m, 20))
            out['frequencies'] = [float(f) for f in freq]
            out['damping'] = [0.0]*len(freq)
            out['phi'] = [[float(v) for v in phi[:, i]] for i in range(phi.shape[1])]
            if has_damp or 'rayleigh' in gm.analysis_kwargs(m):
                out['note'] = 'Damping (dashpots / Rayleigh) is ignored by the sparse solver: undamped modes shown.'
        elif kind == 'buckling':
            lam, phi = buckling_sparse(eldef, forces, fixed_end(feq_list, None, tmax, tmax), gm.n_modes(m, 10), tmax)
            out['factors'] = [float(v) for v in lam]
            out['phi'] = [[float(v) for v in phi[:, i]] for i in range(phi.shape[1])]
        else:
            f = ss.global_forces(eldef, forces, gm.num(a.get('tmax'), 1.0))
            u, R = ss.lin_static_sparse(eldef, f)
            out['t'] = [0.0]
            out['u'] = [[float(v)] for v in u]
            out['reactions'] = _reactions(eldef, R, node_labels, node_gdofs)
            out['elements'] = element_results(eldef, u, fixed=fixed_end(feq_list, sel, tmax, tmax))
    else:
        analysis = fe.Analysis(eldef, forces=forces, **gm.analysis_kwargs(m))
        eldef = analysis.eldef
        if kind == 'eig':
            lambd, phi = analysis.run_eig()
            freq = np.abs(np.imag(lambd))/(2*np.pi)
            order = np.argsort(freq)
            freq, lam, phi = freq[order], np.asarray(lambd)[order], np.real(np.asarray(phi)[:, order])
            with np.errstate(all='ignore'):
                zeta = -np.real(lam)/np.abs(lam)
            nmax = min(gm.n_modes(m, 30), phi.shape[1])
            out['frequencies'] = [float(f) for f in freq[:nmax]]
            out['damping'] = [float(z) if np.isfinite(z) else 0.0 for z in zeta[:nmax]]
            out['phi'] = [[float(v) for v in phi[:, i]] for i in range(nmax)]
        elif kind == 'buckling':
            ns = {'np': np, 'fe': fe, 'eldef': eldef, 'analysis': analysis}
            exec(compile(gm.buckling_code(m), '<beef-gui buckling>', 'exec'), ns)
            lam, phi = ns['lambd_b'], ns['phi_b']
            nmax = min(gm.n_modes(m, 10), phi.shape[1])
            out['factors'] = [float(v) for v in lam[:nmax]]
            out['phi'] = [[float(v) for v in phi[:, i]] for i in range(nmax)]
        else:
            getattr(analysis, {'lin_static': 'run_lin_static', 'static': 'run_static',
                               'lin_dynamic': 'run_lin_dynamic', 'dynamic': 'run_dynamic'}[kind])()
            u = np.asarray(analysis.u, dtype=float)
            t = np.asarray(analysis.t, dtype=float)
            if u_base is not None:          # response is incremental about the base state: report totals
                u_inc = u; u = u + u_base[:, None]
            stride = max(1, int(np.ceil(u.size/MAX_RESULT_VALUES)))
            idx = np.unique(np.r_[np.arange(0, len(t), stride), len(t) - 1])
            if stride > 1:
                out['note'] = f'Time history thinned to every {stride}. sample for display ({len(idx)} of {len(t)}).'
            out['t'] = [float(v) for v in t[idx]]
            out['u'] = np.where(np.isfinite(u[:, idx]), u[:, idx], 0.0).tolist()
            uf = u[:, -1]
            nonlinear = kind in ('static', 'dynamic')
            try:
                fvec = analysis.get_global_forces(t[-1])
                if nonlinear:
                    eldef.deform(uf)
                    R = eldef.q - fvec
                else:
                    R = eldef.k @ uf - fvec
                out['reactions'] = _reactions(eldef, R, node_labels, node_gdofs)
            except Exception as exc:
                out['reactions'] = []
                print(f'Reactions not available: {exc}')
            if u_base is not None:
                out['elements'] = _add_results(E_base, element_results(eldef, u_inc[:, -1],
                                                                       fixed=fixed_end(feq_list, sel, t[-1], tmax)))
                out['note'] = ((out.get('note') or '') + ' Linear dynamics about the static base state: displacements and '
                               'section forces are base + increment (increment from the base-state tangent stiffness).').strip()
            else:
                out['elements'] = element_results(eldef, uf, nonlinear=nonlinear,
                                                  fixed=fixed_end(feq_list, sel, t[-1], tmax))
    print(f"Solved in {time.time()-t1:.2f} s", flush=True)
    if 'frequencies' in out:
        print('Natural frequencies [Hz]: ' + ', '.join(f'{f:.4g}' for f in out['frequencies'][:12])
              + (' ...' if len(out['frequencies']) > 12 else ''))
    if 'factors' in out:
        print('Critical load factors: ' + ', '.join(f'{f:.4g}' for f in out['factors'][:8]))
    if 'u' in out:
        print(f"Max |u| = {np.max(np.abs(np.asarray(out['u']))):.4g}")
    sess.ns.update(eldef=eldef, analysis=analysis, model_dict=m, result=out, u_base=u_base, base_elements=E_base,
                   element_feq=feq_list)
    return out


def element_results_at(sess, time_index):
    """Section forces at another time step of the last time-domain/static analysis."""
    an, res, m = sess.ns.get('analysis'), sess.ns.get('result'), sess.ns.get('model_dict') or {}
    if an is None or res is None or res.get('kind') not in ('static', 'dynamic', 'lin_dynamic', 'lin_static'):
        raise UserError('No time-domain result available.')
    k = int(time_index)
    u = np.asarray(res['u'], dtype=float)[:, k]
    tk = res['t'][k] if res.get('t') else None
    fixed = fixed_end(sess.ns.get('element_feq'), gm.force_selection(m), tk, gm.num(m.get('analysis', {}).get('tmax'), 1.0))
    u_base, E_base = sess.ns.get('u_base'), sess.ns.get('base_elements')
    if u_base is not None and E_base is not None:
        return _add_results(E_base, element_results(an.eldef, u - u_base, fixed=fixed))
    return element_results(an.eldef, u, nonlinear=res['kind'] in ('static', 'dynamic'), fixed=fixed)


def refine(m, n, element_labels=None):
    """Refine the GUI model's mesh with beef's ElDef.refine_mesh. Returns new nodes/elements (GUI format) and the
    map {old element label: [new element labels]}. Node labels of existing nodes are kept (beef convention)."""
    n = int(n)
    if n < 2:
        raise UserError('The refinement factor must be at least 2.')
    import beef  # noqa: F401
    from beef import fe
    ns = {'np': np, 'fe': fe}
    exec(compile(gm.model_code(m, assemble=False), '<beef-gui model>', 'exec'), ns)
    eldef = ns['eldef']
    labels = None if element_labels is None else [int(v) for v in element_labels]
    if labels is not None and not labels:
        raise UserError('No elements selected for refinement.')
    print(f"refine_mesh(n={n}) on {'all' if labels is None else len(labels)} element(s) of {len(eldef.elements)}", flush=True)
    emap = eldef.refine_mesh(n, element_labels=labels, assemble=False)
    old = {int(e['label']): e for e in m['elements']}
    new_els = {int(el.label): el for el in eldef.elements}
    is2 = eldef.domain == '2d'
    elements = []
    for old_label, new_labels in emap.items():
        src = old[int(old_label)]
        for nl in new_labels:
            el = new_els[int(nl)]
            elements.append({'label': int(nl), 'n1': int(el.nodes[0].label), 'n2': int(el.nodes[1].label),
                             'section': src.get('section'), 'etype': src.get('etype') or 'beam', 'part': src.get('part') or '1'})
    nodes = []
    for nd in sorted(eldef.nodes, key=lambda x: int(x.label)):
        c = np.asarray(nd.coordinates, dtype=float)
        d = {'label': int(nd.label), 'x': float(c[0]), 'y': float(c[1])}
        d['z'] = 0.0 if is2 else float(c[2])
        nodes.append(d)
    print(f"{len(m['elements'])} -> {len(elements)} elements, {len(m['nodes'])} -> {len(nodes)} nodes", flush=True)
    return {'nodes': nodes, 'elements': elements, 'map': {str(k): [int(v) for v in vs] for k, vs in emap.items()}}


def buckling_sparse(eldef, forces, fixed, n_modes=10, t=1.0):
    """Linear buckling with sparse matrices: linear static state of the reference loads -> element axial forces
    (k T u - f_eq) -> geometric stiffness K_g(N0) -> smallest positive lambda of (K + lambda K_g) phi = 0,
    solved as the symmetric problem (-K_g) phi = (1/lambda) K phi (K positive definite)."""
    import scipy.sparse as sp
    from scipy.linalg import eigh
    from scipy.sparse.linalg import eigsh, splu, LinearOperator
    K, C, M = ss.assemble_sparse(eldef)
    L = ss.constraint_map(eldef)
    Kr = (L.T @ K @ L).tocsc()
    lu = splu(Kr)
    u = L @ lu.solve(L.T @ ss.global_forces(eldef, forces, t))
    E = element_results(eldef, u, fixed=fixed)
    for el, Nv in zip(eldef.elements, E['values']['N']):
        el.N0 = 0.5*(Nv[0] + Nv[1])
    n = int(eldef.ndofs)
    rows, cols, vals = [], [], []
    for el in eldef.elements:
        gd = np.asarray(el.global_dofs, dtype=int)
        r, c = np.meshgrid(gd, gd, indexing='ij')
        rows.append(r.ravel()); cols.append(c.ravel()); vals.append(np.asarray(el.get_kg_axial(), dtype=float).ravel())
    Kg = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    Kgr = (L.T @ Kg @ L).tocsc()
    Kgr = 0.5*(Kgr + Kgr.T)
    nr = Kr.shape[0]
    if nr <= 1500:
        mu, q = eigh(-Kgr.toarray(), 0.5*(Kr + Kr.T).toarray())
    else:
        k = int(max(1, min(n_modes, nr - 2)))
        Minv = LinearOperator(Kr.shape, matvec=lu.solve, dtype=float)
        mu, q = eigsh(-Kgr, k=k, M=0.5*(Kr + Kr.T), Minv=Minv, which='LA')
    keep = mu > 1e-12*max(np.max(np.abs(mu)), 1e-300)
    lam = 1.0/mu[keep]; q = q[:, keep]
    order = np.argsort(lam)[:n_modes]
    lam, q = lam[order], q[:, order]
    if lam.size == 0:
        raise UserError('No buckling (no positive load factor): the reference loads cause no compression.')
    phi = L @ q
    phi = phi/np.max(np.abs(phi), axis=0)
    return lam, phi
