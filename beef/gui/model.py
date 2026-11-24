"""GUI model (JSON dict) -> beef Python code.

The GUI keeps its model as a plain dict (nodes, elements, sections, constraints, features, forces,
analysis). This module turns it into ordinary beef code. The server builds the model by executing
exactly this code, so the exported script is always the same model that the GUI solved.
"""
import re

import numpy as np

from .session import UserError

LARGE_DOFS = 1500      # above this, modal / linear static default to the sparse solver


# ------------------------------------------------------------------ parsing helpers
def parse_ints(s):
    """'1,2,5-8' -> [1, 2, 5, 6, 7, 8]"""
    out = []
    if s is None:
        return out
    if isinstance(s, (int, np.integer)):
        return [int(s)]
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'^(-?\d+)\s*-\s*(-?\d+)$', part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            step = 1 if b >= a else -1
            out += list(range(a, b + step, step))
        elif re.match(r'^-?\d+$', part):
            out.append(int(part))
        else:
            raise UserError(f'Cannot read node/DOF list {s!r} (use e.g. 1,2,5-8)')
    return out


def parse_dofs(s):
    t = str(s).strip().lower()
    if t in ('all', 'trans', 'rot'):
        return t
    return parse_ints(s)


def parse_vec3(s):
    if s is None or str(s).strip() == '':
        return None
    try:
        v = [float(x) for x in re.split(r'[\s,;]+', str(s).strip()) if x]
    except ValueError:
        raise UserError(f'Orientation vector e2 must be three numbers, got {s!r}')
    if len(v) != 3 or not any(v):
        raise UserError(f'Orientation vector e2 must be three numbers (not all zero), got {s!r}')
    return v


def num(v, default=0.0):
    if v is None or v == '':
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise UserError(f'Not a number: {v!r}')
    return x


def pynum(v):
    x = num(v)
    if not np.isfinite(x):
        return 'np.inf' if x > 0 else ('-np.inf' if x < 0 else 'np.nan')
    r = repr(float(x))
    return r[:-2] if r.endswith('.0') and abs(x) < 1e15 else r


def ident(s):
    x = re.sub(r'[^A-Za-z0-9_]', '_', str(s))
    if re.match(r'^[0-9]', x):
        x = 's_' + x
    return x or 'sec'


def ndofs_per_node(m):
    return 3 if m.get('domain') == '2d' else 6


def model_dofs(m):
    return len(m.get('nodes', [])) * ndofs_per_node(m)


def is_large(m):
    return model_dofs(m) > LARGE_DOFS


def _sanitize_expr(expr):
    s = str(expr or '0').replace('^', '**')
    if re.search(r'__|import|lambda|exec|eval|open', s):
        raise UserError(f'Load expression not allowed: {expr!r}')
    return re.sub(r'\b(sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh|exp|sqrt|log|log10|abs|pi|heaviside|sign)\b',
                  r'np.\1', s)


# ------------------------------------------------------------------ code generation
def _sec_vars(m):
    out, used = {}, set()
    for s in m.get('sections', []):
        v = 'sec_' + ident(s['name'])
        while v in used:
            v += '_'
        used.add(v)
        out[s['name']] = v
    return out


def sections_code(m):
    is2 = m.get('domain') == '2d'
    L = ['# --- Sections ---']
    for s, var in zip(m.get('sections', []), _sec_vars(m).values()):
        p = [f"E={pynum(s.get('E'))}", f"rho={pynum(s.get('rho'))}", f"A={pynum(s.get('A'))}", f"I_y={pynum(s.get('I_y'))}"]
        if not is2:
            p.append('I_z=None' if s.get('I_z') in ('', None) else f"I_z={pynum(s.get('I_z'))}")
            p.append(f"J={pynum(s.get('J'))}")
        p.append(f"poisson={pynum(s.get('poisson', 0.3))}")
        if s.get('m') not in ('', None):
            p.append(f"m={pynum(s['m'])}")
        if s.get('G') not in ('', None):
            p.append(f"G={pynum(s['G'])}")
        p.append(f"kappa={pynum(s.get('kappa', 1))}")
        e2 = None if is2 else parse_vec3(s.get('e2'))
        if e2:
            p.append(f"e2=np.array([{', '.join(pynum(v) for v in e2)}])")
        p.append(f"shear_deformation={bool(s.get('shear'))}")
        p.append(f"mass_formulation={str(s.get('massform') or 'euler')!r}")
        p.append(f"name={str(s['name'])!r}")
        L.append(f"{var} = fe.Section({', '.join(p)})")
    if not m.get('sections'):
        L.append("sec_default = fe.Section(name='default')   # no section defined in the GUI")
    return '\n'.join(L)


def _node_matrix(nodes, name, is2):
    rows = [f"[{int(n['label'])}, {pynum(n['x'])}, {pynum(n['y'])}]" if is2 else
            f"[{int(n['label'])}, {pynum(n['x'])}, {pynum(n['y'])}, {pynum(n.get('z', 0))}]" for n in nodes]
    return f"{name} = np.array([\n    " + ',\n    '.join(rows) + "\n], dtype=float)"


def _el_matrix(els, name):
    return f"{name} = np.array([" + ', '.join(f"[{int(e['label'])}, {int(e['n1'])}, {int(e['n2'])}]" for e in els) + "], dtype=int)"


def _compress_list(name, values):
    """sections/element types as a compact expression (runs of identical values)."""
    runs = []
    for v in values:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    if len(runs) == 1 and len(values) > 3:
        return f"{name} = [{runs[0][0]}]*{runs[0][1]}"
    if len(values) > 40:
        return f"{name} = " + ' + '.join(f"[{v}]*{n}" for v, n in runs)
    return f"{name} = [{', '.join(values)}]"


def constraints_code(m, var='constraints'):
    cons = m.get('constraints', [])
    if not cons:
        return f'{var} = []'
    nt = 'beam2d' if m.get('domain') == '2d' else 'beam3d'
    items = []
    for c in cons:
        masters = parse_ints(c.get('masters'))
        if not masters:
            continue
        dofs = parse_dofs(c.get('dofs', 'all'))
        dstr = repr(dofs)
        slaves = parse_ints(c.get('slaves')) if str(c.get('slaves') or '').strip() else None
        if slaves is not None and len(slaves) != len(masters):
            raise UserError(f"Constraint {c.get('name')!r}: {len(masters)} master nodes but {len(slaves)} slave nodes")
        sl = f', slave_nodes={slaves}' if slaves else ''
        items.append(f"fe.Constraint({masters}{sl}, dofs={dstr}, node_type={nt!r}, name={str(c.get('name') or '')!r})")
    return f'{var} = [\n    ' + ',\n    '.join(items) + '\n]'


def features_code(m, var='features'):
    feats = m.get('features', [])
    if not feats:
        return f'{var} = []'
    items = []
    for f in feats:
        nodes = parse_ints(f.get('nodes'))
        dofs = parse_ints(f.get('dofs')) or [0]
        if not nodes:
            continue
        if f.get('ftype') == 'Matrix':
            mat = np.array(f.get('matrix') or [], dtype=float)
            n = len(nodes)*len(dofs)
            if mat.shape != (n, n):
                raise UserError(f"Matrix feature on node(s) {nodes}: matrix must be {n}x{n}, got {mat.shape}")
            rows = ', '.join('[' + ', '.join(pynum(v) for v in r) + ']' for r in mat)
            nm = f", name={str(f['name'])!r}" if f.get('name') else ''
            items.append(f"fe.CustomMatrix({str(f.get('mtype') or 'k')!r}, {nodes}, np.array([{rows}]), dofs={dofs}{nm})")
            continue
        cls = {'Spring': 'Spring', 'Dashpot': 'Dashpot', 'PointMass': 'PointMass'}.get(f.get('ftype'), 'Spring')
        for d in dofs:   # beef features couple ONE dof per object
            items.append(f"fe.{cls}({nodes}, [{d}], {pynum(f.get('value'))})")
    return f'{var} = [\n    ' + ',\n    '.join(items) + '\n]'


BASE_KINDS = ('eig', 'lin_dynamic', 'dynamic')       # analyses that can start from a static base state


def base_mode(m):
    """'none', 'lin_static' or 'static' (only for analyses that support a base state)."""
    a = m.get('analysis', {})
    b = a.get('base') or 'none'
    return b if (a.get('kind') in BASE_KINDS and b in ('lin_static', 'static')) else 'none'


def load_case(f):
    return 'base' if f.get('case') == 'base' else 'analysis'


def force_selection(m):
    """Which loads the main analysis uses: static analyses use all loads; analyses that follow a base state
    use the 'analysis' loads (the 'base' loads define the preload)."""
    kind = m.get('analysis', {}).get('kind', 'lin_static')
    return 'analysis' if kind in BASE_KINDS else None


def _element_set_expr(e):
    """Python expression for the elements an element load acts on (uses `eldef`)."""
    how = e.get('target') or 'all'
    if how == 'list':
        labels = parse_ints(e.get('elements'))
        if not labels:
            raise UserError(f"Element load {e.get('name')!r}: no elements listed")
        return f'eldef.get_elements({labels})'
    if how == 'section':
        sec = str(e.get('section') or '')
        if not sec:
            raise UserError(f"Element load {e.get('name')!r}: choose a section")
        return f'[el for el in eldef.elements if el.section.name == {sec!r}]'
    return 'eldef.elements'


def _element_load_args(m, e, i):
    """(elements expression, load keyword arguments, name) of a GUI element load."""
    nm = str(e.get('name') or f'element load {i+1}')
    kw = []
    opt = e.get('option') or 'gravity'
    if opt == 'mass_per_length':
        kw.append(f"mass_per_length={pynum(e.get('value'))}")
    elif opt == 'total_mass':
        kw.append(f"total_mass={pynum(e.get('value'))}")
    elif opt == 'fun':
        expr = str(e.get('fun') or '').strip()
        if not expr or re.search(r'__|import|exec|eval|open\(', expr):
            raise UserError(f"Element load {nm!r}: invalid expression {expr!r}")
        kw.append(f'fun=lambda sec: ({expr})')
    if opt != 'fun' and e.get('g') not in ('', None) and num(e.get('g')) != 9.81:
        kw.append(f"g={pynum(e.get('g'))}")
    d = None
    txt = str(e.get('direction') or '').strip()
    if txt:
        v = [float(x) for x in re.split(r'[\s,;]+', txt) if x]
        n = 2 if m.get('domain') == '2d' else 3
        if len(v) >= n and any(v[:n]):
            d = v[:n]
    if d:
        kw.append(f"direction={d}")
    if e.get('include_moments') is False:
        kw.append('include_moments=False')
    return _element_set_expr(e), kw, nm


def element_loads_code(m, var='forces', select=None):
    """Lines adding equivalent nodal loads of constant element loads (fe.create_nodal_load_from_element_load)."""
    a = m.get('analysis', {})
    L = []
    for i, e in enumerate(m.get('element_loads', [])):
        if select is not None and load_case(e) != select:
            continue
        els, kw, nm = _element_load_args(m, e, i)
        kw = list(kw)
        if select == 'base':
            kw.append('t=1.0')
        elif e.get('mode') == 'ramp':
            kw.append(f"t={pynum(a.get('tmax', 1))}")
        kw.append(f'name={nm!r}')
        L.append(f"{var} += fe.create_nodal_load_from_element_load({els}, {', '.join(kw)})")
    return '\n'.join(L)


def element_equivalent_code(m, var='element_feq'):
    """Code collecting the local equivalent (fixed-end) load vectors of all element loads, used to correct the
    section forces: var = [(case, mode, {element label: f_eq}), ...]."""
    L = [f'{var} = []']
    for i, e in enumerate(m.get('element_loads', [])):
        els, kw, nm = _element_load_args(m, e, i)
        L.append(f"{var}.append(({load_case(e)!r}, {str(e.get('mode') or 'constant')!r}, "
                 f"fe.get_element_equivalent_loads({els}{', ' if kw else ''}{', '.join(kw)})))")
    return '\n'.join(L)


def forces_code(m, var='forces', select=None):
    """Code for a list of fe.Force (nodal loads, then equivalent nodal loads of element loads).
    select: None = all loads, 'analysis' or 'base' (base loads become ramps reaching their amplitude at
    pseudo-time 1, for the incremental static base analysis)."""
    el = element_loads_code(m, var, select)
    code = _nodal_forces_code(m, var, select)
    return code + ('\n' + el if el else '')


def _nodal_forces_code(m, var='forces', select=None):
    a = m.get('analysis', {})
    forces = [f for f in m.get('forces', []) if select is None or load_case(f) == select]
    if not forces:
        return f'{var} = []'
    pre, items = [], []
    for i, f in enumerate(forces):
        nodes = parse_ints(f.get('nodes'))
        if not nodes:
            continue
        dof, nm = int(f.get('dof', 0)), str(f.get('name') or f'load{i+1}')
        mode = f.get('mode', 'constant')
        if select == 'base':
            if mode == 'history':
                raise UserError(f"Load {nm!r}: a time-history load cannot be a static base load")
            items.append(f"fe.Force({nodes}, {dof}, {pynum(f.get('amp'))}, t=1.0, name={nm!r})")
            continue
        if mode == 'constant':
            items.append(f"fe.Force({nodes}, {dof}, {pynum(f.get('amp'))}, name={nm!r})")
        elif mode == 'ramp':
            items.append(f"fe.Force({nodes}, {dof}, {pynum(f.get('amp'))}, t={pynum(a.get('tmax', 1))}, name={nm!r})")
        else:
            pre.append(f"t_force = np.arange({pynum(a.get('t0', 0))}, {pynum(a.get('tmax', 1))}+{pynum(a.get('dt', 1))}/2, {pynum(a.get('dt', 1))})")
            pre.append(f"amp_{i} = ({_sanitize_expr(f.get('texpr'))}) * np.ones_like(t_force)")
            items.append(f"fe.Force({nodes}, {dof}, amp_{i}, t=t_force, name={nm!r})")
    head = ''
    if pre:
        seen = []
        for p in pre:
            if p not in seen:
                seen.append(p)
        head = '# time axis & load histories\n' + seen[0] + '\nt = t_force\n' + '\n'.join(seen[1:]) + '\n'
    return head + f'{var} = [\n    ' + ',\n    '.join(items) + '\n]'


def model_code(m, assemble=True):
    """Code defining `eldef` (a Part or an Assembly) from the GUI model."""
    if not m.get('nodes') or not m.get('elements'):
        raise UserError('The model needs nodes and elements.')
    is2 = m.get('domain') == '2d'
    secvar = _sec_vars(m)
    default = next(iter(secvar.values()), 'sec_default')
    labels = {int(n['label']) for n in m['nodes']}
    for e in m['elements']:
        for k in ('n1', 'n2'):
            if int(e[k]) not in labels:
                raise UserError(f"Element {e['label']}: node {e[k]} does not exist")
    kw = (', include_linear_kg=True' if m.get('analysis', {}).get('include_kg') else '') + ('' if assemble else ', assemble=False')
    groups = []
    for e in m['elements']:
        g = str(e.get('part') or '1')
        if g not in groups:
            groups.append(g)
    L = [sections_code(m), '']
    if len(groups) <= 1:
        L += ['# --- Part ---', _node_matrix(m['nodes'], 'node_matrix', is2), _el_matrix(m['elements'], 'element_matrix'),
              _compress_list('sections', [secvar.get(e.get('section'), default) for e in m['elements']]),
              _compress_list('element_types', [repr(e.get('etype') or 'beam') for e in m['elements']]), '',
              constraints_code(m), features_code(m), '',
              'part = fe.Part(node_matrix, element_matrix, sections=sections, element_types=element_types, '
              f'constraints=constraints, features=features{kw})',
              'eldef = part']
        return '\n'.join(L)
    L.append('# --- Parts ---')
    pvars = []
    for gi, g in enumerate(groups, 1):
        els = [e for e in m['elements'] if str(e.get('part') or '1') == g]
        used = {int(e[k]) for e in els for k in ('n1', 'n2')}
        nodes = [n for n in m['nodes'] if int(n['label']) in used]
        L += [f'# part {g!r}', _node_matrix(nodes, f'node_matrix_{gi}', is2), _el_matrix(els, f'element_matrix_{gi}'),
              _compress_list(f'sections_{gi}', [secvar.get(e.get('section'), default) for e in els]),
              _compress_list(f'element_types_{gi}', [repr(e.get('etype') or 'beam') for e in els]),
              f"part{gi} = fe.Part(node_matrix_{gi}, element_matrix_{gi}, sections=sections_{gi}, "
              f"element_types=element_types_{gi}{'' if assemble else ', assemble=False'})", '']
        pvars.append(f'part{gi}')
    L += [constraints_code(m), features_code(m), '',
          f"assembly = fe.Assembly([{', '.join(pvars)}], constraints=constraints, features=features{kw})",
          'eldef = assembly']
    return '\n'.join(L)


def analysis_kwargs(m):
    a = m.get('analysis', {})
    kind = a.get('kind', 'lin_static')
    kw = {}
    if kind in ('static', 'dynamic', 'lin_dynamic', 'lin_static'):
        kw.update(t0=num(a.get('t0'), 0.0), dt=num(a.get('dt'), 1.0), tmax=num(a.get('tmax'), 1.0))
    if kind in ('static', 'dynamic'):
        kw['itmax'] = int(num(a.get('itmax'), 10))
        tol = {}
        if a.get('tolU') not in ('', None):
            tol['u'] = num(a['tolU'])
        if a.get('tolR') not in ('', None):
            tol['r'] = num(a['tolR'])
        if tol:
            kw['tol'] = tol
    if kind in ('dynamic', 'lin_dynamic', 'eig'):
        rm, rk = num(a.get('rMass'), 0.0), num(a.get('rStiff'), 0.0)
        if rm or rk:
            kw['rayleigh'] = {'mass': rm, 'stiffness': rk}
    if kind in ('dynamic', 'lin_dynamic'):
        kw['newmark_factors'] = {'beta': num(a.get('beta'), 0.25), 'gamma': num(a.get('gamma'), 0.5), 'alpha': num(a.get('alpha'), 0.0)}
    return kw


def n_modes(m, default=30):
    v = m.get('analysis', {}).get('nModes')
    try:
        return max(1, int(v)) if v not in ('', None) else default
    except (TypeError, ValueError):
        return default


def buckling_code(m):
    """Linear buckling about the linear static state of the reference load (export / beef solver path).

    beef's run_lin_buckling is not usable in all versions, and its 3D elements do not compute element forces in
    linear updates, so the axial forces are evaluated explicitly (k T u - f_eq of element loads), and the
    symmetric problem (-K_g) phi = (1/lambda) K phi is solved (K positive definite)."""
    iN = '(q[3] - q[0])/2' if m.get('domain') == '2d' else '(q[6] - q[0])/2'
    L = ['from scipy.linalg import eigh, solve',
         'L = eldef.L',
         'K = L.T @ eldef.k @ L',
         'u_ref = L @ solve(K, L.T @ analysis.get_global_forces(analysis.t[-1]))   # reference load state']
    if m.get('element_loads'):
        L += [element_equivalent_code(m),
              'feq = {}                            # fixed-end loads of element loads (corrects the axial forces)',
              'for _case, _mode, _d in element_feq:',
              '    for _k, _v in _d.items():',
              '        feq[_k] = feq.get(_k, 0) + _v']
    else:
        L.append('feq = {}')
    L += ['for el in eldef.elements:           # axial force of each element -> N0 of the geometric stiffness',
          '    q = el.get_local_kd() @ el.tmat @ u_ref[el.global_dofs] - feq.get(el.label, 0)',
          f'    el.N0 = {iN}',
          'Kg = L.T @ eldef.get_kg_axial() @ L',
          'mu, phi_r = eigh(-Kg, K)            # (-Kg) phi = (1/lambda) K phi',
          'keep = mu > 1e-12*np.max(np.abs(mu))',
          'lambd_b = 1/mu[keep][::-1]          # critical load factors, ascending',
          'phi_b = L @ phi_r[:, keep][:, ::-1] # buckling modes']
    return '\n'.join(L)


def base_code(m):
    """Code that runs the static base analysis and replaces `eldef` by the base state.

    Nonlinear base: the deformed eldef after run_static (deformed geometry and internal forces, so beef's element
    stiffness is the tangent stiffness including the geometric stiffness). Linear base: undeformed geometry with
    the linearized geometric stiffness K_g(N0) from the axial forces of the linear static solution."""
    mode = base_mode(m)
    if mode == 'none':
        return ''
    a = m.get('analysis', {})
    if a.get('kind') == 'dynamic':
        raise UserError('Nonlinear dynamic analysis cannot start from a static base state: beef\'s run_dynamic always '
                        'starts from the undeformed state at rest. Use linear dynamic (linearized about the base state).')
    is2 = m.get('domain') == '2d'
    L = ['# --- Static base state (preload) for the following analysis ---', forces_code(m, 'base_forces', 'base')]
    if not [f for f in m.get('forces', []) + m.get('element_loads', []) if load_case(f) == 'base']:
        raise UserError("A static base state is selected, but no load has case 'static base' (Loads tab).")
    if mode == 'static':
        n = max(1, int(num(a.get('baseSteps'), 10)))
        kw = [f't0={1/n!r}', f'dt={1/n!r}', 'tmax=1.0', f"itmax={int(num(a.get('itmax'), 10))}"]
        tol = {k: num(a[f]) for k, f in (('u', 'tolU'), ('r', 'tolR')) if a.get(f) not in ('', None)}
        if tol:
            kw.append(f'tol={tol!r}')
        L += [f"base = fe.Analysis(eldef, forces=base_forces, {', '.join(kw)})",
              'base.run_static()                   # nonlinear (corotational) static, base loads ramped in increments',
              'u_base = base.u[:, -1]',
              'eldef = base.eldef                  # deformed state: tangent stiffness incl. geometric stiffness']
    else:
        iN = '(q[3] - q[0])/2' if is2 else '(q[6] - q[0])/2'
        L += ['base = fe.Analysis(eldef, forces=base_forces, t0=1.0, dt=1.0, tmax=1.0)',
              'base.run_lin_static()',
              'u_base = base.u[:, -1]',
              'eldef = base.eldef',
              'eldef.reset_deformation()           # linear base: keep the undeformed geometry',
              'for el in eldef.elements:           # axial forces of the base state -> geometric stiffness K_g(N0)',
              '    q = el.get_local_kd() @ el.tmat @ u_base[el.global_dofs]',
              f'    el.N0 = {iN}',
              'eldef.include_linear_kg = True',
              'eldef.assemble()']
    return '\n'.join(L)


def export_script(m):
    """A complete, readable beef script for the GUI model."""
    a = m.get('analysis', {})
    kind = a.get('kind', 'lin_static')
    big_eig = kind == 'eig' and is_large(m) and base_mode(m) == 'none'
    import beef
    L = ['"""Generated by the BEEF GUI - a model for beef (BEam Elements Framework).',
         f'Generated with beef {beef.__version__}.  Docs: https://knutankv.github.io/beef/"""',
         'import numpy as np', 'from beef import fe']
    if big_eig:
        L.append('from scipy.linalg import eigh')
    L += ['', model_code(m), '']
    if big_eig:
        nm = n_modes(m, 20)
        L += [f'# --- Modal analysis (undamped; {model_dofs(m)} DOFs is large for the state-space run_eig()) ---',
              'L = eldef.L                         # constraint basis (dense null space: may take a minute)',
              'K = L.T @ eldef.k @ L', 'M = L.T @ eldef.m @ L', f'n_modes = {nm}',
              'lambd, q = eigh(K, M, subset_by_index=[0, n_modes-1])',
              'freqs = np.sqrt(np.abs(lambd))/(2*np.pi)   # natural frequencies [Hz]',
              'phi = L @ q                         # mode shapes, all DOFs',
              'for i in range(n_modes):', '    print(f"Mode {i+1}: f = {freqs[i]:.4f} Hz")']
        return '\n'.join(L) + '\n'
    bc = base_code(m)
    if bc:
        L += [bc, '']
    L += [forces_code(m, select=force_selection(m)), '', '# --- Analysis ---']
    args = ', '.join(['eldef', 'forces=forces'] + [f'{k}={v!r}' for k, v in analysis_kwargs(m).items()])
    L.append(f'analysis = fe.Analysis({args})')
    run = {'lin_static': 'analysis.run_lin_static()', 'static': 'analysis.run_static()',
           'eig': 'lambd, phi = analysis.run_eig()', 'lin_dynamic': 'analysis.run_lin_dynamic()',
           'dynamic': 'analysis.run_dynamic()', 'buckling': buckling_code(m) if kind == 'buckling' else ''}[kind]
    L += [run, '', '# --- Postprocess ---']
    if kind == 'eig':
        L += ['freqs = np.abs(np.imag(lambd))/(2*np.pi)   # natural frequencies [Hz]',
              'order = np.argsort(freqs); freqs = freqs[order]; phi = phi[:, order]',
              f'for i in range(min({n_modes(m, 10)}, len(freqs))):', '    print(f"Mode {i+1}: f = {freqs[i]:.4f} Hz")',
              '# analysis.eldef.deform(phi[:, 0]*5.0); analysis.eldef.plot()   # requires pyvista']
    elif kind == 'buckling':
        L += ['print("Critical load factors (lambda):", lambd_b[:10])', '# buckling load = lambda * reference load']
    else:
        keys = "['N', 'Q', 'M']" if m.get('domain') == '2d' else "['N', 'Qy', 'Qz', 'Mx', 'My', 'Mz']"
        nl = 'True' if kind in ('static', 'dynamic') else 'False'
        L += ['u = analysis.u                      # DOFs x time',
              'print("Max |displacement|:", np.nanmax(np.abs(u)))',
              f'results = fe.Results(analysis, element_results={keys})',
              f'results.process(nonlinear={nl})', '# element forces: results.output[key]  (n_elements x time)']
    return '\n'.join(L) + '\n'
