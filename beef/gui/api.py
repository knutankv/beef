"""JSON API endpoints of the BEEF GUI."""
import ast
import platform
from pathlib import Path

from . import __version__
from . import examples as ex
from . import model as gm
from . import solve
from .server import api
from .session import UserError


@api('info')
def info(sess, p):
    import beef, numpy, scipy
    return {'gui_version': __version__, 'beef_version': beef.__version__, 'beef_path': str(Path(beef.__file__).parent),
            'numpy': numpy.__version__, 'scipy': scipy.__version__, 'python': platform.python_version(),
            'large_dofs': gm.LARGE_DOFS, 'sparse_kinds': list(solve.SPARSE_KINDS)}


@api('examples')
def examples(sess, p):
    return {'dirs': [str(d) for d in ex.example_dirs()], 'examples': ex.list_examples()}


@api('example')
def example(sess, p):
    return ex.load_example(p.get('index', 0))


@api('run')
def run(sess, p):
    m = p.get('model')
    if not m:
        raise UserError('No model sent.')
    solver = p.get('solver', 'auto')
    solve.choose_solver(m, solver)       # validate before queueing
    label = f"{m.get('analysis', {}).get('kind', 'analysis')} ({len(m.get('nodes', []))} nodes)"
    return sess.run('analysis', label, lambda: solve.run(sess, m, solver), wait=2.0)


@api('element_results')
def element_results(sess, p):
    return solve.element_results_at(sess, p.get('time_index', -1))


@api('script')
def script(sess, p):
    m = p.get('model')
    if not m:
        raise UserError('No model sent.')
    return {'script': gm.export_script(m)}


@api('console')
def console(sess, p):
    """Run Python in the session namespace (eldef, analysis, result, np, fe, beef ...)."""
    code = str(p.get('code') or '').strip()
    if not code:
        return {'job': None, 'done': True, 'result': None, 'log': ''}

    def job():
        print('>>> ' + code.replace('\n', '\n... '))
        tree = ast.parse(code, '<console>', 'exec')
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, '<console>', 'exec'), sess.ns)
        if last is not None:            # like a REPL: show the value of a trailing expression
            val = eval(compile(ast.Expression(last.value), '<console>', 'eval'), sess.ns)
            if val is not None:
                print(repr(val))
        return None
    return sess.run('console', code.splitlines()[0][:60], job, wait=10.0)


# ------------------------------------------------------------------ jobs
@api('jobs')
def jobs(sess, p):
    return {'busy': sess.jobs.busy, 'jobs': [j.to_dict() for j in reversed(list(sess.jobs.jobs.values()))][:50]}


@api('job')
def job(sess, p):
    j = sess.jobs.jobs.get(int(p.get('id', -1)))
    if j is None:
        raise UserError('No such job')
    d = j.to_dict(with_log=True, with_result=bool(p.get('result')) and j.status == 'done')
    return d


@api('job/cancel')
def job_cancel(sess, p):
    return sess.jobs.cancel(int(p['id'])).to_dict()


@api('jobs/clear')
def jobs_clear(sess, p):
    for k in [k for k, j in sess.jobs.jobs.items() if j.status not in ('queued', 'running')]:
        sess.jobs.jobs.pop(k)
    return True


@api('refine')
def refine(sess, p):
    m = p.get('model')
    if not m:
        raise UserError('No model sent.')
    n, labels = p.get('n', 2), p.get('element_labels')
    return sess.run('refine', f'refine mesh x{n}', lambda: solve.refine(m, n, labels), wait=10.0)
