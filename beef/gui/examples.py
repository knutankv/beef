"""Example models: the ``*.beef`` model files in the ``examples`` folder next to this module.

A ``.beef`` file is the GUI model as JSON (the same format as *Save* in the GUI writes), with optional
``name``, ``description`` and ``notes`` fields. Drop more files into the folder (or point the
``BEEF_GUI_EXAMPLES`` environment variable to another folder) to make them show up as examples.
"""
import json
import os
from pathlib import Path

from .session import UserError

FORMAT = 'beef-gui-model'


def example_dirs():
    dirs = [Path(__file__).parent / 'examples']
    extra = os.environ.get('BEEF_GUI_EXAMPLES')
    if extra:
        dirs += [Path(p) for p in extra.split(os.pathsep) if p]
    return [d for d in dirs if d.is_dir()]


def example_files():
    return [f for d in example_dirs() for f in sorted(d.glob('*.beef'))]


def read_model(path):
    path = Path(path)
    try:
        m = json.loads(path.read_text(encoding='utf8'))
    except Exception as exc:
        raise UserError(f'Could not read {path.name}: {exc}')
    if not isinstance(m, dict) or 'nodes' not in m:
        raise UserError(f'{path.name} is not a beef GUI model file')
    return m


def list_examples():
    out = []
    for i, f in enumerate(example_files()):
        try:
            m = read_model(f)
            out.append({'index': i, 'file': f.name, 'name': m.get('name') or f.stem,
                        'desc': m.get('description', ''), 'nodes': len(m.get('nodes', [])),
                        'elements': len(m.get('elements', []))})
        except UserError as exc:
            out.append({'index': i, 'file': f.name, 'name': f.stem, 'desc': str(exc), 'error': True})
    return out


def load_example(index):
    files = example_files()
    if not 0 <= int(index) < len(files):
        raise UserError('Unknown example')
    return read_model(files[int(index)])
