"""BEEF GUI - a browser-based graphical user interface for beef, served by a local Python process.

Start with ``beef-gui``, ``python -m beef.gui`` or ``beef.gui.launch()``. The page drives a live
Python session that runs the installed beef package, so the GUI always uses the beef version you have.
"""
__version__ = "0.2.0"


def launch(**kwargs):
    """Start the GUI server (blocking). See ``beef.gui.server.serve`` for options."""
    from .server import serve
    return serve(**kwargs)
