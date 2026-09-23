'''
Global settings for beef.

The settings defined here are used as defaults by other functions in beef, and
apply whenever the corresponding argument is left as None. Set them once at the
top of a script, e.g.:

    import beef
    beef.set_default_plotter('background')
'''

default_plotter = None
'''Plotter class used by default by the plotting functions (None implies pyvista.Plotter).'''


def set_default_plotter(plotter):
    '''
    Set the plotter used by default by the plotting functions in beef.

    Arguments
    ----------
    plotter : {'standard', 'background'}, bool, None or class
        'standard', False or None results in pyvista.Plotter;
        'background' or True results in pyvistaqt.BackgroundPlotter;
        a plotter class may also be given directly

    Notes
    ----------
    Intended to be called once, at the top of a script. The default may still be
    overridden per call, by passing background_plotter to the plotting function.
    '''

    global default_plotter

    if plotter is None or plotter is False:
        plotter = 'standard'
    elif plotter is True:
        plotter = 'background'

    if plotter == 'standard':
        import pyvista as pv
        default_plotter = pv.Plotter
    elif plotter == 'background':
        from pyvistaqt import BackgroundPlotter
        default_plotter = BackgroundPlotter
    elif isinstance(plotter, str):
        raise ValueError(f"Unknown plotter '{plotter}', valid strings are 'standard' and 'background'")
    else:
        default_plotter = plotter


def get_default_plotter():
    '''
    Get the plotter class used by default by the plotting functions in beef.

    Returns
    ----------
    plotter : class
        plotter class, pyvista.Plotter if no default has been set
    '''

    if default_plotter is None:
        import pyvista as pv
        return pv.Plotter

    return default_plotter
