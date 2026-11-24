"""Import beef and build the namespace shared by the analyses and the console."""
import os
import warnings

import numpy as np

COMPAT_SHIMS = []


def install_numpy_compat():
    """Restore NumPy aliases removed in NumPy 2 that older code may still use."""
    for name, target in (("trapz", "trapezoid"), ("Inf", "inf"), ("NaN", "nan"), ("in1d", "isin")):
        try:
            getattr(np, name)
        except AttributeError:
            if hasattr(np, target):
                setattr(np, name, getattr(np, target))
                COMPAT_SHIMS.append(f"np.{name} -> np.{target}")


def build_namespace():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    install_numpy_compat()
    import scipy
    import beef
    from beef import fe
    ns = {"np": np, "numpy": np, "scipy": scipy, "beef": beef, "fe": fe,
          "eldef": None, "analysis": None, "result": None, "model_dict": None,
          "__file_dir__": os.getcwd()}
    return ns
