"""Conversion of Python / NumPy objects into JSON-safe structures."""
import inspect
import math
import numbers

import numpy as np


def clean_float(x):
    """Return a JSON-safe float (NaN/inf -> None)."""
    try:
        x = float(x)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def arr(a, decimals=None):
    """Convert an array-like to nested lists. Complex arrays become {re, im}."""
    if a is None:
        return None
    a = np.asarray(a)
    if np.iscomplexobj(a):
        return {"re": arr(np.real(a), decimals), "im": arr(np.imag(a), decimals), "complex": True}
    if a.dtype.kind in "biu":
        return a.tolist()
    if a.dtype.kind == "b":
        return a.tolist()
    if a.dtype.kind == "f":
        a = a.astype(float)
        if decimals is not None:
            a = np.round(a, decimals)
        finite = np.isfinite(a)
        if finite.all():
            return a.tolist()
        obj = a.astype(object)
        obj[~finite] = None
        return obj.tolist()
    return [jsonable(v) for v in a.tolist()] if a.ndim else jsonable(a.item())


def jsonable(obj, depth=0):
    """Recursively convert obj into JSON-compatible data."""
    if depth > 8:
        return short_repr(obj)
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return clean_float(obj)
    if isinstance(obj, numbers.Complex):
        return {"re": clean_float(obj.real), "im": clean_float(obj.imag), "complex": True}
    if isinstance(obj, np.ndarray):
        return arr(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v, depth + 1) for v in obj]
    return short_repr(obj)


def short_repr(obj, n=300):
    try:
        r = repr(obj)
    except Exception as exc:  # pragma: no cover
        r = f"<unrepresentable {type(obj).__name__}: {exc}>"
    return r if len(r) <= n else r[:n] + " …"


def type_name(obj):
    t = type(obj)
    mod = t.__module__
    return t.__name__ if mod in ("builtins",) else f"{mod}.{t.__name__}"


def summarize(obj, preview=True):
    """Short, structured description of any object (used by explorer/variables)."""
    info = {"type": type_name(obj)}
    if isinstance(obj, np.ndarray):
        info.update(kind="array", shape=list(obj.shape), dtype=str(obj.dtype))
        if preview:
            if obj.size <= 400:
                info["value"] = arr(obj)
            else:
                flat = obj.ravel()
                info["head"] = arr(flat[:12])
            if obj.size and obj.dtype.kind in "fiuc":
                with np.errstate(all="ignore"):
                    mag = np.abs(obj)
                    info["stats"] = {"min_abs": clean_float(np.nanmin(mag)), "max_abs": clean_float(np.nanmax(mag))}
        info["repr"] = f"ndarray{tuple(obj.shape)} {obj.dtype}"
        return info
    if isinstance(obj, (bool, numbers.Number, np.generic)):
        info.update(kind="scalar", value=jsonable(obj), repr=short_repr(obj))
        return info
    if isinstance(obj, str):
        info.update(kind="str", value=obj if len(obj) < 2000 else obj[:2000] + "…", repr=short_repr(obj))
        return info
    if isinstance(obj, (list, tuple)):
        info.update(kind="sequence", length=len(obj), repr=short_repr(obj))
        if preview and len(obj) <= 50 and all(isinstance(v, (numbers.Number, str, bool, type(None))) for v in obj):
            info["value"] = jsonable(obj)
        else:
            info["items"] = [short_repr(v, 80) for v in list(obj)[:20]]
        return info
    if isinstance(obj, dict):
        info.update(kind="dict", length=len(obj), keys=[str(k) for k in list(obj)[:100]], repr=short_repr(obj))
        return info
    if inspect.isclass(obj):
        info.update(kind="class", repr=short_repr(obj), signature=safe_signature(obj), doc=first_doc(obj))
        return info
    if callable(obj):
        info.update(kind="callable", repr=short_repr(obj), signature=safe_signature(obj), doc=first_doc(obj))
        return info
    if inspect.ismodule(obj):
        info.update(kind="module", repr=short_repr(obj))
        return info
    s = None
    try:
        s = str(obj)
    except Exception:
        pass
    info.update(kind="object", repr=short_repr(obj), str=(s if s and len(s) < 3000 else None))
    return info


def safe_signature(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None


def first_doc(obj, full=False):
    doc = inspect.getdoc(obj) or ""
    if full:
        return doc
    return doc.strip().split("\n\n")[0][:400] if doc else ""
