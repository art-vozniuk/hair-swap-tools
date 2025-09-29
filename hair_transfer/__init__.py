import sys as _sys
from importlib import import_module as _imp

_pkg = __name__
for _mod in ("ref_encoder", "diffusers", "utils"):
    try:
        _sys.modules.setdefault(_mod, _imp(f"{_pkg}.{_mod}"))
    except Exception:
        pass
del _sys, _imp, _pkg, _mod