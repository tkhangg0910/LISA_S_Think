"""Shims that let transformers 4.31 run on torch >= 2.6 (needed for B200/sm_100).

Installed as `sitecustomize.py` inside env-b200 only, so it is applied
automatically at interpreter startup and never touches the L4 env.

Each shim is guarded: if the upstream API is still there, nothing happens.
Set LISA_NO_TORCH_COMPAT=1 to disable the whole thing.
"""

import os

if os.environ.get("LISA_NO_TORCH_COMPAT") != "1":
    try:
        import torch

        # 1) torch 2.6 flipped torch.load's `weights_only` default to True.
        #    transformers 4.31 and segment_anything call torch.load() bare, and
        #    DeepSpeed checkpoints legitimately contain non-tensor objects.
        if tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 6):
            _orig_load = torch.load

            def _load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _orig_load(*args, **kwargs)

            if getattr(torch.load, "__module__", None) != __name__:
                _load.__module__ = __name__
                torch.load = _load

        # 2) torch 2.2 renamed _register_pytree_node -> register_pytree_node.
        #    transformers 4.31 still calls the underscore name at import time.
        import torch.utils._pytree as _pt

        if not hasattr(_pt, "_register_pytree_node") and hasattr(
            _pt, "register_pytree_node"
        ):
            _pt._register_pytree_node = _pt.register_pytree_node

    except Exception:  # never let a shim break the interpreter
        pass
