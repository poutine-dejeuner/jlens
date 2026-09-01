"""Bridge to anthropics/jacobian-lens installed in site-packages.

Our local package is also named 'jlens', which shadows the upstream
because our repo root is on sys.path. This module handles the collision.
"""

import importlib
import os
import site
import sys


def get_upstream():
    """Return the upstream anthropics/jacobian-lens package.

    Temporarily removes our local jlens paths from sys.path so that
    'import jlens' resolves to the site-packages installation.
    """
    if 'upstream_jlens' in sys.modules:
        return sys.modules['upstream_jlens']

    # Find upstream in site-packages  
    upstream_path = None
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, 'jlens', '__init__.py')
        if os.path.exists(candidate):
            upstream_path = sp
            break

    if upstream_path is None:
        raise ImportError(
            "anthropics/jacobian-lens not found in site-packages. "
            "Install with: uv add 'jlens @ git+https://github.com/anthropics/jacobian-lens.git'"
        )

    # Remove our local package paths from sys.path so 'jlens' resolves upstream
    saved_local = sys.modules.pop('jlens', None)
    saved_paths = []
    cleanup = []
    for p in sys.path[:]:
        local_init = os.path.join(p, 'jlens', '__init__.py')
        if os.path.exists(local_init):
            real = os.path.realpath(local_init)
            if upstream_path not in real:
                saved_paths.append(p)
                sys.path.remove(p)
                cleanup.append(p)

    try:
        import jlens as upstream
        import jlens.fitting
        import jlens.hf
        import jlens.hooks
        import jlens.protocol
        import jlens.lens
        sys.modules['upstream_jlens'] = upstream
        return upstream
    finally:
        # Restore paths
        for p in saved_paths:
            sys.path.insert(0, p)
        # Restore local jlens
        if saved_local is not None:
            sys.modules['jlens'] = saved_local
        else:
            sys.modules.pop('jlens', None)
