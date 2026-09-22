"""Prepare optional PyTorch DLL loading before Qt on Windows.

PyQt5 and the current PyTorch wheel can load incompatible Windows runtime
state when Qt is imported first, causing WinError 1114 for ``c10.dll``.  The
GUI calls :func:`prepare_optional_torch_dlls` before importing pyqtgraph.  This
loads only c10.dll when an installed torch package can be located; it does not
import torch, initialize CUDA, or make torch a CPU-path requirement.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import sys


_DLL_DIRECTORY_HANDLES = []
_DLL_HANDLES = []
_PREPARED = False


def prepare_optional_torch_dlls() -> bool:
    """Prime optional PyTorch's DLL loader before Qt on Windows.

    Return ``True`` when c10.dll is already loaded or was loaded here. Missing
    PyTorch, non-Windows platforms, and preload failures leave CPU startup
    untouched and return ``False``. Explicit CUDA selection still reports the
    full import/device error through the CUDA backend.
    """
    global _PREPARED
    if _PREPARED:
        return True
    if sys.platform != "win32":
        return False
    try:
        spec = importlib.util.find_spec("torch")
        locations = None if spec is None else spec.submodule_search_locations
        if not locations:
            return False
        torch_lib = Path(next(iter(locations))) / "lib"
        c10_dll = torch_lib / "c10.dll"
        if not c10_dll.is_file():
            return False
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(torch_lib)))
        _DLL_HANDLES.append(ctypes.WinDLL(str(c10_dll)))
    except (ImportError, OSError, RuntimeError, StopIteration, ValueError):
        return False
    _PREPARED = True
    return True
