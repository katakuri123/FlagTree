# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""PPU FlagCX runtime discovery and device-library configuration.

This module is imported by the PPU compiler (to attach the Device IR bitcode
as an external library) and by ``triton.experimental.tle.language.communication``
(to discover the host library and headers).  It deliberately performs file
system checks only: importing the FlagCX ctypes wrapper pulls in torch and
must stay out of the compile-only path.
"""

from __future__ import annotations

import os
from pathlib import Path

_BITCODE_NAME = "libflagcx_device.bc"
_SHARED_NAME = "libflagcx.so"
_INCLUDE_NAME = "include"
_ENV_KEYS = ("USE_FLAGCX", "USE_DIST", "USE_DISTRIBUTED", "USE_TLE_DIST", "USE_TLE_DISTRIBUTED")


class FlagcxRuntimeConfig:

    def __init__(self, backend_dir: Path, path_order: int = 0):
        # Keep the installed module path: `triton/backends/ppu` is a symlink to
        # third_party/ppu/backend.  The package root is taken from the imported
        # triton package so a resolved backend path cannot lose the
        # triton/experimental/tle/language sibling.
        import triton as _triton
        self.backend_dir = Path(backend_dir)
        self.triton_path = Path(_triton.__file__).resolve().parent
        self.bitcode_path = None
        self.shared_lib_path = None
        self.include_path = None
        self.is_available = False

        self.enabled = not any(os.environ.get(key, "ON").strip().lower() in ("off", "0", "false")
                               for key in _ENV_KEYS)
        if not self.enabled:
            return

        module_path = os.environ.get("FLAGCX_MODULE_PATH")
        self.module_path = Path(module_path).expanduser() if module_path else None

        bitcode = self._select(
            "FLAGCX_BITCODE_PATH",
            self.module_path / _BITCODE_NAME if self.module_path else None,
            self.backend_dir / "lib" / _BITCODE_NAME,
            os.path.expanduser("~/.flagtree/flagcx") + "/" + _BITCODE_NAME,
        )
        shared = self._select(
            "FLAGCX_LIB_PATH",
            self.module_path / _SHARED_NAME if self.module_path else None,
            self.triton_path / "_C" / _SHARED_NAME,
            os.path.expanduser("~/.flagtree/flagcx") + "/" + _SHARED_NAME,
        )
        include = self._select(
            "FLAGCX_INCLUDE_PATH",
            self.module_path / _INCLUDE_NAME if self.module_path else None,
            self.triton_path / "experimental" / "tle" / "language" / _INCLUDE_NAME,
            os.path.expanduser("~/.flagtree/flagcx") + "/" + _INCLUDE_NAME,
            directory=True,
        )
        self.bitcode_path = bitcode
        self.shared_lib_path = shared
        self.include_path = include
        wrapper = self.triton_path / "experimental" / "tle" / "language" / "flagcx_wrapper.py"
        complete = bool(bitcode and shared and include and wrapper.is_file())
        if self.module_path is not None and not complete:
            raise RuntimeError(f"Incomplete FlagCX bundle at {self.module_path}: expected "
                               f"{_BITCODE_NAME}, {_SHARED_NAME}, {_INCLUDE_NAME}/ and flagcx_wrapper.py")
        self.is_available = complete

        # PPU's symmetric window requires VMM, but VMM signal buffers cannot
        # be IPC-exported on PPU (verified on SDK 2.1.0).  TLE only needs peer
        # pointers, which the asymmetric IPC window provides.  Keep this
        # backend-local and leave the default (symmetric) for other backends.
        self.window_flags = int(os.environ.get("FLAGCX_PPU_WINDOW_FLAGS", "0"))
        os.environ.setdefault("FLAGCX_VMM_ENABLE", "0")
        # The allocator shim only calls host FlagCX entry points; building it
        # with torch's CUDA extension path would require a CUDA toolchain on
        # a PPU-only host.  Other backends keep the default (True).
        self.allocator_with_cuda = False

    @staticmethod
    def _select(env_key, *candidates, directory: bool = False):
        override = os.environ.get(env_key)
        if override:
            path = Path(override).expanduser()
            exists = path.is_dir() if directory else path.is_file()
            if not exists:
                kind = "directory" if directory else "file"
                raise RuntimeError(f"{env_key} does not point to an existing {kind}: {path}")
            return path.resolve()
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate)
            if path.is_dir() if directory else path.is_file():
                return path.resolve()
        return None


flagcx_rt_conf = FlagcxRuntimeConfig(Path(__file__).parent)


class Distributed:

    def __init__(self):
        self.extern_libs = {"libflagcx": str(flagcx_rt_conf.bitcode_path)} if flagcx_rt_conf.is_available else {}

    def get_extern_libs(self):
        return self.extern_libs

    @property
    def is_available(self):
        return flagcx_rt_conf.is_available
