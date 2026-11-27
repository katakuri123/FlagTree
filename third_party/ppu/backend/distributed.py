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

import functools
import hashlib
import json
import os
import warnings
from pathlib import Path

_BITCODE_NAME = "libflagcx_device.bc"
_SHARED_NAME = "libflagcx.so"
_INCLUDE_NAME = "include"
_METADATA_NAME = "metadata.json"
_METADATA_VERSION = 1
_EXPECTED_TRIPLE = "ppu"
_ENV_KEYS = ("USE_FLAGCX", "USE_DIST", "USE_DISTRIBUTED", "USE_TLE_DIST", "USE_TLE_DISTRIBUTED")


@functools.lru_cache(maxsize=32)
def _file_sha256_cached(path: str, mtime_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    stat = path.stat()
    return _file_sha256_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _headers_sha256(include_dir: Path) -> str:
    digest = hashlib.sha256()
    for header in sorted(include_dir.glob("*.h")):
        digest.update(header.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(header.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
        # Do not mutate the environment here: this module is imported by the
        # compiler (compile-only runs) and may share a process with other
        # FlagCX users.  apply_runtime_env() runs immediately before the
        # communicator is created.
        self.window_flags = int(os.environ.get("FLAGCX_PPU_WINDOW_FLAGS", "0"))
        # The allocator shim only calls host FlagCX entry points; building it
        # with torch's CUDA extension path would require a CUDA toolchain on
        # a PPU-only host.  Other backends keep the default (True).
        self.allocator_with_cuda = False

        self.metadata = None
        self.metadata_paths = []
        self.metadata_error = None
        self._load_bundle_metadata()

    def _metadata_candidates(self) -> list[Path]:
        candidates: list[Path] = []

        def add(candidate: Path) -> None:
            if candidate.is_file() and candidate not in candidates:
                candidates.append(candidate)

        if self.module_path is not None:
            add(self.module_path / _METADATA_NAME)
        for path in (self.bitcode_path, self.shared_lib_path, self.include_path):
            if path is None:
                continue
            add(path / _METADATA_NAME if path.is_dir() else path.parent / _METADATA_NAME)
        return candidates

    def _load_bundle_metadata(self) -> None:
        """Lightweight metadata read; hashes are checked at runtime init."""
        try:
            paths = self._metadata_candidates()
            self.metadata_paths = paths
            if not paths:
                return
            merged: dict = {}
            consistency_keys = ("metadata_version", "flagcx_commit", "arch", "target_triple",
                                "host_library_sha256", "device_bitcode_sha256", "headers_sha256",
                                "wrapper_sha256")
            for path in paths:
                try:
                    data = json.loads(path.read_text())
                except (OSError, ValueError) as exc:
                    raise RuntimeError(f"invalid FlagCX metadata {path}: {exc}") from exc
                if not isinstance(data, dict):
                    raise RuntimeError(f"invalid FlagCX metadata {path}: expected a JSON object")
                version = data.get("metadata_version")
                if version is not None and version > _METADATA_VERSION:
                    raise RuntimeError(f"FlagCX metadata {path} uses unsupported version {version}")
                for key in consistency_keys:
                    value = data.get(key)
                    if value is None:
                        continue
                    if key in merged and merged[key] != value:
                        raise RuntimeError(
                            f"FlagCX bundle mismatch for {key}: {merged[key]!r} vs {value!r} ({path})")
                    merged[key] = value
            triple = merged.get("target_triple")
            if triple is not None and triple != _EXPECTED_TRIPLE:
                raise RuntimeError(
                    f"FlagCX device bitcode target triple is {triple!r}, expected {_EXPECTED_TRIPLE!r}")
            expected_arch = os.environ.get("FLAGCX_PPU_ARCH", "ppu001").strip()
            arch = merged.get("arch")
            if arch is not None and expected_arch and arch != expected_arch:
                raise RuntimeError(f"FlagCX bundle arch is {arch!r}, expected {expected_arch!r}")
            self.metadata = merged
        except RuntimeError as exc:
            self.metadata_error = str(exc)

    def validate_bundle(self) -> None:
        """Verify host library, device bitcode, headers and wrapper consistency."""
        if not self.is_available:
            return
        if self.metadata_error:
            raise RuntimeError(self.metadata_error)
        if self.metadata is None:
            warnings.warn(
                "FlagCX bundle has no metadata.json; host/device artifact consistency "
                "cannot be verified",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        mismatches = []
        for key, path in (
            ("host_library_sha256", self.shared_lib_path),
            ("device_bitcode_sha256", self.bitcode_path),
            ("wrapper_sha256",
             self.triton_path / "experimental" / "tle" / "language" / "flagcx_wrapper.py"),
        ):
            expected = self.metadata.get(key)
            if not expected:
                continue
            actual = _file_sha256(path)
            if expected != actual:
                mismatches.append(f"{key}: metadata={expected} actual={actual}")
        expected_headers = self.metadata.get("headers_sha256")
        if expected_headers:
            actual = _headers_sha256(self.include_path)
            if expected_headers != actual:
                mismatches.append(f"headers_sha256: metadata={expected_headers} actual={actual}")
        if mismatches:
            raise RuntimeError("FlagCX bundle ABI mismatch:\n  " + "\n  ".join(mismatches))

        bitcode_override = os.environ.get("FLAGCX_BITCODE_PATH")
        lib_override = os.environ.get("FLAGCX_LIB_PATH")
        if bitcode_override and lib_override:
            bitcode_parent = Path(bitcode_override).expanduser().resolve().parent
            lib_parent = Path(lib_override).expanduser().resolve().parent
            if bitcode_parent != lib_parent and len(self.metadata_paths) < 2:
                warnings.warn(
                    "FLAGCX_BITCODE_PATH and FLAGCX_LIB_PATH point to different bundle roots "
                    "but only one metadata.json was found; version consistency cannot be verified",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def apply_runtime_env(self) -> None:
        """Apply PPU FlagCX defaults just before communicator initialization.

        User-provided values always win.  The validated PPU path uses
        FLAGCX_VMM_ENABLE=0 with the asymmetric IPC window; the symmetric
        window is rejected because its VMM signal buffers cannot be exported
        over IPC on PPU.
        """
        if not self.enabled:
            return
        if self.window_flags & 0x1:
            raise RuntimeError(
                "PPU does not support the symmetric FlagCX window "
                "(FLAGCX_PPU_WINDOW_FLAGS=1): VMM signal buffers cannot be "
                "IPC-exported on PPU. Keep FLAGCX_PPU_WINDOW_FLAGS=0 (asymmetric).")
        vmm = os.environ.get("FLAGCX_VMM_ENABLE")
        if vmm is None:
            os.environ["FLAGCX_VMM_ENABLE"] = "0"
        elif vmm.strip().lower() in ("1", "on", "true"):
            warnings.warn(
                "FLAGCX_VMM_ENABLE=1 is set explicitly; the validated PPU TLE path "
                "uses FLAGCX_VMM_ENABLE=0 with the asymmetric IPC window.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.validate_bundle()

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
