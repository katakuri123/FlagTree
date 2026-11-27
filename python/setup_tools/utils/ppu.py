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
"""PPU FlagCX dependency registration and build integration.

The PPU backend needs the FlagCX Device API host library and the matching
``libflagcx_device.bc``.  Both are built from the backend-local checkout
registered as ``flagcx_ppu`` so the shared ``flagcx`` pin used by the NVIDIA,
Iluvatar and HCU communication backends is not moved.

The build works around two properties of the PPU FlagCX build that are
verified on PPU SDK 2.1.0:

* ``COMPILE_KERNEL=1`` is required: the Device API net descriptors are
  constructed by a device-compiled helper linked into ``libflagcx.so``.
* The bitcode's ``check_ir`` target expects an ``alippu`` target triple while
  this SDK emits ``ppu``.  The produced bitcode is valid and matches the PPU
  ``libdevice.ppu.bc`` triple, so the build requests the bitcode target and
  the public-symbol check directly instead of ``llvm_ir``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

_BC_NAME = "libflagcx_device.bc"
_SO_NAME = "libflagcx.so"
_STAMP_NAME = ".flagcx_ppu_build_stamp"
_DEFAULT_ARCH = "ppu001"
_REQUIRED_HOST_SYMBOLS = ("flagcxDevNetLaunchConstruct", "flagcxDevNetSizeOf")


class PPUFlagCXError(RuntimeError):
    """Raised when the PPU FlagCX dependency cannot be built or installed."""


def _info(message: str) -> None:
    print(f"[PPU-FLAGCX] {message}", flush=True)


def _ppu_home() -> Path:
    home = Path(os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME") or "/usr/local/PPU_SDK").expanduser().resolve()
    if not (home / "bin" / "hgcc").is_file():
        raise PPUFlagCXError(f"PPU SDK not found at {home}: expected {home / 'bin' / 'hgcc'}")
    return home


def _bitcode_arch() -> str:
    arch = os.environ.get("FLAGCX_PPU_ARCH", _DEFAULT_ARCH).strip()
    if not arch or any(ch in arch for ch in "/ \t\n"):
        raise PPUFlagCXError(f"invalid FLAGCX_PPU_ARCH: {arch!r}")
    return arch


def _tool(name: str, home: Path) -> str:
    for candidate in (home / "bin" / name, Path("/usr/bin") / name):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise PPUFlagCXError(f"required tool {name!r} not found under {home}/bin or PATH")


def _run(cmd, cwd: Path, env=None) -> None:
    _info("run: " + " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=True, env=env)


def _source_revision(source: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(source), text=True).strip()
    except Exception:
        return "unknown"


def _ensure_json_headers(source: Path) -> None:
    header = source / "third-party" / "json" / "single_include" / "nlohmann" / "json.hpp"
    if header.is_file():
        return
    override = os.environ.get("JSON_INCLUDE_DIR")
    if override and (Path(override) / "nlohmann" / "json.hpp").is_file():
        return
    if (source / ".git").exists():
        _info("initializing FlagCX third-party/json submodule")
        subprocess.run(["git", "submodule", "update", "--init", "--depth", "1", "third-party/json"],
                       cwd=str(source), check=True)
    if not header.is_file():
        raise PPUFlagCXError("FlagCX needs nlohmann/json.hpp. Initialize FlagCX's third-party/json "
                             "submodule or set JSON_INCLUDE_DIR.")


def _host_lib(source: Path) -> Path:
    return source / "build" / "lib" / _SO_NAME


def _bitcode_dir(source: Path, arch: str) -> Path:
    return source / "build" / "ppu" / arch


def _bitcode_path(source: Path, arch: str) -> Path:
    return _bitcode_dir(source, arch) / "lib" / _BC_NAME


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_headers(include_dir: Path) -> str:
    digest = hashlib.sha256()
    for header in sorted(include_dir.glob("*.h")):
        digest.update(header.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(header.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sdk_version(home: Path) -> str:
    release = home / "release.yaml"
    if release.is_file():
        for line in release.read_text().splitlines():
            if line.strip().startswith("version:"):
                return line.split(":", 1)[1].strip()
    magic = home / "VERSION.txt"
    if magic.is_file():
        return magic.read_text().strip()
    return "unknown"


def _bitcode_target(bitcode: Path, home: Path) -> tuple[str, str]:
    """Read the bitcode's triple/datalayout from the first lines of llvm-dis."""
    llvm_dis = home / "bin" / "llvm-dis"
    if not llvm_dis.is_file():
        return "unknown", "unknown"
    proc = subprocess.Popen([str(llvm_dis), "-o", "-", str(bitcode)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            text=True)
    head = []
    try:
        for _ in range(6):
            line = proc.stdout.readline()
            if not line:
                break
            head.append(line)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    triple = "unknown"
    datalayout = "unknown"
    for line in head:
        if line.startswith("target triple"):
            triple = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("target datalayout"):
            datalayout = line.split("=", 1)[1].strip().strip('"')
    return triple, datalayout


def _stamp_text(source: Path, arch: str) -> str:
    return ("flagcx_commit={}\narch={}\n"
            "flags=USE_PPU=1 USE_ACCL_BAREX=1 USE_UCX=0 COMPILE_KERNEL=1\n").format(_source_revision(source), arch)


def _forced_rebuild() -> bool:
    return any(os.environ.get(key, "").strip().lower() in ("1", "on", "true", "yes")
               for key in ("FLAGCX_RECOMPILE", "DIST_RECOMPILE"))


class PPUFlagCXRegistrar:

    def __init__(self, configs, backend):
        self.flagtree_dir = Path(configs.flagtree_root_dir).resolve()
        self.source = Path(backend.dst_path).resolve()
        self.backend_name = configs.flagtree_backend or "ppu"
        self.home = _ppu_home()
        self.arch = _bitcode_arch()
        backend_lib_dir = self.flagtree_dir / "third_party" / self.backend_name / "backend" / "lib"
        self.stamp_path = backend_lib_dir / _STAMP_NAME
        self.installed_bitcode = backend_lib_dir / _BC_NAME
        self.installed_shared = self.flagtree_dir / "python" / "triton" / "_C" / _SO_NAME

    def _artifacts_ready(self) -> bool:
        if _forced_rebuild() or not self.stamp_path.is_file():
            return False
        try:
            expected = _stamp_text(self.source, self.arch)
        except OSError:
            return False
        return (self.stamp_path.read_text() == expected and _host_lib(self.source).is_file()
                and _bitcode_path(self.source, self.arch).is_file() and self.installed_shared.is_file()
                and self.installed_bitcode.is_file())

    def _metadata_ready(self) -> bool:
        metadata_path = self.installed_bitcode.parent / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            return False
        required = ("metadata_version", "flagcx_commit", "arch", "target_triple",
                    "host_library_sha256", "device_bitcode_sha256", "headers_sha256")
        return all(key in metadata for key in required)

    def _build(self) -> None:
        if not self.source.is_dir():
            raise PPUFlagCXError(f"FlagCX source not found: {self.source}")
        _ensure_json_headers(self.source)
        jobs = str(min(32, os.cpu_count() or 1))
        env = os.environ.copy()
        env.setdefault("DEVICE_HOME", str(self.home))
        env.setdefault("CCL_HOME", str(self.home))
        make = _tool("make", self.home)
        common = [make, f"DEVICE_HOME={self.home}", f"CCL_HOME={self.home}"]

        # Do not pass DEVICE_BITCODE_ARCH here: makefiles/ppu.mk forwards it to
        # hgcc's -arch, which only accepts vm_*/ppu_10/15/17/all.  The host
        # helper objects use hgcc's default device target; the separately
        # built Device IR bitcode uses --ppu-arch=ppu001.
        _run([make, "-j", jobs, "USE_PPU=1", "USE_ACCL_BAREX=1", "USE_UCX=0", "COMPILE_KERNEL=1",
              f"DEVICE_HOME={self.home}", f"CCL_HOME={self.home}"], cwd=self.source, env=env)

        bitcode = _bitcode_path(self.source, self.arch)
        _run(common + ["-C", "bindings/ir/ppu", f"BITCODE_LIB_ARCH={self.arch}",
                       f"BUILDDIR={_bitcode_dir(self.source, self.arch)}", "headers", "check_public_symbols",
                       str(bitcode)], cwd=self.source, env=env)

        host_lib = _host_lib(self.source)
        if not host_lib.is_file():
            raise PPUFlagCXError(f"FlagCX host library was not produced: {host_lib}")
        if not bitcode.is_file():
            raise PPUFlagCXError(f"FlagCX PPU device bitcode was not produced: {bitcode}")
        include_dir = _bitcode_dir(self.source, self.arch) / "include"
        if not include_dir.is_dir():
            raise PPUFlagCXError(f"FlagCX PPU device headers were not produced: {include_dir}")

        nm = _tool("nm", self.home)
        defined = subprocess.run([nm, "-D", "--defined-only", str(host_lib)], check=True, capture_output=True,
                                 text=True).stdout
        missing = [symbol for symbol in _REQUIRED_HOST_SYMBOLS if symbol not in defined]
        if missing:
            raise PPUFlagCXError(f"FlagCX host library {host_lib} lacks Device API symbols: {', '.join(missing)} "
                                 "(was COMPILE_KERNEL=1 honored?)")

    def _install(self) -> None:
        host_lib = _host_lib(self.source)
        bitcode = _bitcode_path(self.source, self.arch)
        include_dir = _bitcode_dir(self.source, self.arch) / "include"
        self.installed_shared.parent.mkdir(parents=True, exist_ok=True)
        self.installed_bitcode.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(host_lib, self.installed_shared)
        shutil.copy2(bitcode, self.installed_bitcode)

        wrapper = self.source / "src" / "flagcx" / "api.py"
        if not wrapper.is_file():
            wrapper = self.source / "plugin" / "interservice" / "flagcx_wrapper.py"
        if not wrapper.is_file():
            raise PPUFlagCXError(f"FlagCX Python wrapper not found under {self.source}")

        target_triple, target_datalayout = _bitcode_target(bitcode, self.home)
        metadata = {
            "metadata_version": 1,
            "backend": self.backend_name,
            "arch": self.arch,
            "target_triple": target_triple,
            "target_datalayout": target_datalayout,
            "sdk_version": _sdk_version(self.home),
            "build_flags": "USE_PPU=1 USE_ACCL_BAREX=1 USE_UCX=0 COMPILE_KERNEL=1",
            "flagcx_commit": _source_revision(self.source),
            "host_library": _SO_NAME,
            "device_bitcode": _BC_NAME,
            "host_library_sha256": _sha256_file(host_lib),
            "device_bitcode_sha256": _sha256_file(bitcode),
            "headers_sha256": _sha256_headers(include_dir),
            "wrapper_sha256": _sha256_file(wrapper),
        }
        (self.installed_bitcode.parent / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        language_dir = self.flagtree_dir / "python" / "triton" / "experimental" / "tle" / "language"
        backend_dir = self.flagtree_dir / "third_party" / self.backend_name / "backend"
        language_dir.mkdir(parents=True, exist_ok=True)
        backend_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wrapper, language_dir / "flagcx_wrapper.py")
        shutil.copy2(wrapper, backend_dir / "flagcx_wrapper.py")

        for destination in (language_dir / "include", backend_dir / "include"):
            destination.mkdir(parents=True, exist_ok=True)
            for header in sorted(include_dir.glob("*.h")):
                shutil.copy2(header, destination / header.name)
            manifest = include_dir / ".flagcx_device_headers"
            if manifest.is_file():
                shutil.copy2(manifest, destination / manifest.name)

        self.stamp_path.write_text(_stamp_text(self.source, self.arch))
        _info(f"installed {self.installed_shared} and {self.installed_bitcode}")

    def run(self) -> None:
        if self._artifacts_ready():
            if not self._metadata_ready():
                _info("refreshing FlagCX PPU metadata from existing artifacts")
                self._install()
                return
            _info("FlagCX PPU artifacts are up to date; skipping rebuild "
                  "(set FLAGCX_RECOMPILE=1 to force)")
            return
        self._build()
        self._install()


def handle_flagcx_ppu(configs, backend, cache=None, **kwargs):
    """Build and install the backend-local PPU FlagCX checkout."""
    if configs is None or backend is None:
        raise PPUFlagCXError("handle_flagcx_ppu requires configs and backend")
    PPUFlagCXRegistrar(configs, backend).run()


def get_package_data(backends):
    # The FlagCX artifacts and device headers are generated while setup.py is
    # running, so they are not discoverable as ordinary Python modules.
    return {
        "triton._C": ["libflagcx.so"],
        "triton.backends.ppu": ["lib/libflagcx_device.bc", "lib/metadata.json"],
        "triton.experimental.tle.language": ["flagcx_wrapper.py", "include/*.h"],
    }


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    return
