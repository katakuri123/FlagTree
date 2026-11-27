"""Runtime configuration and FlagCX bundle consistency tests for PPU."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("triton._C.libtriton.tle", reason="libtriton was built without FLAGTREE_TLE")

import triton  # noqa: E402
from triton.backends.ppu.compiler import HGGCOptions  # noqa: E402
from triton.backends.ppu.distributed import FlagcxRuntimeConfig  # noqa: E402

_TRITON_PATH = Path(triton.__file__).resolve().parent
_WRAPPER = _TRITON_PATH / "experimental" / "tle" / "language" / "flagcx_wrapper.py"
_REAL_INCLUDE = _TRITON_PATH / "experimental" / "tle" / "language" / "include"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _headers_hash(headers: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(headers):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(headers[name])
        digest.update(b"\0")
    return digest.hexdigest()


def _make_bundle(root, *, commit="deadbeef", bitcode=b"bitcode", shared=b"shared",
                 target_triple="ppu", write_metadata=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / "libflagcx_device.bc").write_bytes(bitcode)
    (root / "libflagcx.so").write_bytes(shared)
    include = root / "include"
    include.mkdir(exist_ok=True)
    headers = {"flagcx.h": b"header"}
    for name, content in headers.items():
        (include / name).write_bytes(content)
    if write_metadata:
        metadata = {
            "metadata_version": 1,
            "backend": "ppu",
            "arch": "ppu001",
            "target_triple": target_triple,
            "flagcx_commit": commit,
            "host_library_sha256": _sha(shared),
            "device_bitcode_sha256": _sha(bitcode),
            "headers_sha256": _headers_hash(headers),
            "wrapper_sha256": _sha(_WRAPPER.read_bytes()),
        }
        (root / "metadata.json").write_text(json.dumps(metadata))
    return root


@pytest.fixture(autouse=True)
def clean_flagcx_env(monkeypatch):
    for key in ("FLAGCX_MODULE_PATH", "FLAGCX_BITCODE_PATH", "FLAGCX_LIB_PATH",
                "FLAGCX_INCLUDE_PATH", "FLAGCX_PPU_WINDOW_FLAGS", "FLAGCX_VMM_ENABLE",
                "FLAGCX_PPU_ARCH"):
        monkeypatch.delenv(key, raising=False)


def _config(monkeypatch, tmp_path, root):
    monkeypatch.setenv("FLAGCX_MODULE_PATH", str(root))
    return FlagcxRuntimeConfig(tmp_path / "backend")


def test_import_does_not_set_vmm(monkeypatch):
    env = {k: v for k, v in os.environ.items() if k != "FLAGCX_VMM_ENABLE"}
    result = subprocess.run(
        [sys.executable, "-c",
         "import os; import triton.backends.ppu.distributed; "
         "print(os.environ.get('FLAGCX_VMM_ENABLE', '<unset>'))"],
        env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "<unset>"


def test_apply_runtime_env_sets_ppu_default(monkeypatch, tmp_path):
    root = _make_bundle(tmp_path / "bundle")
    conf = _config(monkeypatch, tmp_path, root)
    assert "FLAGCX_VMM_ENABLE" not in os.environ
    conf.apply_runtime_env()
    assert os.environ["FLAGCX_VMM_ENABLE"] == "0"


def test_explicit_vmm_is_preserved(monkeypatch, tmp_path):
    root = _make_bundle(tmp_path / "bundle")
    conf = _config(monkeypatch, tmp_path, root)
    monkeypatch.setenv("FLAGCX_VMM_ENABLE", "1")
    with pytest.warns(RuntimeWarning, match="FLAGCX_VMM_ENABLE=1"):
        conf.apply_runtime_env()
    assert os.environ["FLAGCX_VMM_ENABLE"] == "1"


def test_symmetric_window_is_rejected(monkeypatch, tmp_path):
    root = _make_bundle(tmp_path / "bundle")
    monkeypatch.setenv("FLAGCX_PPU_WINDOW_FLAGS", "1")
    conf = _config(monkeypatch, tmp_path, root)
    with pytest.raises(RuntimeError, match="symmetric FlagCX window"):
        conf.apply_runtime_env()


def test_bitcode_hash_mismatch_is_rejected(monkeypatch, tmp_path):
    root = _make_bundle(tmp_path / "bundle")
    conf = _config(monkeypatch, tmp_path, root)
    (root / "libflagcx_device.bc").write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="device_bitcode_sha256"):
        conf.validate_bundle()


def test_unsupported_target_triple_is_rejected(monkeypatch, tmp_path):
    root = _make_bundle(tmp_path / "bundle", target_triple="alippu")
    conf = _config(monkeypatch, tmp_path, root)
    with pytest.raises(RuntimeError, match="target triple"):
        conf.validate_bundle()


def test_missing_metadata_warns(monkeypatch, tmp_path):
    root = _make_bundle(tmp_path / "bundle", write_metadata=False)
    conf = _config(monkeypatch, tmp_path, root)
    with pytest.warns(RuntimeWarning, match="no metadata.json"):
        conf.validate_bundle()


def test_extern_lib_content_is_part_of_the_cache_key(tmp_path):
    lib = tmp_path / "libtest.bc"
    lib.write_bytes(b"version-one")
    first = HGGCOptions(arch="sm80", extern_libs={"libtest": str(lib)}).hash()
    lib.write_bytes(b"version-two")
    second = HGGCOptions(arch="sm80", extern_libs={"libtest": str(lib)}).hash()
    assert first != second


def test_mixed_bundle_roots_are_rejected(monkeypatch, tmp_path):
    first = _make_bundle(tmp_path / "first", commit="aaaa", bitcode=b"first-bitcode")
    second = _make_bundle(tmp_path / "second", commit="bbbb", shared=b"second-shared")
    monkeypatch.setenv("FLAGCX_BITCODE_PATH", str(first / "libflagcx_device.bc"))
    monkeypatch.setenv("FLAGCX_LIB_PATH", str(second / "libflagcx.so"))
    monkeypatch.setenv("FLAGCX_INCLUDE_PATH", str(_REAL_INCLUDE))
    conf = FlagcxRuntimeConfig(tmp_path / "backend")
    with pytest.raises(RuntimeError, match="bundle mismatch"):
        conf.validate_bundle()
