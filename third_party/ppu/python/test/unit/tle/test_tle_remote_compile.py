"""Compile-level coverage for ``tle.remote`` on the PPU backend.

Phase 3 enables the intra-node device space only.  The node space must stay
rejected by the PPU compiler guard until the inter-node path is validated.
"""

import os
import re
import shutil

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.errors import CompilationError

pytest.importorskip("triton._C.libtriton.tle", reason="libtriton was built without FLAGTREE_TLE")


def _ppu_sdk_available() -> bool:
    sdk = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
    return bool(sdk and os.path.isfile(os.path.join(sdk, "bin", "ppu-llc"))) or bool(shutil.which("ppu-llc"))


import triton.experimental.tle.language as tle  # noqa: E402
from triton.experimental.tle import is_tle_supported  # noqa: E402
from triton.runtime import DistributedRtContext  # noqa: E402

_GPU_TARGET = GPUTarget("cuda", 80, 32)


@triton.jit
def _remote_read_write(out, ctx: tl.constexpr, peer: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    remote_read = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=peer, offset=0)
    remote_write = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=peer, offset=BLOCK)
    values = tl.load(remote_read + offs)
    tl.store(out + offs, values)
    tl.store(remote_write + offs, values + 1.0)


@triton.jit
def _remote_without_offset(out, ctx: tl.constexpr, peer: tl.constexpr):
    remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=peer)
    tl.store(out, tl.load(remote))


@triton.jit
def _remote_node_space(out, ctx: tl.constexpr, peer: tl.constexpr):
    remote = tle.remote(ctx, space="node", dtype=tl.float32, shard_id=peer)
    tl.store(out, tl.load(remote))


def _compile(fn, signature, constexprs):
    source = triton.compiler.ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(source, target=_GPU_TARGET)


def test_remote_is_declared_supported():
    assert is_tle_supported("remote")


def test_remote_device_lowering_uses_intra_pointer_abi():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    ctx = DistributedRtContext(0x1000, 0x2000)
    compiled = _compile(
        _remote_read_write,
        {"out": "*fp32", "ctx": "constexpr", "peer": "constexpr", "BLOCK": "constexpr"},
        {"ctx": ctx, "peer": 1, "BLOCK": 64},
    )
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "remote_pointers" in ttgir
    assert "flagcxGetIntraPointerC" in llir
    # flagcxGetIntraPointerC returns a generic pointer; the PPU kernel's
    # global address space is AS1, so the ABI boundary performs one cast.
    assert re.search(r"addrspacecast ptr .* to ptr addrspace\(1\)", llir)
    assert "inttoptr (i64" in llir
    assert "hgbin" in compiled.asm


def test_remote_device_requires_offset():
    ctx = DistributedRtContext(0x1000, 0x2000)
    with pytest.raises(CompilationError, match="device space remote pointers require an offset"):
        _compile(
            _remote_without_offset,
            {"out": "*fp32", "ctx": "constexpr", "peer": "constexpr"},
            {"ctx": ctx, "peer": 1},
        )


def test_remote_node_space_is_rejected():
    ctx = DistributedRtContext(0x1000, 0x2000)
    with pytest.raises(NotImplementedError, match=r"tle\.remote\(space='node'\)"):
        _compile(
            _remote_node_space,
            {"out": "*fp32", "ctx": "constexpr", "peer": "constexpr"},
            {"ctx": ctx, "peer": 1},
        )
