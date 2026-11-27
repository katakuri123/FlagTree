"""Compile-level coverage for the PPU FlagCX space barrier (Phase 5.1).

The verified scope is ``space=device`` and single-node ``space=world`` with
thread/warp/block groups.  Inter-node, grid/grid_axis and cluster/submesh
barriers must stay rejected by the compiler guard.
"""

import os
import shutil

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

pytest.importorskip("triton._C.libtriton.tle", reason="libtriton was built without FLAGTREE_TLE")


def _ppu_sdk_available() -> bool:
    sdk = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
    return bool(sdk and os.path.isfile(os.path.join(sdk, "bin", "ppu-llc"))) or bool(shutil.which("ppu-llc"))


import triton.experimental.tle.language as tle  # noqa: E402
from triton.experimental.tle import is_tle_supported  # noqa: E402
from triton.runtime import DistributedRtContext  # noqa: E402

_GPU_TARGET = GPUTarget("cuda", 80, 32)
_DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))
_WORLD_MESH = tle.device_mesh(tle.MeshConfig(node=1, device=2))
_INTER_MESH = tle.device_mesh(tle.MeshConfig(node=2, device=2))
_BLOCK_MESH = tle.device_mesh({"block": [("block_x", 2)]})


@triton.jit
def _barrier_ops(out, ctx: tl.constexpr, device_mesh: tl.constexpr, world_mesh: tl.constexpr):
    tle.distributed_barrier(device_mesh, device_dptr=ctx, space="device", barrier_kind="sync",
                            group_kind="block", index=0, context_id=0)
    tle.distributed_barrier(device_mesh, device_dptr=ctx, space="device", barrier_kind="arrive",
                            group_kind="warp", index=1, context_id=1)
    tle.distributed_barrier(device_mesh, device_dptr=ctx, space="device", barrier_kind="wait",
                            group_kind="warp", index=1, context_id=1)
    tle.distributed_barrier(world_mesh, device_dptr=ctx, space="world", barrier_kind="sync",
                            group_kind="block", index=2, context_id=0)
    tl.store(out, 1)


@triton.jit
def _barrier_thread_only(out, ctx: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(mesh, device_dptr=ctx, space="device", barrier_kind="sync",
                            group_kind="thread", index=7, context_id=3)
    tl.store(out, 1)


@triton.jit
def _barrier_world_only(out, ctx: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(mesh, device_dptr=ctx, space="world", barrier_kind="sync",
                            group_kind="block", index=0, context_id=0)
    tl.store(out, 1)


@triton.jit
def _barrier_world_warp(out, ctx: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(mesh, device_dptr=ctx, space="world", barrier_kind="sync",
                            group_kind="warp", index=0, context_id=0)
    tl.store(out, 1)


@triton.jit
def _barrier_inter(out, ctx: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(mesh, device_dptr=ctx, space="inter", barrier_kind="sync",
                            group_kind="block", index=0, context_id=0)
    tl.store(out, 1)


@triton.jit
def _barrier_invalid_group(out, ctx: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(mesh, device_dptr=ctx, space="device", barrier_kind="sync",
                            group_kind="grid", index=0, context_id=0)
    tl.store(out, 1)


@triton.jit
def _barrier_invalid_index(out, ctx: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(mesh, device_dptr=ctx, space="device", barrier_kind="sync",
                            group_kind="block", index=8, context_id=0)
    tl.store(out, 1)


@triton.jit
def _barrier_invalid_context(out, ctx: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(mesh, device_dptr=ctx, space="device", barrier_kind="sync",
                            group_kind="block", index=0, context_id=4)
    tl.store(out, 1)


@triton.jit
def _barrier_grid(out, mesh: tl.constexpr):
    tle.distributed_barrier(mesh)
    tl.store(out, 1)


def _compile(fn, signature, constexprs, options=None):
    source = triton.compiler.ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(source, target=_GPU_TARGET, options=options)


def _ctx():
    return DistributedRtContext(0x1000, 0x2000)


def test_barrier_is_declared_supported():
    assert is_tle_supported("distributed_barrier")


def test_barrier_space_lowering_uses_as0_device_api():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _barrier_ops,
        {"out": "*i32", "ctx": "constexpr", "device_mesh": "constexpr", "world_mesh": "constexpr"},
        {"ctx": _ctx(), "device_mesh": _DEVICE_MESH, "world_mesh": _WORLD_MESH},
        options={"num_warps": 1},
    )
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "distributed_barrier" in ttgir
    for symbol in ("flagcxDevBarrierSync", "flagcxDevBarrierArrive", "flagcxDevBarrierWait"):
        assert symbol in llir, symbol
    assert "inttoptr (i64" in llir
    assert "addrspacecast" not in llir
    # No NVIDIA grid-barrier inline PTX on the space path.
    assert "atom.add.release.gpu" not in llir
    assert "ld.acquire.gpu" not in llir
    # Collective barrier groups need the per-thread system fence.
    assert "ppu.fence.sc.sys" in llir
    assert "hgbin" in compiled.asm


def test_thread_group_is_rejected_as_unverified():
    # A thread-group FlagCX barrier must be executed by exactly one thread per
    # index (the epoch slot is shared per rank).  TLE has no thread-id mask in
    # this phase, so a normal distributed call would run it on every thread.
    with pytest.raises(NotImplementedError, match=r"group_kind='thread'.*UNVERIFIED"):
        _compile(_barrier_thread_only, {"out": "*i32", "ctx": "constexpr", "mesh": "constexpr"},
                 {"ctx": _ctx(), "mesh": _DEVICE_MESH})


def test_world_space_requires_proven_single_node(monkeypatch):
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    signature = {"out": "*i32", "ctx": "constexpr", "mesh": "constexpr"}
    constexprs = {"ctx": _ctx(), "mesh": _WORLD_MESH}
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    with pytest.raises(NotImplementedError, match=r"\(space='world'\) on multi-node"):
        _compile(_barrier_world_only, signature, constexprs)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    with pytest.raises(NotImplementedError, match=r"\(space='world'\) on multi-node"):
        _compile(_barrier_world_only, signature, constexprs)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    compiled = _compile(_barrier_world_only, signature, constexprs)
    assert "flagcxDevBarrierSync" in compiled.asm["llir"]


@pytest.mark.parametrize(
    ("fn", "mesh", "match"),
    [
        (_barrier_inter, _INTER_MESH, r"space='inter'"),
        (_barrier_invalid_index, _DEVICE_MESH, r"index=8"),
        (_barrier_invalid_context, _DEVICE_MESH, r"context_id=4"),
    ],
)
def test_invalid_space_index_context_are_rejected(fn, mesh, match):
    with pytest.raises(NotImplementedError, match=match):
        _compile(fn, {"out": "*i32", "ctx": "constexpr", "mesh": "constexpr"},
                 {"ctx": _ctx(), "mesh": mesh})


def test_invalid_group_is_rejected_by_tle_verifier(capfd):
    # The TLE op verifier rejects non FlagCX group kinds while building TTIR,
    # before the PPU capability guard runs.  The PPU guard keeps the same
    # check as defense in depth.  The Python exception is generic; the
    # verifier diagnostic is emitted on stderr.
    with pytest.raises(RuntimeError, match=r"error encountered during parsing"):
        _compile(_barrier_invalid_group, {"out": "*i32", "ctx": "constexpr", "mesh": "constexpr"},
                 {"ctx": _ctx(), "mesh": _DEVICE_MESH})
    assert "group_kind must be 'thread', 'warp', or 'block'" in capfd.readouterr().err


@triton.jit
def _barrier_warp_only(out, ctx: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(mesh, device_dptr=ctx, space="device", barrier_kind="sync",
                            group_kind="warp", index=0, context_id=0)
    tl.store(out, 1)


def test_warp_group_requires_single_warp_cta():
    with pytest.raises(NotImplementedError, match=r"group_kind='warp'\).*num_warps=1"):
        _compile(_barrier_warp_only, {"out": "*i32", "ctx": "constexpr", "mesh": "constexpr"},
                 {"ctx": _ctx(), "mesh": _DEVICE_MESH})
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(_barrier_warp_only,
                        {"out": "*i32", "ctx": "constexpr", "mesh": "constexpr"},
                        {"ctx": _ctx(), "mesh": _DEVICE_MESH}, options={"num_warps": 1})
    assert "flagcxDevBarrierSync" in compiled.asm["llir"]


def test_world_warp_group_is_rejected_as_unverified():
    with pytest.raises(NotImplementedError, match=r"space='world', group_kind='warp'.*UNVERIFIED"):
        _compile(_barrier_world_warp, {"out": "*i32", "ctx": "constexpr", "mesh": "constexpr"},
                 {"ctx": _ctx(), "mesh": _WORLD_MESH}, options={"num_warps": 1})


def test_grid_barrier_is_rejected():
    with pytest.raises(NotImplementedError, match=r"space=None"):
        _compile(_barrier_grid, {"out": "*i32", "mesh": "constexpr"}, {"mesh": _BLOCK_MESH})
