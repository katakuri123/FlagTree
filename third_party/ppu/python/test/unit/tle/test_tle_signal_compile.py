"""Compile-level coverage for ``tle.signal``/``tle.signal_wait`` on PPU.

Verifies the shared FlagCX signal lowering reaches AS0 device calls for all
wait kinds and signal spaces.  Runtime visibility lives in
``test_tle_signal_wait.py``.
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


@triton.jit
def _signal_inter_only(out, ctx: tl.constexpr, peer: tl.constexpr):
    tle.signal(ctx, peer, slot_id=0, op="inc", space="inter_node", group_kind="block")
    tl.store(out, 1)


@triton.jit
def _signal_block_only(out, ctx: tl.constexpr, peer: tl.constexpr):
    tle.signal(ctx, peer, slot_id=0, op="inc", space="intra_node", group_kind="block")
    tl.store(out, 1)


@triton.jit
def _signal_thread_only(out, ctx: tl.constexpr, peer: tl.constexpr):
    tle.signal(ctx, peer, slot_id=0, op="inc", space="intra_node", group_kind="thread")
    tl.store(out, 1)


@triton.jit
def _signal_wait_ops(out, ctx: tl.constexpr, peer: tl.constexpr):
    tle.signal(ctx, peer, slot_id=0, op="inc", space="intra_node", group_kind="block")
    tle.signal(ctx, peer, slot_id=1, value=2, op="add", space="intra_node", group_kind="block")
    tle.signal(ctx, peer, slot_id=2, op="inc", space="intra_node", group_kind="block")
    tle.signal(ctx, peer, slot_id=3, op="inc", space="world", group_kind="block")
    tle.signal(ctx, peer, slot_id=6, op="inc", space="intra_node", group_kind="warp", scope="device")
    tle.signal_wait(ctx, slot_id=0, wait_kind="signal", target=1, group_kind="block", order="acquire")
    tle.signal_wait(ctx, slot_id=4, wait_kind="counter", target=1, group_kind="block", order="relaxed")
    tle.signal_wait(ctx, slot_id=5, wait_kind="shadow", group_kind="block", order="acquire")
    tle.signal_wait(ctx, slot_id=7, wait_kind="signal", target=2, group_kind="warp", order="relaxed")
    tl.store(out, 1)


def _compile_fn(fn):
    ctx = DistributedRtContext(0x1000, 0x2000)
    source = triton.compiler.ASTSource(
        fn=fn,
        signature={"out": "*i32", "ctx": "constexpr", "peer": "constexpr"},
        constexprs={"ctx": ctx, "peer": 1},
    )
    return triton.compile(source, target=_GPU_TARGET)


def _compile():
    return _compile_fn(_signal_wait_ops)


def test_signal_primitives_are_declared_supported():
    assert is_tle_supported(("signal", "signal_wait"))


def test_signal_wait_lowering_reaches_as0_device_calls():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile()
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    assert "tle.signal" in ttgir
    assert "tle.signal_wait" in ttgir
    for symbol in ("flagcxDevSignalInc", "flagcxDevSignalAdd", "flagcxDevWaitSignal",
                   "flagcxDevWaitCounter", "flagcxDevWaitSignalMeetShadow"):
        assert symbol in llir, symbol
    # The comm pointer is generic AS0; the signal path does no AS cast.
    assert "inttoptr (i64" in llir
    assert "addrspacecast" not in llir
    # The inlined FlagCX signal uses PPU system-scope atomics.
    assert "ppu.atom.add.release.sys" in llir
    assert "ppu.atom.add.acq_rel.sys" in llir
    assert "ppu.fence.sc.sys" in llir
    assert "hgbin" in compiled.asm


def test_signal_inter_space_is_rejected():
    with pytest.raises(NotImplementedError, match=r"space='inter'"):
        _compile_fn(_signal_inter_only)


def test_collective_signal_emits_system_fence():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    llir = _compile_fn(_signal_block_only).asm["llir"]
    # The collective signal is emitted by one thread/lane; the per-thread
    # system fence publishes the whole group's prior payload stores.
    assert "ppu.fence.sc.sys" in llir


def test_thread_group_signal_has_no_collective_fence():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    llir = _compile_fn(_signal_thread_only).asm["llir"]
    # Every thread emits its own release signal, so no extra group fence.
    assert "ppu.fence.sc.sys" not in llir
    assert "ppu.atom.add.release.sys" in llir
