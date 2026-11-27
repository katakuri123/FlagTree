"""shard_id coverage for the PPU backend.

The rank axes (``device``/``node``) lower to the shared FlagCX rank accessors;
the mesh axes are pure ``program_id`` arithmetic.  This file keeps the
single-process checks in the PPU unit suite; the multi-rank runtime check
lives in ``test_tle_shard_id_distributed.py``.
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
from triton.experimental.tle.language.distributed import _resolve_launch_axis  # noqa: E402
from triton.runtime import DistributedRtContext  # noqa: E402

_GPU_TARGET = GPUTarget("cuda", 80, 32)


@triton.jit
def _mesh_coordinates(out, mesh: tl.constexpr, ndim: tl.constexpr):
    pid = tl.program_id(0)
    for axis in tl.static_range(ndim):
        tl.store(out + pid * ndim + axis, tle.shard_id(mesh, axis))


@triton.jit
def _invalid_axis(out, mesh: tl.constexpr):
    tl.store(out, tle.shard_id(mesh, "missing_axis"))


@triton.jit
def _rank_axes(out, ctx: tl.constexpr, mesh: tl.constexpr):
    device_axis = tle.shard_id(mesh, "device", device_dptr=ctx)
    node_axis = tle.shard_id(mesh, "node", device_dptr=ctx)
    tl.store(out + 0, device_axis)
    tl.store(out + 1, node_axis)


def _compile(fn, signature, constexprs):
    source = triton.compiler.ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(source, target=_GPU_TARGET)


def test_shard_id_is_declared_supported():
    assert is_tle_supported("shard_id")


def test_shard_id_rank_axes_compile_to_flagcx_accessors():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    ctx = DistributedRtContext(0x1000, 0x2000)
    compiled = _compile(
        _rank_axes,
        {"out": "*i32", "ctx": "constexpr", "mesh": "constexpr"},
        {"ctx": ctx, "mesh": tle.device_mesh(tle.MeshConfig(node=2, device=2))},
    )
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "get_device_id" in ttgir
    assert "get_world_rank" in ttgir
    assert "get_num_pes" in ttgir
    # The FlagCX bitcode is inlined before llir; the generic AS0 conversion
    # and the PPU platform metadata are the surviving witnesses.
    assert "inttoptr (i64" in llir
    assert "PpuPlatform" in llir
    assert "hgbin" in compiled.asm


@pytest.mark.parametrize("shape", [(2, 1), (2, 1, 2), (1, 4), (2, 3)])
def test_shard_id_mesh_axes_match_program_coordinates(shape):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a PPU device")
    mesh = tle.device_mesh({"block": [(f"axis{i}", size) for i, size in enumerate(shape)]})
    count = 1
    for size in shape:
        count *= size
    output = torch.full((count, len(shape)), -1, device="cuda", dtype=torch.int32)
    _mesh_coordinates[(count, )](output, mesh, len(shape))

    expected = []
    for pid in range(count):
        row = []
        for axis, size in enumerate(shape):
            stride = 1
            for trailing in shape[axis + 1:]:
                stride *= trailing
            row.append(pid // stride % size)
        expected.append(row)
    torch.testing.assert_close(output.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0)


def test_shard_id_rejects_unknown_axis():
    mesh = tle.device_mesh({"block": [("block_y", 2), ("block_x", 4)]})
    with pytest.raises(ValueError, match="unknown mesh axis"):
        _resolve_launch_axis(mesh, "missing_axis")
    with pytest.raises(IndexError, match="out of range"):
        _resolve_launch_axis(mesh, 2)


def test_shard_id_rejects_unknown_axis_at_compile_time():
    mesh = tle.device_mesh({"block": [("block_x", 2)]})
    with pytest.raises(Exception, match="unknown mesh axis"):
        _compile(_invalid_axis, {"out": "*i32", "mesh": "constexpr"}, {"mesh": mesh})
