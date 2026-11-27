"""Multi-rank runtime check for ``tle.shard_id`` on PPU.

Run with torchrun on a single node (or one process per node for a real
multi-node run)::

    LOCAL_WORLD_SIZE=2 torchrun --nproc_per_node=2 --nnodes=1 \
        third_party/ppu/python/test/unit/tle/test_tle_shard_id_distributed.py

The PPU FlagCX runtime currently needs the P2P staging buffers to be smaller
than FlagCX's 64 MB default; ``test_tle_shard_id_distributed.sh`` applies the
validated values unless the caller overrides them.  Without
``WORLD_SIZE``/``LOCAL_WORLD_SIZE`` the module is skipped, so the ordinary
single-process PPU unit suite is unaffected.
"""

import os

import pytest

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "0"))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
if WORLD_SIZE <= 0 or LOCAL_WORLD_SIZE <= 0 or WORLD_SIZE % LOCAL_WORLD_SIZE:
    pytest.skip("torchrun with WORLD_SIZE/LOCAL_WORLD_SIZE is required", allow_module_level=True)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import triton.experimental.tle.language as tle  # noqa: E402
from triton.experimental.tle.language.distributed import _get_world_rank, n_pes  # noqa: E402


@triton.jit
def _rank_components(out, ctx: tl.constexpr):
    tl.store(out + 0, _get_world_rank(ctx))
    tl.store(out + 1, n_pes(ctx))


@triton.jit
def _shard_axes(out, ctx: tl.constexpr, mesh: tl.constexpr):
    device_axis = tle.shard_id(mesh, "device", device_dptr=ctx)
    node_axis = tle.shard_id(mesh, "node", device_dptr=ctx)
    tl.store(out + 0, device_axis)
    tl.store(out + 1, node_axis)


def test_shard_id_device_and_node_axes():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    tle.init_communicator()
    buf = torch.zeros(64, dtype=torch.float32, device="cuda") + rank
    ctx = tle.create_dist_tensor(buf)
    mesh = tle.device_mesh(
        tle.MeshConfig(node=WORLD_SIZE // LOCAL_WORLD_SIZE, device=LOCAL_WORLD_SIZE))

    components = torch.full((2, ), -1, dtype=torch.int32, device="cuda")
    _rank_components[(1, )](components, ctx, num_warps=4)
    world_rank, intra_size = components.cpu().tolist()
    assert world_rank == rank, (rank, world_rank)
    assert intra_size == LOCAL_WORLD_SIZE, (rank, intra_size)

    out = torch.full((2, ), -1, dtype=torch.int32, device="cuda")
    _shard_axes[(1, )](out, ctx, mesh, num_warps=4)
    torch.cuda.synchronize()

    device_axis, node_axis = out.cpu().tolist()
    print(f"[rank {rank}] world={world_rank} intra={intra_size} "
          f"device={device_axis} node={node_axis} "
          f"(expected {local_rank}/{rank // LOCAL_WORLD_SIZE})", flush=True)
    assert device_axis == local_rank, (rank, device_axis, local_rank)
    assert node_axis == rank // LOCAL_WORLD_SIZE, (rank, node_axis)
    assert dist.get_world_size() == WORLD_SIZE

    tle.cleanup_communicator()
    print(f"[rank {rank}] PASS", flush=True)


if __name__ == "__main__":
    test_shard_id_device_and_node_axes()
