"""Multi-rank runtime coverage for ``tle.remote(space="device")`` on PPU.

Run with torchrun on a single node (or one process per node for a real
multi-node run)::

    LOCAL_WORLD_SIZE=2 torchrun --nproc_per_node=2 --nnodes=1 \
        third_party/ppu/python/test/unit/tle/test_tle_remote_intra_node.py

``test_tle_remote_intra_node.sh`` applies the validated PPU FlagCX defaults.
Without ``WORLD_SIZE``/``LOCAL_WORLD_SIZE`` the module is skipped, so the
ordinary single-process PPU unit suite is unaffected.
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

SMALL_N = int(os.environ.get("REMOTE_N", "4096"))
LARGE_N = int(os.environ.get("REMOTE_LARGE_N", str(1 << 20)))
GRID_BLOCK = 1024

SMALL_DATA = 0
SMALL_WB = SMALL_N
LARGE_DATA = 2 * SMALL_N
LARGE_WB = LARGE_DATA + LARGE_N
BUFFER_ELEMS = LARGE_WB + LARGE_N


@triton.jit
def _read_contig(out, ctx: tl.constexpr, peer: tl.constexpr, offset: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=peer, offset=offset)
    tl.store(out + offs, tl.load(remote + offs, volatile=True))


@triton.jit
def _read_strided(out, ctx: tl.constexpr, peer: tl.constexpr, offset: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=peer, offset=offset)
    tl.store(out + offs, tl.load(remote + offs * 2, volatile=True))


@triton.jit
def _read_masked(out, ctx: tl.constexpr, peer: tl.constexpr, offset: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=peer, offset=offset)
    tl.store(out + offs, tl.load(remote + offs, mask=offs < BLOCK // 2, other=-7.0, volatile=True))


@triton.jit
def _read_int32(out, ctx: tl.constexpr, peer: tl.constexpr, offset: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    remote = tle.remote(ctx, space="device", dtype=tl.int32, shard_id=peer, offset=offset)
    tl.store(out + offs, tl.load(remote + offs, volatile=True))


@triton.jit
def _read_fp16(out, ctx: tl.constexpr, peer: tl.constexpr, offset: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    remote = tle.remote(ctx, space="device", dtype=tl.float16, shard_id=peer, offset=offset)
    tl.store(out + offs, tl.load(remote + offs, volatile=True))


@triton.jit
def _write_contig(ctx: tl.constexpr, peer: tl.constexpr, rank: tl.constexpr, offset: tl.constexpr,
                  BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=peer, offset=offset)
    tl.store(remote + offs, offs.to(tl.float32) + rank * 1000.0)


@triton.jit
def _write_masked(ctx: tl.constexpr, peer: tl.constexpr, rank: tl.constexpr, offset: tl.constexpr,
                  BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=peer, offset=offset)
    tl.store(remote + offs, offs.to(tl.float32) + rank * 1000.0, mask=offs < BLOCK // 2)


@triton.jit
def _read_grid(out, ctx: tl.constexpr, peer: tl.constexpr, offset: tl.constexpr, N: tl.constexpr,
               BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=peer, offset=offset)
    tl.store(out + offs, tl.load(remote + offs, mask=mask, other=-1.0, volatile=True), mask=mask)


@triton.jit
def _write_grid(ctx: tl.constexpr, peer: tl.constexpr, rank: tl.constexpr, offset: tl.constexpr,
                N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=peer, offset=offset)
    tl.store(remote + offs, offs.to(tl.float32) + rank * 1000.0, mask=mask)


def _check(name, got, expected, rank):
    if not torch.equal(got, expected):
        bad = (got != expected).nonzero()[:4].flatten().tolist()
        raise AssertionError(f"[rank {rank}] {name} mismatch at {bad}: "
                             f"got[:4]={got[:4].tolist()} expected[:4]={expected[:4].tolist()}")
    print(f"[rank {rank}] {name} PASS", flush=True)


def test_remote_intra_node_matrix():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    tle.init_communicator()
    with torch.cuda.use_mem_pool(tle.get_mem_pool()):
        buf = torch.empty(BUFFER_ELEMS, dtype=torch.float32, device="cuda")
    buf[SMALL_DATA:SMALL_DATA + SMALL_N] = torch.arange(SMALL_N, dtype=torch.float32,
                                                        device="cuda") + rank * 1000
    buf[SMALL_WB:SMALL_WB + SMALL_N] = -1
    buf[LARGE_DATA:LARGE_DATA + LARGE_N] = torch.arange(LARGE_N, dtype=torch.float32,
                                                        device="cuda") + rank * 1000
    buf[LARGE_WB:LARGE_WB + LARGE_N] = -1

    ctx = tle.create_dist_tensor(buf)
    peer = (rank + 1) % WORLD_SIZE
    sender = (rank - 1) % WORLD_SIZE
    peer_small = torch.arange(SMALL_N, dtype=torch.float32, device="cuda") + peer * 1000
    peer_large = torch.arange(LARGE_N, dtype=torch.float32, device="cuda") + peer * 1000

    out = torch.full((SMALL_N, ), -3.0, dtype=torch.float32, device="cuda")
    _read_contig[(1, )](out, ctx, peer, SMALL_DATA, SMALL_N, num_warps=4)
    torch.cuda.synchronize()
    _check("read_contig", out, peer_small, rank)

    _read_contig[(1, )](out, ctx, peer, SMALL_DATA + SMALL_N - 8, 8, num_warps=4)
    torch.cuda.synchronize()
    _check("read_offset", out[:8], peer_small[SMALL_N - 8:], rank)

    _read_strided[(1, )](out, ctx, peer, SMALL_DATA, SMALL_N // 2, num_warps=4)
    torch.cuda.synchronize()
    _check("read_strided", out[:SMALL_N // 2], peer_small[::2], rank)

    _read_masked[(1, )](out, ctx, peer, SMALL_DATA, SMALL_N, num_warps=4)
    torch.cuda.synchronize()
    expected_masked = torch.cat(
        [peer_small[:SMALL_N // 2], torch.full((SMALL_N // 2, ), -7.0, device="cuda")])
    _check("read_masked", out, expected_masked, rank)

    out_i32 = torch.full((SMALL_N, ), -3, dtype=torch.int32, device="cuda")
    _read_int32[(1, )](out_i32, ctx, peer, SMALL_DATA, SMALL_N, num_warps=4)
    torch.cuda.synchronize()
    _check("read_int32", out_i32, peer_small.view(torch.int32), rank)

    out_fp16 = torch.full((SMALL_N, ), -3.0, dtype=torch.float16, device="cuda")
    _read_fp16[(1, )](out_fp16, ctx, peer, SMALL_DATA, SMALL_N, num_warps=4)
    torch.cuda.synchronize()
    _check("read_fp16", out_fp16, peer_small.view(torch.float16)[:SMALL_N], rank)

    dist.barrier()
    _write_contig[(1, )](ctx, peer, rank, SMALL_WB, SMALL_N, num_warps=4)
    torch.cuda.synchronize()
    dist.barrier()
    expected_write = torch.arange(SMALL_N, dtype=torch.float32, device="cuda") + sender * 1000
    _check("write_contig", buf[SMALL_WB:SMALL_WB + SMALL_N], expected_write, rank)

    buf[SMALL_WB:SMALL_WB + SMALL_N] = -1
    dist.barrier()
    _write_masked[(1, )](ctx, peer, rank, SMALL_WB, SMALL_N, num_warps=4)
    torch.cuda.synchronize()
    dist.barrier()
    expected_masked_write = torch.cat(
        [torch.arange(SMALL_N // 2, dtype=torch.float32, device="cuda") + sender * 1000,
         torch.full((SMALL_N // 2, ), -1.0, device="cuda")])
    _check("write_masked", buf[SMALL_WB:SMALL_WB + SMALL_N], expected_masked_write, rank)

    grid = (triton.cdiv(LARGE_N, GRID_BLOCK), )
    out_large = torch.full((LARGE_N, ), -1.0, dtype=torch.float32, device="cuda")
    _read_grid[grid](out_large, ctx, peer, LARGE_DATA, LARGE_N, GRID_BLOCK, num_warps=4)
    torch.cuda.synchronize()
    _check("large_read", out_large, peer_large, rank)

    dist.barrier()
    _write_grid[grid](ctx, peer, rank, LARGE_WB, LARGE_N, GRID_BLOCK, num_warps=4)
    torch.cuda.synchronize()
    dist.barrier()
    expected_large_write = torch.arange(LARGE_N, dtype=torch.float32, device="cuda") + sender * 1000
    _check("large_write", buf[LARGE_WB:LARGE_WB + LARGE_N], expected_large_write, rank)

    tle.cleanup_communicator()
    print(f"[rank {rank}] ALL PASS (N={SMALL_N}, large N={LARGE_N})", flush=True)


if __name__ == "__main__":
    test_remote_intra_node_matrix()
