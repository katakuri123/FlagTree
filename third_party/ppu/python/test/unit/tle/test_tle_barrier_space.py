"""Single-node multi-rank runtime coverage for the PPU FlagCX space barrier.

Each rank runs one kernel with a device-side loop: publish a per-round
payload, barrier, read the previous rank's payload, verify it, then barrier
again before the next round.  Host barriers only separate configurations.
"""

import os

import pytest

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "0"))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
if WORLD_SIZE <= 0 or LOCAL_WORLD_SIZE <= 0 or WORLD_SIZE % LOCAL_WORLD_SIZE:
    pytest.skip("torchrun with WORLD_SIZE/LOCAL_WORLD_SIZE is required", allow_module_level=True)
if WORLD_SIZE != LOCAL_WORLD_SIZE:
    pytest.skip("space barrier coverage is validated on a single node; "
                "multi-node world/inter is UNVERIFIED", allow_module_level=True)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import triton.experimental.tle.language as tle  # noqa: E402

BLOCK = 1024
ROUNDS = int(os.environ.get("BARRIER_ROUNDS", "256"))
LONG_ROUNDS = int(os.environ.get("BARRIER_LONG_ROUNDS", "4096"))
LONG_N = int(os.environ.get("BARRIER_LONG_N", "64"))
LARGE_N = int(os.environ.get("BARRIER_LARGE_N", str(1 << 20)))


@triton.jit
def _sync_loop(buf, err, ctx: tl.constexpr, mesh: tl.constexpr, sender: tl.constexpr,
               rounds, n, delay, BLOCK: tl.constexpr, RANK: tl.constexpr,
               GROUP: tl.constexpr, INDEX: tl.constexpr, CONTEXT: tl.constexpr,
               SPACE: tl.constexpr, VOLATILE: tl.constexpr, CACHE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    # Optional rank-dependent delay to exercise different arrival times.  The
    # volatile load is not optimizable, so the loop actually executes.
    for _ in range(delay):
        tl.load(buf, volatile=True)
    for r in range(rounds):
        round_f = r.to(tl.float32)
        for start in range(0, n, BLOCK):
            mask = start + offs < n
            values = start.to(tl.float32) + offs.to(tl.float32) + RANK * 100000.0 + round_f * 1000.0
            tl.store(buf + start + offs, values, mask=mask)

        tle.distributed_barrier(mesh, device_dptr=ctx, space=SPACE, barrier_kind="sync",
                                group_kind=GROUP, index=INDEX, context_id=CONTEXT,
                                order="acqrel", memory_scope="system")

        remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=sender, offset=0)
        error = tl.zeros((), tl.int32)
        for start in range(0, n, BLOCK):
            mask = start + offs < n
            values = tl.load(remote + start + offs, mask=mask, other=0.0, volatile=VOLATILE,
                             cache_modifier=CACHE)
            expected = (start.to(tl.float32) + offs.to(tl.float32) + sender * 100000.0 +
                        round_f * 1000.0)
            error += tl.sum(tl.where(mask, (values != expected).to(tl.int32), 0))
        if error != 0:
            tl.store(err, 1)

        tle.distributed_barrier(mesh, device_dptr=ctx, space=SPACE, barrier_kind="sync",
                                group_kind=GROUP, index=INDEX, context_id=CONTEXT,
                                order="acqrel", memory_scope="system")


@triton.jit
def _arrive_wait_loop(buf, err, ctx: tl.constexpr, mesh: tl.constexpr, sender: tl.constexpr,
                      rounds, n, BLOCK: tl.constexpr, RANK: tl.constexpr,
                      GROUP: tl.constexpr, INDEX: tl.constexpr, CONTEXT: tl.constexpr,
                      SPACE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for r in range(rounds):
        round_f = r.to(tl.float32)
        for start in range(0, n, BLOCK):
            mask = start + offs < n
            values = start.to(tl.float32) + offs.to(tl.float32) + RANK * 100000.0 + round_f * 1000.0
            tl.store(buf + start + offs, values, mask=mask)

        tle.distributed_barrier(mesh, device_dptr=ctx, space=SPACE, barrier_kind="arrive",
                                group_kind=GROUP, index=INDEX, context_id=CONTEXT,
                                order="acqrel", memory_scope="system")
        tle.distributed_barrier(mesh, device_dptr=ctx, space=SPACE, barrier_kind="wait",
                                group_kind=GROUP, index=INDEX, context_id=CONTEXT,
                                order="acqrel", memory_scope="system")

        remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=sender, offset=0)
        error = tl.zeros((), tl.int32)
        for start in range(0, n, BLOCK):
            mask = start + offs < n
            values = tl.load(remote + start + offs, mask=mask, other=0.0, volatile=True)
            expected = (start.to(tl.float32) + offs.to(tl.float32) + sender * 100000.0 +
                        round_f * 1000.0)
            error += tl.sum(tl.where(mask, (values != expected).to(tl.int32), 0))
        if error != 0:
            tl.store(err, 1)

        tle.distributed_barrier(mesh, device_dptr=ctx, space=SPACE, barrier_kind="arrive",
                                group_kind=GROUP, index=INDEX, context_id=CONTEXT,
                                order="acqrel", memory_scope="system")
        tle.distributed_barrier(mesh, device_dptr=ctx, space=SPACE, barrier_kind="wait",
                                group_kind=GROUP, index=INDEX, context_id=CONTEXT,
                                order="acqrel", memory_scope="system")


def test_barrier_space_matrix():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    tle.init_communicator()
    with torch.cuda.use_mem_pool(tle.get_mem_pool()):
        buf = torch.zeros(max(LONG_N, LARGE_N, 256), dtype=torch.float32, device="cuda")
    ctx = tle.create_dist_tensor(buf)
    err = torch.zeros(1, dtype=torch.int32, device="cuda")
    sender = (rank - 1) % WORLD_SIZE

    device_mesh = tle.device_mesh(tle.MeshConfig(device=WORLD_SIZE))
    world_mesh = tle.device_mesh(tle.MeshConfig(node=1, device=WORLD_SIZE))

    def run(name, kernel, mesh, group, index, context, rounds, n, num_warps,
            volatile=True, delay=0, must_pass=True, cache=""):
        dist.barrier()
        err.zero_()
        if kernel is _sync_loop:
            kernel[(1, )](buf, err, ctx, mesh, sender, rounds, n, delay, BLOCK, rank,
                          group, index, context, "world" if mesh is world_mesh else "device",
                          volatile, cache, num_warps=num_warps)
        else:
            kernel[(1, )](buf, err, ctx, mesh, sender, rounds, n, BLOCK, rank,
                          group, index, context, "world" if mesh is world_mesh else "device",
                          num_warps=num_warps)
        torch.cuda.synchronize()
        failed = err.item() != 0
        print(f"[rank {rank}] {name} {'FAIL' if failed else 'PASS'}", flush=True)
        if failed and must_pass:
            raise AssertionError(f"[rank {rank}] {name} payload mismatch")
        dist.barrier()
        return failed

    run("sync_block_nw4", _sync_loop, device_mesh, "block", 0, 0, ROUNDS, 256, 4)
    run("sync_block_nw1", _sync_loop, device_mesh, "block", 0, 0, ROUNDS, 256, 1)
    run("sync_block_nw2", _sync_loop, device_mesh, "block", 0, 0, ROUNDS, 256, 2)
    run("sync_warp_nw1", _sync_loop, device_mesh, "warp", 0, 0, max(16, ROUNDS // 8), 256, 1)
    run("arrive_wait_block", _arrive_wait_loop, device_mesh, "block", 1, 0,
        max(16, ROUNDS // 8), 256, 4)
    run("index1_context1", _sync_loop, device_mesh, "block", 1, 1,
        max(16, ROUNDS // 8), 256, 4)
    run("index7_context3", _sync_loop, device_mesh, "block", 7, 3,
        max(8, ROUNDS // 32), 256, 4)
    run("arrival_delay", _sync_loop, device_mesh, "block", 3, 0, 16, 256, 4,
        delay=rank * 2000 + 1)
    # Compare normal/`.cg`/volatile peer reads after the barrier.  These are
    # observations, not pass criteria; volatile remains the conservative
    # choice for cross-rank polling (see PPU_TLE_ADAPTATION.md 13.6).
    run("normal_load_observation", _sync_loop, device_mesh, "block", 4, 0, 8, 256, 4,
        volatile=False, must_pass=False)
    run("cg_load_observation", _sync_loop, device_mesh, "block", 4, 0, 8, 256, 4,
        volatile=False, cache=".cg", must_pass=False)
    run("normal_long_observation", _sync_loop, device_mesh, "block", 4, 0, LONG_ROUNDS, LONG_N, 4,
        volatile=False, must_pass=False)
    run("long_4096", _sync_loop, device_mesh, "block", 5, 0, LONG_ROUNDS, LONG_N, 4)
    run("large_payload", _sync_loop, device_mesh, "block", 6, 0, 4, LARGE_N, 4)

    # World-space runs use the same communicator.  Re-init is deliberately
    # not exercised here (FlagCX/NCCL bootstrap race); cleanup must be safe to
    # call more than once.
    run("world_block", _sync_loop, world_mesh, "block", 2, 0,
        max(8, ROUNDS // 32), 256, 4)
    run("world_arrive_wait", _arrive_wait_loop, world_mesh, "block", 3, 0,
        max(8, ROUNDS // 32), 256, 4)
    run("world_index1_context1", _sync_loop, world_mesh, "block", 1, 1,
        max(8, ROUNDS // 32), 256, 4)

    tle.cleanup_communicator()
    assert not dist.is_initialized()
    tle.cleanup_communicator()
    assert not dist.is_initialized()
    print(f"[rank {rank}] ALL PASS", flush=True)


if __name__ == "__main__":
    test_barrier_space_matrix()
