"""Device-side ping-pong signal/wait test for the PPU backend.

Each rank runs one kernel for the whole ring: it publishes a per-round
payload, signals the next rank, waits for the previous rank's signal, verifies
the payload, sends an ACK, and waits for the ACK from the next rank.  Host
barriers only separate configurations; no host synchronization participates in
per-round ordering.

Run on one node::

    LOCAL_WORLD_SIZE=2 torchrun --nproc_per_node=2 --nnodes=1 \
        third_party/ppu/python/test/unit/tle/test_tle_signal_pingpong.py
"""

import os

import pytest

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "0"))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
if WORLD_SIZE <= 0 or LOCAL_WORLD_SIZE <= 0 or WORLD_SIZE % LOCAL_WORLD_SIZE:
    pytest.skip("torchrun with WORLD_SIZE/LOCAL_WORLD_SIZE is required", allow_module_level=True)
if WORLD_SIZE != LOCAL_WORLD_SIZE:
    pytest.skip("ping-pong is validated on a single node; multi-node paths are UNVERIFIED",
                allow_module_level=True)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import triton.experimental.tle.language as tle  # noqa: E402

BLOCK = 1024
LONG_ROUNDS = int(os.environ.get("PINGPONG_ROUNDS", "4096"))
LARGE_ROUNDS = int(os.environ.get("PINGPONG_LARGE_ROUNDS", "4"))
STAR_ROUNDS = int(os.environ.get("PINGPONG_STAR_ROUNDS", "128"))


@triton.jit
def _ring_pingpong(buf, err, ctx: tl.constexpr, peer: tl.constexpr, sender: tl.constexpr,
                   rounds, base, N, BLOCK: tl.constexpr, RANK: tl.constexpr,
                   SIGNALS: tl.constexpr, GROUP: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for r in range(rounds):
        round_f = r.to(tl.float32)
        for start in range(0, N, BLOCK):
            mask = start + offs < N
            values = start.to(tl.float32) + offs.to(tl.float32) + RANK * 100000.0 + round_f * 1000.0
            tl.store(buf + start + offs, values, mask=mask)

        tle.signal(ctx, peer, slot_id=0, op="inc", space="intra_node", group_kind=GROUP)
        target = base + (r + 1) * SIGNALS
        tle.signal_wait(ctx, slot_id=0, wait_kind="signal", target=target, group_kind=GROUP,
                        order="acquire")

        remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=sender, offset=0)
        error = tl.zeros((), tl.int32)
        for start in range(0, N, BLOCK):
            mask = start + offs < N
            values = tl.load(remote + start + offs, mask=mask, other=0.0, volatile=True)
            expected = (start.to(tl.float32) + offs.to(tl.float32) + sender * 100000.0 +
                        round_f * 1000.0)
            error += tl.sum(tl.where(mask, (values != expected).to(tl.int32), 0))
        if error != 0:
            tl.store(err, 1)

        tle.signal(ctx, sender, slot_id=1, op="inc", space="intra_node", group_kind=GROUP)
        tle.signal_wait(ctx, slot_id=1, wait_kind="signal", target=target, group_kind=GROUP,
                        order="acquire")


@triton.jit
def _star_consumer(buf, err, ctx: tl.constexpr, rounds, base, N, BLOCK: tl.constexpr,
                   PRODUCERS: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for r in range(rounds):
        round_f = r.to(tl.float32)
        target = base + (r + 1) * PRODUCERS
        tle.signal_wait(ctx, slot_id=0, wait_kind="signal", target=target, group_kind="block",
                        order="acquire")
        for source in tl.static_range(1, PRODUCERS + 1):
            remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=source, offset=0)
            error = tl.zeros((), tl.int32)
            for start in range(0, N, BLOCK):
                mask = start + offs < N
                values = tl.load(remote + start + offs, mask=mask, other=0.0, volatile=True)
                expected = (start.to(tl.float32) + offs.to(tl.float32) + source * 100000.0 +
                            round_f * 1000.0)
                error += tl.sum(tl.where(mask, (values != expected).to(tl.int32), 0))
            if error != 0:
                tl.store(err, 1)
            tle.signal(ctx, source, slot_id=1, op="inc", space="intra_node", group_kind="block")


@triton.jit
def _star_producer(buf, ctx: tl.constexpr, rounds, base, N, BLOCK: tl.constexpr,
                   RANK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for r in range(rounds):
        round_f = r.to(tl.float32)
        for start in range(0, N, BLOCK):
            mask = start + offs < N
            values = start.to(tl.float32) + offs.to(tl.float32) + RANK * 100000.0 + round_f * 1000.0
            tl.store(buf + start + offs, values, mask=mask)
        tle.signal(ctx, 0, slot_id=0, op="inc", space="intra_node", group_kind="block")
        tle.signal_wait(ctx, slot_id=1, wait_kind="signal", target=base + (r + 1),
                        group_kind="block", order="acquire")


def test_signal_pingpong_matrix():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    tle.init_communicator()
    max_n = 1 << 20
    with torch.cuda.use_mem_pool(tle.get_mem_pool()):
        buf = torch.zeros(max_n, dtype=torch.float32, device="cuda")
    ctx = tle.create_dist_tensor(buf)
    err = torch.zeros(1, dtype=torch.int32, device="cuda")
    peer = (rank + 1) % WORLD_SIZE
    sender = (rank - 1) % WORLD_SIZE
    base = 0

    def run_ring(name, n, rounds, group, signals, num_warps):
        nonlocal base
        dist.barrier()
        err.zero_()
        _ring_pingpong[(1, )](buf, err, ctx, peer, sender, rounds, base, n, BLOCK, rank,
                              signals, group, num_warps=num_warps)
        torch.cuda.synchronize()
        if err.item() != 0:
            raise AssertionError(f"[rank {rank}] {name} payload mismatch")
        print(f"[rank {rank}] {name} PASS (rounds={rounds}, N={n}, group={group})", flush=True)
        base += rounds * signals
        dist.barrier()

    run_ring("block_long", 256, LONG_ROUNDS, "block", 1, 4)
    run_ring("block_large", max_n, LARGE_ROUNDS, "block", 1, 4)
    run_ring("warp_ring", 256, 64, "warp", 4, 4)
    run_ring("thread_ring", 256, 64, "thread", 32, 1)

    if WORLD_SIZE > 1:
        dist.barrier()
        err.zero_()
        if rank == 0:
            _star_consumer[(1, )](buf, err, ctx, STAR_ROUNDS, base, 256, BLOCK, WORLD_SIZE - 1,
                                  num_warps=4)
        else:
            _star_producer[(1, )](buf, ctx, STAR_ROUNDS, base, 256, BLOCK, rank, num_warps=4)
        torch.cuda.synchronize()
        if err.item() != 0:
            raise AssertionError(f"[rank {rank}] star payload mismatch")
        print(f"[rank {rank}] star_multi_producer PASS (rounds={STAR_ROUNDS})", flush=True)
        dist.barrier()

    tle.cleanup_communicator()
    print(f"[rank {rank}] ALL PASS (world={WORLD_SIZE})", flush=True)


if __name__ == "__main__":
    test_signal_pingpong_matrix()
