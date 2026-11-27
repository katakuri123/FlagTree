"""Multi-rank signal/wait coverage for the PPU backend.

Producer and consumer run in separate kernels (the producer writes its
registered buffer and signals in the same kernel; the consumer waits with
acquire order and then reads the producer's buffer through tle.remote).
Host barriers only separate scenarios; within a scenario the wait is resolved
on the device.

Run on one node::

    LOCAL_WORLD_SIZE=2 torchrun --nproc_per_node=2 --nnodes=1 \
        third_party/ppu/python/test/unit/tle/test_tle_signal_wait.py

``test_tle_signal_wait.sh`` applies the validated PPU FlagCX defaults and a
wall-clock timeout for deadlock detection.
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

SMALL_N = int(os.environ.get("SIGNAL_N", "4096"))
LARGE_N = int(os.environ.get("SIGNAL_LARGE_N", str(1 << 20)))
ROUNDS = int(os.environ.get("SIGNAL_ROUNDS", "256"))
BLOCK = 1024


@triton.jit
def _produce_inc(buf, data_value: tl.constexpr, ctx: tl.constexpr, peer: tl.constexpr,
                 slot: tl.constexpr, unused: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(buf + offs, tl.full((BLOCK, ), data_value, tl.float32))
    tle.signal(ctx, peer, slot_id=slot, op="inc", space="intra_node", group_kind="block")


@triton.jit
def _produce_add(buf, data_value: tl.constexpr, ctx: tl.constexpr, peer: tl.constexpr,
                 slot: tl.constexpr, value: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(buf + offs, tl.full((BLOCK, ), data_value, tl.float32))
    tle.signal(ctx, peer, slot_id=slot, value=value, op="add", space="intra_node", group_kind="block")


@triton.jit
def _produce_world_inc(buf, data_value: tl.constexpr, ctx: tl.constexpr, peer: tl.constexpr,
                       slot: tl.constexpr, unused: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(buf + offs, tl.full((BLOCK, ), data_value, tl.float32))
    tle.signal(ctx, peer, slot_id=slot, op="inc", space="world", group_kind="block")


@triton.jit
def _produce_to_rank(buf, data_value: tl.constexpr, ctx: tl.constexpr, target_rank,
                     slot: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(buf + offs, tl.full((BLOCK, ), data_value, tl.float32))
    tle.signal(ctx, target_rank, slot_id=slot, op="inc", space="intra_node", group_kind="block")


@triton.jit
def _wait_read(out, ctx: tl.constexpr, sender: tl.constexpr, slot: tl.constexpr,
               target, BLOCK: tl.constexpr):
    tle.signal_wait(ctx, slot_id=slot, wait_kind="signal", target=target, group_kind="block",
                    order="acquire")
    offs = tl.arange(0, BLOCK)
    remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=sender, offset=0)
    tl.store(out + offs, tl.load(remote + offs))


@triton.jit
def _wait_read_multi(out, ctx: tl.constexpr, slot: tl.constexpr, target, BLOCK: tl.constexpr,
                     WORLD: tl.constexpr):
    tle.signal_wait(ctx, slot_id=slot, wait_kind="signal", target=target, group_kind="block",
                    order="acquire")
    offs = tl.arange(0, BLOCK)
    for peer in tl.static_range(1, WORLD):
        remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=peer, offset=0)
        tl.store(out + peer * BLOCK + offs, tl.load(remote + offs))


@triton.jit
def _produce_large(buf, data_value: tl.constexpr, ctx: tl.constexpr, peer: tl.constexpr,
                   slot: tl.constexpr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for start in range(0, N, BLOCK):
        mask = start + offs < N
        values = start.to(tl.float32) + offs.to(tl.float32) + data_value
        tl.store(buf + start + offs, values, mask=mask)
    tle.signal(ctx, peer, slot_id=slot, op="inc", space="intra_node", group_kind="block")


@triton.jit
def _wait_read_large(out, ctx: tl.constexpr, sender: tl.constexpr, slot: tl.constexpr,
                     target, N, BLOCK: tl.constexpr):
    tle.signal_wait(ctx, slot_id=slot, wait_kind="signal", target=target, group_kind="block",
                    order="acquire")
    offs = tl.arange(0, BLOCK)
    remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=sender, offset=0)
    for start in range(0, N, BLOCK):
        mask = start + offs < N
        values = tl.load(remote + start + offs, mask=mask, other=-1.0)
        tl.store(out + start + offs, values, mask=mask)


@triton.jit
def _produce_stress(buf, data_value: tl.constexpr, ctx: tl.constexpr, peer: tl.constexpr,
                    slot: tl.constexpr, rounds, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(buf + offs, tl.full((BLOCK, ), data_value, tl.float32))
    for _ in range(rounds):
        tle.signal(ctx, peer, slot_id=slot, op="inc", space="intra_node", group_kind="block")


@triton.jit
def _wait_read_stress(out, ctx: tl.constexpr, sender: tl.constexpr, slot: tl.constexpr,
                      target, BLOCK: tl.constexpr):
    tle.signal_wait(ctx, slot_id=slot, wait_kind="signal", target=target, group_kind="block",
                    order="acquire")
    offs = tl.arange(0, BLOCK)
    remote = tle.remote(ctx, space="device", dtype=tl.float32, shard_id=sender, offset=0)
    tl.store(out + offs, tl.load(remote + offs))


def _check(name, got, expected, rank):
    if not torch.equal(got, expected):
        bad = (got != expected).nonzero()[:4].flatten().tolist()
        raise AssertionError(f"[rank {rank}] {name} mismatch at {bad}: "
                             f"got[:4]={got[:4].tolist()} expected[:4]={expected[:4].tolist()}")
    print(f"[rank {rank}] {name} PASS", flush=True)


def test_signal_wait_matrix():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    tle.init_communicator()
    with torch.cuda.use_mem_pool(tle.get_mem_pool()):
        buf = torch.zeros(LARGE_N, dtype=torch.float32, device="cuda")
    ctx = tle.create_dist_tensor(buf)
    peer = (rank + 1) % WORLD_SIZE
    sender = (rank - 1) % WORLD_SIZE

    out = torch.full((SMALL_N, ), -1.0, dtype=torch.float32, device="cuda")

    def phase(name, producer, slot, target, value):
        data_value = rank * 1000 + value
        dist.barrier()
        producer[(1, )](buf, data_value, ctx, peer, slot, value, SMALL_N, num_warps=4)
        _wait_read[(1, )](out, ctx, sender, slot, target, SMALL_N, num_warps=4)
        torch.cuda.synchronize()
        expected = torch.full((SMALL_N, ), float(sender * 1000 + value), dtype=torch.float32,
                              device="cuda")
        _check(name, out, expected, rank)
        dist.barrier()

    phase("inc_1", _produce_inc, 0, 1, 1)
    phase("inc_2", _produce_inc, 0, 2, 2)
    phase("inc_3", _produce_inc, 0, 3, 3)
    phase("add_3", _produce_add, 1, 3, 4)
    phase("world_inc", _produce_world_inc, 2, 1, 5)

    if WORLD_SIZE > 1:
        dist.barrier()
        out_multi = torch.full((WORLD_SIZE * SMALL_N, ), -1.0, dtype=torch.float32, device="cuda")
        if rank == 0:
            _wait_read_multi[(1, )](out_multi, ctx, 3, WORLD_SIZE - 1, SMALL_N, WORLD_SIZE,
                                    num_warps=4)
        else:
            _produce_to_rank[(1, )](buf, rank * 1000 + 6, ctx, 0, 3, SMALL_N, num_warps=4)
        torch.cuda.synchronize()
        if rank == 0:
            for source in range(1, WORLD_SIZE):
                expected = torch.full((SMALL_N, ), float(source * 1000 + 6), dtype=torch.float32,
                                      device="cuda")
                _check(f"multi_producer_{source}",
                       out_multi[source * SMALL_N:(source + 1) * SMALL_N], expected, rank)
        dist.barrier()
        print(f"[rank {rank}] multi_producer PASS", flush=True)

    dist.barrier()
    out_large = torch.full((LARGE_N, ), -1.0, dtype=torch.float32, device="cuda")
    _produce_large[(1, )](buf, rank * 1000 + 7, ctx, peer, 4, LARGE_N, BLOCK, num_warps=4)
    _wait_read_large[(1, )](out_large, ctx, sender, 4, 1, LARGE_N, BLOCK, num_warps=4)
    torch.cuda.synchronize()
    expected_large = torch.arange(LARGE_N, dtype=torch.float32, device="cuda") + sender * 1000 + 7
    _check("large_payload", out_large, expected_large, rank)
    dist.barrier()

    dist.barrier()
    _produce_stress[(1, )](buf, rank * 1000 + 8, ctx, peer, 5, ROUNDS, SMALL_N, num_warps=4)
    _wait_read_stress[(1, )](out, ctx, sender, 5, ROUNDS, SMALL_N, num_warps=4)
    torch.cuda.synchronize()
    expected_stress = torch.full((SMALL_N, ), float(sender * 1000 + 8), dtype=torch.float32,
                                 device="cuda")
    _check("stress", out, expected_stress, rank)
    dist.barrier()

    tle.cleanup_communicator()
    print(f"[rank {rank}] ALL PASS (world={WORLD_SIZE}, rounds={ROUNDS})", flush=True)


if __name__ == "__main__":
    test_signal_wait_matrix()
