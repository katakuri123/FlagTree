"""Manual lite-mode launch probe: plain kernel + implicit distributed args.

Not pytest-collected (the module sets FLAGTREE_LITE_DIST before importing
triton).  Run through ``lite_mode_probe.sh``.
"""
import os

os.environ["FLAGTREE_LITE_DIST"] = "1"

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _add(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    tle.init_communicator()
    with torch.cuda.use_mem_pool(tle.get_mem_pool()):
        buf = torch.zeros(64, dtype=torch.float32, device="cuda")
    ctx = tle.create_dist_tensor(buf)

    n = 1024
    x = torch.arange(n, dtype=torch.float32, device="cuda")
    y = torch.arange(n, dtype=torch.float32, device="cuda") + 1.0
    out = torch.full((n, ), -1.0, dtype=torch.float32, device="cuda")
    _add[(1, )](x, y, out, n, 1024, num_warps=4)
    torch.cuda.synchronize()
    expected = x + y
    if not torch.allclose(out, expected):
        raise SystemExit(f"[rank {rank}] lite-mode add mismatch")
    print(f"[rank {rank}] lite-mode add PASS", flush=True)
    tle.cleanup_communicator()
    print(f"[rank {rank}] lite-mode cleanup PASS", flush=True)


if __name__ == "__main__":
    main()
