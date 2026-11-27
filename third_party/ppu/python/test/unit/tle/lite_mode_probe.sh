#!/bin/bash
# Manual lite-mode launch check for the PPU backend.
#
#   PYTHON_BIN=/path/to/venv/bin/python NPROC_PER_NODE=2 \
#       bash lite_mode_probe.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
nproc_per_node="${NPROC_PER_NODE:-2}"
port="${MASTER_PORT:-29671}"

export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-${nproc_per_node}}"
export FLAGTREE_LITE_DIST=1
export FLAGCX_ADAPTOR="${FLAGCX_ADAPTOR:-ppu}"
export FLAGCX_USE_HETERO_COMM="${FLAGCX_USE_HETERO_COMM:-1}"
export FLAGCX_MEM_ENABLE="${FLAGCX_MEM_ENABLE:-1}"
export FLAGCX_VMM_ENABLE="${FLAGCX_VMM_ENABLE:-0}"
export FLAGCX_P2P_TRANSPORT="${FLAGCX_P2P_TRANSPORT:-accl}"
export FLAGCX_P2P_BUFFER_SIZE="${FLAGCX_P2P_BUFFER_SIZE:-1048576}"
export FLAGCX_P2P_CHUNK_SIZE="${FLAGCX_P2P_CHUNK_SIZE:-1048576}"
export FLAGCX_DEBUG="${FLAGCX_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

while ss -ltn | grep -q ":${port} "; do
    port=$((port + 2))
done

exec timeout "${RUN_TIMEOUT:-300}" "${python_bin}" -m torch.distributed.run \
    --nproc_per_node="${nproc_per_node}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${port}" \
    "${script_dir}/lite_mode_probe.py"
