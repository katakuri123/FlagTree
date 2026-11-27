#!/bin/bash
# Multi-rank shard_id check for the PPU backend.
#
#   PYTHON_BIN=/path/to/venv/bin/python NPROC_PER_NODE=2 \
#       bash test_tle_shard_id_distributed.sh
#
# For a real multi-node run, set NNODES/NODE_RANK/MASTER_ADDR and
# LOCAL_WORLD_SIZE on every node.  The P2P buffer defaults below are the
# validated PPU values; FlagCX's 64 MB default fails in the PPU proxy thread.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
nproc_per_node="${NPROC_PER_NODE:-2}"
nnodes="${NNODES:-1}"
node_rank="${NODE_RANK:-0}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
port="${MASTER_PORT:-29640}"

export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-${nproc_per_node}}"
export FLAGCX_ADAPTOR="${FLAGCX_ADAPTOR:-ppu}"
export FLAGCX_USE_HETERO_COMM="${FLAGCX_USE_HETERO_COMM:-1}"
export FLAGCX_MEM_ENABLE="${FLAGCX_MEM_ENABLE:-1}"
export FLAGCX_VMM_ENABLE="${FLAGCX_VMM_ENABLE:-0}"
export FLAGCX_P2P_TRANSPORT="${FLAGCX_P2P_TRANSPORT:-accl}"
export FLAGCX_P2P_BUFFER_SIZE="${FLAGCX_P2P_BUFFER_SIZE:-1048576}"
export FLAGCX_P2P_CHUNK_SIZE="${FLAGCX_P2P_CHUNK_SIZE:-1048576}"
export FLAGCX_DEBUG="${FLAGCX_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

if [ "${nnodes}" -eq 1 ]; then
    while ss -ltn | grep -q ":${port} "; do
        port=$((port + 2))
    done
fi

exec "${python_bin}" -m torch.distributed.run \
    --nproc_per_node="${nproc_per_node}" \
    --nnodes="${nnodes}" \
    --node_rank="${node_rank}" \
    --master_addr="${master_addr}" \
    --master_port="${port}" \
    "${script_dir}/test_tle_shard_id_distributed.py"
