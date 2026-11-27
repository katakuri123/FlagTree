TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Raw
    "raw.call",
    "raw.call_smem",

    ## TLE-Lite
    "shard_id",
    "remote",
    "signal",
    "signal_wait",
    "load",
    "extract_tile",
    "insert_tile",
    "cumsum",

    ## TLE-Struct GPU
    "gpu.buffered_tensor.slot",
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.warp_specialize",
]
