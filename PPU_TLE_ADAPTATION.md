# FlagTree PPU TLE-Lite 适配进度

本文件由 Phase 1 起维护，记录基准状态、设计决策、代码改动、测试命令与结果。后续会话应以此为接续依据。

## 0. 基准

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/flagos-ai/FlagTree`（本地 `/data/wenxuekun/FlagTree`） |
| 分支 | `main` |
| HEAD | `864c4bebde8f74219847510f56e63aa86e6f9a9b` |
| 工作区 | clean（基线构建前） |
| PPU Backend | `third_party/ppu/` |
| Triton 基线 | 3.6.0 |
| FlagCX pin（全局） | `c93438decaba8d935e7aa6ae8377e56fd3910b32`（仅 Host 支持，无 PPU Device Bitcode） |
| 构建 LLVM | `cmake/llvm-hash.txt` = `f6ded0be897e2878612dd903f7e8bb85448269e5` |

## 1. Phase 1.1：PPU 编译与运行基线

### 1.1 环境

| 项 | 值 |
|---|---|
| Python | 3.12.3 |
| 构建 venv | `/data/wenxuekun/flagtree-phase1/venv-build`（`include-system-site-packages=true`，用于运行期访问 torch） |
| PPU SDK | `/usr/local/PPU_SDK`，2.1.0，LLVM 13.0.1 |
| `ppu-llc` | `/usr/local/PPU_SDK/bin/ppu-llc` |
| 设备 | 16 × PPU-ZW810E（cap 8.0） |
| PyTorch | 2.10.0（系统包） |
| 构建工具 | GCC 13.3.0, CMake 3.26.0, Ninja 1.10/1.13 |
| 系统 Triton | 厂商包 `3.6.0+v0.1.0.ppu2.1.0.oe`（未修改） |

### 1.2 构建命令

```bash
TRITON_PTXAS_PATH=/data/wenxuekun/FlagTree/third_party/nvidia/backend/bin/ptxas \
TRITON_PTXAS_BLACKWELL_PATH=/data/wenxuekun/FlagTree/third_party/nvidia/backend/bin/ptxas-blackwell \
TRITON_CUOBJDUMP_PATH=/data/wenxuekun/FlagTree/third_party/nvidia/backend/bin/cuobjdump \
TRITON_NVDISASM_PATH=/data/wenxuekun/FlagTree/third_party/nvidia/backend/bin/nvdisasm \
TRITON_CUDACRT_PATH=/usr/local/PPU_SDK/CUDA_SDK/include \
TRITON_CUDART_PATH=/usr/local/PPU_SDK/CUDA_SDK/include \
TRITON_CUPTI_INCLUDE_PATH=/usr/local/PPU_SDK/CUDA_SDK/include \
TRITON_CUPTI_LIB_PATH=/usr/local/PPU_SDK/CUDA_SDK/lib64 \
FLAGTREE_BACKEND=ppu MAX_JOBS=64 \
  /data/wenxuekun/flagtree-phase1/venv-build/bin/python -m pip install -e . --no-build-isolation -v
```

说明：NVIDIA 工具环境变量用于跳过与 PPU 构建无关、且在本机网络不可达的 CUDA redist 下载；普通环境不需要这些变量。LLVM 1.23 GB 压缩包自动下载至 `~/.triton/llvm/llvm-f6ded0be-ubuntu-x64`。

### 1.3 构建结果

- 退出码 0；`flagtree-0.7.0+ppu.git864c4beb` 可编辑安装成功。
- `libtriton.so`：`python/triton/_C/libtriton.so`（907 MB，assert 构建）。
- `import triton`（加 `PYTHONPATH=/data/wenxuekun/FlagTree/python`）指向仓库构建；`FLAGTREE_BACKEND='ppu'`；`triton._C.libtriton.tle` 可用；`triton.backends` 仅路由 `ppu`。
- 系统厂商 Triton 未被修改（`python3 -c "import triton"` 仍指向 `/usr/local/lib/python3.12/site-packages/triton`）。
- 工作区 `git status` 仍为 clean；构建产生的 `python/triton/backends/ppu` 等为 gitignored 符号链接。

### 1.4 测试结果

测试统一环境：

```bash
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK
export PYTHONPATH=/data/wenxuekun/FlagTree/python
V=/data/wenxuekun/flagtree-phase1/venv-build/bin/python
```

| 测试项 | 命令 | 结果 |
|---|---|---|
| TLE smoke + tile | `$V -m pytest -q third_party/ppu/python/test/unit/tle/test_tle_smoke.py third_party/ppu/python/test/unit/tle/test_tle_tile_ops.py` | PASS：10 passed（357.46s，首次冷编译较慢） |
| Local Pointer | `$V -m pytest -q .../test_tle_gpu_local_ptr.py` | PASS：11 passed, 1 xfailed（已知 cp.async <4B 问题） |
| Async Load | `$V -m pytest -q .../test_tle_aiu_async_load.py` | PASS：37 passed |
| GEMM + Local Store | `$V -m pytest -q .../test_tle_gemm.py .../test_tle_local_store.py` | PASS：7 passed |
| Cumsum（运行期） | `$V -m pytest -q python/test/tle/unit/test_tle_cumsum.py --first-parameter-only` | PASS：3 passed, 2 skipped, 8 deselected |
| Vector Add（运行期） | `$V /data/wenxuekun/flagtree-phase1/tests/smoke_baseline.py` | PASS：`vector_add PASS` |
| Row Reduction（运行期） | 同上 | PASS：`row_reduce PASS` |

### 1.5 Phase 1.1 结论

PASS。FlagTree PPU 基线可构建、可编译、可运行；本地 TLE 功能无回归。基线构建未引入 FlagCX 或通信代码改动。

## 2. Phase 1.2：FlagCX 依赖与构建系统适配

### 2.1 FlagCX pin 决策

| 项 | 值 |
|---|---|
| 全局 `flagcx` pin（不变） | `c93438decaba8d935e7aa6ae8377e56fd3910b32` |
| PPU 专用 pin（新增） | `4879b88dafb631cef9361cc7eae5b0449d5b63ac`（FlagCX #741，含 `bindings/ir/ppu` Device IR 与 cann-shmem Device API backend） |
| 本地路径 | `third_party/ppu/third_party/flagcx` |
| 注册名 | `flagcx_ppu`（`python/setup_tools/utils/__init__.py`） |

选择“后端本地独立 checkout”而非升级全局 pin：新版 FlagCX 改动了 symmetric window/VMM 行为、Python wrapper 路径和若干 host API，直接替换会同时影响 NVIDIA、Iluvatar、HCU 已验证的通信路径；回移 Device IR 提交又需要修改 FlagCX 构建系统，范围不可控。独立 pin 的代价仅是 PPU 构建多下载一份源码。

影响分析：

- `python/setup_tools/setup_helper.py` 的 `FLAGCX_SUPPORT_BACKENDS` 仍为 `{"nvidia", "iluvatar"}`，共享 `flagcx` 的下载/构建逻辑对 PPU 不生效。
- `cmake/FlagTreeOptions.cmake` 的 `FLAGCX_SUPPORT_BACKENDS` 增加 `ppu`，只用于给 PPU 构建定义 `FLAGCX_ENABLED`，从而编译/链接共享的 `FlagCxToLLVM`。
- `enable_flagtree_third_party` 将 `flagcx_ppu` 归一化为 `USE_FLAGCX`，`USE_FLAGCX=OFF` 可同时禁用两份依赖。

### 2.2 PPU FlagCX Registrar

实现：`python/setup_tools/utils/ppu.py`（`PPUFlagCXRegistrar`）。

| 步骤 | 命令/行为 |
|---|---|
| SDK 发现 | `PPU_SDK`/`PPU_HOME`，默认 `/usr/local/PPU_SDK`；校验 `bin/hgcc` |
| JSON 依赖 | 使用 FlagCX `third-party/json` 子模块；支持 `JSON_INCLUDE_DIR` 覆盖 |
| Host Library | `make -j USE_PPU=1 USE_ACCL_BAREX=1 USE_UCX=0 COMPILE_KERNEL=1 DEVICE_HOME=<sdk> CCL_HOME=<sdk>` |
| Device Bitcode | `make -C bindings/ir/ppu BITCODE_LIB_ARCH=ppu001 BUILDDIR=... headers check_public_symbols <bc>` |
| 架构 | `FLAGCX_PPU_ARCH`，默认 `ppu001` |
| 符号校验 | `nm -D --defined-only` 检查 `flagcxDevNetLaunchConstruct`、`flagcxDevNetSizeOf` |
| 缓存 | `.flagcx_ppu_build_stamp` 记录 commit/arch/flags；`FLAGCX_RECOMPILE`/`DIST_RECOMPILE` 强制重建 |
| 安装 | `python/triton/_C/libflagcx.so`、`third_party/ppu/backend/lib/{libflagcx_device.bc,metadata.json}`、language/backend 两份 wrapper 与 18 个头文件 |
| 包装 | `get_package_data()` 声明上述非 Python 产物；已静态验证 4 组 glob 均命中实际文件 |

已知构建特例：

1. Host 构建不能传 `DEVICE_BITCODE_ARCH`：`makefiles/ppu.mk` 会把它转发给 `hgcc -arch`，而 hgcc 只接受 `vm_*/ppu_10/15/17/all`；bitcode 构建单独使用 `--ppu-arch=ppu001`。
2. FlagCX 的 `check_ir` 目标期望 triple `alippu`，本机 SDK 2.1.0 生成 `ppu`。Registrar 直接请求 `headers`、`check_public_symbols` 和最终 `.bc` 目标，产物 triple 与 `libdevice.ppu.bc` 一致（`target triple = "ppu"`）。

### 2.3 验收结果

| 项 | 结果 |
|---|---|
| 正式构建流程产出 `libflagcx.so`（38 MB） | PASS |
| 正式构建流程产出 `libflagcx_device.bc`（32 MB，LLVM 13，triple `ppu`） | PASS |
| Host Library 含 Device API 符号 | PASS |
| 产物路径与编译器/运行时发现逻辑一致 | PASS |
| 无临时手工复制文件；缓存可复现 | PASS |

## 3. Phase 1.3：编译器公共基础设施适配

### 3.1 Extern Library 与 Bitcode 链接

- 新增 `third_party/ppu/backend/distributed.py::FlagcxRuntimeConfig`，按 `FLAGCX_BITCODE_PATH` → `FLAGCX_MODULE_PATH` → backend `lib/` → `~/.flagtree/flagcx` 顺序发现 bitcode；`Distributed().get_extern_libs()` 返回 `{"libflagcx": <bc>}`。
- `HGGCOptions.__post_init__` 将该库并入 `extern_libs`；普通 kernel 仍只拿到 libdevice + 该条目。
- `make_llir` 链接前用 `ppu.need_extern_lib(llvm_mod, "flagcx")` 过滤：只有 IR 中仍存在未定义的 flagcx 外部符号时才链接 32 MB bitcode。`triton_ppu.cc` 新增对应 C++ 实现。实验确认 `get_device_id` 调用被内联后仍保留 `inttoptr (i64 ...)` 与 FlagCX 类型元数据，说明 bitcode 确实参与链接与优化。

### 3.2 地址空间参数化

- PPU datalayout：`e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32`；FlagCX PPU bitcode 导出 generic AS0 指针。
- 共享 lowering 原先把通信上下文指针硬编码为 AS1，PPU `ppu-llc` 报 `Error : callee should have been handled before caller`；改为 AS0 后通过。
- 处理方式：`FlagcxUtils.h/.cpp` 与 `FlagCxOpToLLVM` 各 population 函数增加 `unsigned ptrAddressSpace = 1`；PPU 注册时显式传 `0`，NVIDIA/HCU 使用默认值 `1`，语义不变。
- 未引入任何无依据的 `addrspacecast`；`inttoptr` 的目标地址空间与函数声明、bitcode 定义一致。

### 3.3 Conversion Pattern 注册

- `third_party/ppu/lib/TritonPPUGPUToLLVM/CMakeLists.txt`：`FLAGCX_ENABLED` 时链接 `FlagCxToLLVM`。
- `TritonPPUGPUToLLVM.cpp`：TLE 预转换阶段注册 `populateSignalOpToLLVMPatterns`、`populateFlagCxOpToLLVMPatterns(ptrAddressSpace=0)`；主转换阶段注册 `populateRemotePointersOpToLLVMPatterns`、`populateNodeTransferOpToLLVMPatterns`；`RemotePointers/NodePut/NodeGet` 在预转换 target 中保持 legal。
- `make_ttgir` 增加 `add_fuse_node_remote_transfers`（所有 TLE kernel）与 lite-mode 的 `add_params_for_distribution`，并输出与 NVIDIA 同名的 `tle_node_buffer_bindings` metadata。
- `tle.distributed_barrier`、WGMMA、TMA 仍未注册 Lowering；PPU 白名单移除陈旧的 `distributed_barrier`，前端会提前给出明确错误，而不是在编译深处失败。

### 3.4 验收结果

| 项 | 结果 |
|---|---|
| 公共通信 IR 可识别并 lower 到 LLVM | PASS（compile probe） |
| 设备函数调用符合 PPU ABI（AS0） | PASS |
| LLVM IR 无 AS 不匹配转换 | PASS |
| Device Bitcode 成功链接 | PASS |
| ppu-llc 目标代码生成 | PASS |
| NVIDIA/HCU lowering 行为保持 | NOT RUN（静态核对：默认参数 1，未改其调用点；本机无对应硬件） |

## 4. Phase 1.4：分布式 Runtime 基础设施

### 4.1 模块与发现

`third_party/ppu/backend/distributed.py` 提供：

- `FlagcxRuntimeConfig`：发现 `libflagcx_device.bc`、`libflagcx.so`、`include/`、`flagcx_wrapper.py`；支持 `FLAGCX_MODULE_PATH`/`FLAGCX_BITCODE_PATH`/`FLAGCX_LIB_PATH`/`FLAGCX_INCLUDE_PATH`；bundle 不完整时抛错而不是静默降级。
- `USE_FLAGCX`（以及 `USE_DIST` 等）关闭时不启用。
- `DistributedRtContext`、`init_communicator`、`create_dist_tensor`、`cleanup_communicator`、`get_mem_pool` 复用 `python/triton/experimental/tle/language/communication.py`，未新增 PPU 专属上下文结构或硬编码结构体偏移。
- Driver 隐式参数沿用公共 `DistributedRtContext._get_needed_params()`（`device_comm_ptr`、`device_mem_ptr`）与 `add_args_to_jitfunction`；PPU 未定义第二套 ABI。

### 4.2 Window/VMM/IPC 冲突隔离

| 观察 | 结论 |
|---|---|
| `FLAGCX_WIN_COLL_SYMMETRIC` 要求 `FLAGCX_VMM_ENABLE=1` | FlagCX 语义 |
| PPU 上 VMM signal buffer 无法完成 IPC 导出 | 底层限制（SDK 2.1.0） |
| asymmetric window（flags=0）+ `FLAGCX_VMM_ENABLE=0` 可完成节点内 IPC | 已验证 |
| TLE 节点内原语只需要 peer pointer，不依赖 symmetric 集合窗口 | 满足 Phase 1/2/3 需求 |

隔离方式：`FlagcxRuntimeConfig.window_flags` 默认 `0`（`FLAGCX_PPU_WINDOW_FLAGS` 可覆盖），并在 PPU 进程内 `os.environ.setdefault("FLAGCX_VMM_ENABLE", "0")`；`communication.py` 通过 `getattr(flagcx_rt_conf, "window_flags", FLAGCX_WIN_COLL_SYMMETRIC)` 取值。NVIDIA/Iluvatar/HCU 的默认 symmetric 行为不变，未全局关闭 VMM。

### 4.3 已知运行期配置

FlagCX 默认 `FLAGCX_P2P_BUFFER_SIZE=64MB` 在 PPU proxy 线程 `hggcMalloc` 失败，导致 `flagcxDevCommCreate → preconnectFullMesh` 返回错误 1。双卡验证使用：

```bash
FLAGCX_P2P_BUFFER_SIZE=1048576 FLAGCX_P2P_CHUNK_SIZE=1048576
```

该限制属于 FlagCX/PPU 分配器行为，Phase 1 不硬编码到 Runtime；后续需与 FlagCX 确认默认值或按平台能力配置。

### 4.4 验收结果

| 项 | 结果 |
|---|---|
| 单 Rank 上下文创建/销毁 | PASS |
| 双进程双卡上下文初始化 | PASS（见 5.3，需上述 P2P buffer 配置） |
| Driver 隐式参数传递 | PASS（rank probe 读取到正确 rank） |
| Window/资源释放 | PASS（`cleanup_communicator` 正常返回） |
| 普通非通信 Kernel 不受影响 | PASS（回归测试） |
| Lite-mode | 编译 pass 已接入；运行期未执行（`FLAGTREE_LITE_DIST=1` + 公共 API 属 Phase 2） |

## 5. Phase 1.5：最小端到端验证

测试脚本：`/data/wenxuekun/flagtree-phase1/tests/`。

### 5.1 Level 1：编译验证

命令：

```bash
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK
export PYTHONPATH=/data/wenxuekun/FlagTree/python
$V /data/wenxuekun/flagtree-phase1/tests/flagcx_compile_probe.py
```

`flagcx_compile_probe.py` 构造 `DistributedRtContext(0x1000, 0x2000)`，编译调用私有 `_get_local_rank` 的 kernel，断言：TTGIR 含 `get_device_id`/`get local rank`；LLIR 含 AS0 `inttoptr (i64 ...)`；LLIR 含 `PpuPlatform`/`Barrier` 元数据（证明 FlagCX bitcode 已链接）。结果：**PASS**。

### 5.2 Level 2：单卡执行验证

命令：`torchrun --nproc_per_node=1 ... flagcx_rank_probe.py`。结果：

```text
[rank 0] local_rank=0 world=1
[rank 0] PASS
```

调用的是真实 `flagcxDevCommGetIntraRank`（经 `_get_local_rank` lowering），返回值写入设备内存并被 Host 读回；不是空通信 kernel。**PASS**。

### 5.3 Level 3：双卡执行验证

命令：

```bash
FLAGCX_ADAPTOR=ppu FLAGCX_USE_HETERO_COMM=1 FLAGCX_MEM_ENABLE=1 FLAGCX_VMM_ENABLE=0 \
FLAGCX_P2P_TRANSPORT=accl FLAGCX_P2P_BUFFER_SIZE=1048576 FLAGCX_P2P_CHUNK_SIZE=1048576 \
torchrun --nproc_per_node=2 ... flagcx_rank_probe.py
```

结果：

```text
[rank 1] local_rank=1 world=2
[rank 0] local_rank=0 world=2
[rank 1] PASS
[rank 0] PASS
```

重复执行一次仍 PASS。**PASS**。

失败分类记录：首次双卡尝试在 `flagcxDevCommCreate` 失败。逐层定位为 FlagCX `flagcxP2pAllocateShareableBuffer` 的默认 64 MB `deviceMalloc` 在 PPU proxy 线程失败，属于 FlagCX/PPU 分配器配置问题，不是编译器、Device API ABI 或 P2P 网络环境问题；缩小 P2P buffer 后通过。

环境限制：本轮未使用 `CUDA_VISIBLE_DEVICES` 强制单进程单卡（该方式在本机 PPU 运行时导致 segfault），双卡通过 `torchrun` 默认设备映射完成。

## 6. Phase 1.6：回归测试与代码审查

### 6.1 测试结果

| 测试项 | 命令/范围 | 结果 |
|---|---|---|
| PPU TLE 回归 | `pytest -q third_party/ppu/python/test/unit/tle/{test_tle_smoke,test_tle_tile_ops,test_tle_gpu_local_ptr,test_tle_aiu_async_load,test_tle_gemm,test_tle_local_store}.py` | **65 passed, 1 xfailed**（0 failed；最终冷编译 370.61s，热缓存复跑 5.28s） |
| 普通 Triton Kernel | `smoke_baseline.py`（vector add + row reduction） | PASS |
| FlagCX Host Library | Registrar 构建 + `nm` 符号检查 | PASS |
| Device Bitcode | 构建 + 链接 + `ppu-llc` | PASS |
| MLIR/LLVM Conversion | `flagcx_compile_probe.py` | PASS |
| 单卡 Runtime | `flagcx_rank_probe.py` | PASS |
| 双卡 Runtime | 同脚本 2 进程 | PASS |
| 其他后端公共 IR | 未在 NVIDIA/HCU 实机执行 | NOT RUN（依赖沙箱无对应硬件） |

`test_ppu_rejects_distributed_barrier_with_clear_error` 的断言已从“编译期报错包含 `tle.distributed_barrier`”调整为“错误信息同时包含 `distributed_barrier` 与 `ppu`”。原因是移除陈旧白名单后，前端在编译前即抛出 `TLEUnsupportedPrimitiveError`，这是更早、更明确的拒绝路径，测试意图不变。

### 6.2 代码审查结论

| 检查项 | 结论 |
|---|---|
| 不必要的公共代码修改 | 仅 `FlagcxUtils`/`FlagCxOpToLLVM` 的 AS 参数化、`communication.py` 的窗口/allocator 配置、setup/cmake 接入；默认值均保持旧行为 |
| 重复 PPU 专属实现 | 无；复用共享 FlagCx lowering、DistributedRtContext、communication.py |
| 错误改变 NVIDIA/HCU 行为 | 未发现；AS 默认 1、window 默认 symmetric、`with_cuda` 默认 True |
| 提前开放未支持原语 | 未开放；反而移除了陈旧的 `distributed_barrier` 白名单项 |
| 硬编码本机路径 | 无；仅可配置默认值（`PPU_SDK` 默认 `/usr/local/PPU_SDK`、bitcode 架构默认 `ppu001`） |
| 浮动依赖 | 无；`flagcx_ppu` 固定 `4879b88...` |
| 临时调试代码 | 无遗留；新增 log 均为构建/运行时诊断信息 |
| 空/伪实现/静默降级 | 无；bundle 不完整时抛错，FlagCX 关闭时明确禁用 |
| 不必要的性能开销 | 普通 kernel 不链接 FlagCX bitcode（`need_extern_lib` 过滤） |
| Git 操作 | 未提交、未推送、未创建 PR；改动按逻辑分组保留在工作区 |

### 6.3 Phase 1 十二项验收回答

| # | 问题 | 状态 | 依据 |
|---|---|---|---|
| 1 | FlagTree PPU 是否实际构建成功 | PASS | `pip install -e .` 退出 0；版本 `flagtree-0.7.0+ppu.git864c4beb` |
| 2 | 普通 Triton 与已有本地 TLE Kernel 回归 | PASS | 65 passed / 1 xfailed；vector add、reduction PASS |
| 3 | FlagCX 最终 commit 及原因 | PASS | `4879b88dafb631cef9361cc7eae5b0449d5b63ac`，后端本地独立 pin，全局 pin 不变 |
| 4 | Device Bitcode 能否由正式构建流程获得 | PASS | Registrar 构建 `libflagcx_device.bc`（32 MB，triple `ppu`） |
| 5 | AS0/AS1 差异如何通过公共接口处理 | PASS | `ptrAddressSpace` 参数化，PPU=0，其他默认 1 |
| 6 | Conversion Pattern 注册机制是否建立 | PASS | PPU CMake + `TritonGPUToLLVM.cpp` 注册共享 FlagCx/Signal/NodeTransfer patterns |
| 7 | DistributedRtContext 创建/销毁 | PASS | 单 Rank、双 Rank probe 均正常初始化与清理 |
| 8 | 最小设备函数调用是否实际执行成功 | PASS | `flagcxDevCommGetIntraRank` 返回 0/1 并写回 Host |
| 9 | 双卡 Runtime 是否实际验证 | PASS（带 P2P buffer 配置） | 两 Rank 均 PASS，重复执行 PASS |
| 10 | Window/VMM/IPC 冲突是否解决或隔离 | PASS（已隔离） | PPU 专用 asymmetric window + backend-local VMM 默认，其他后端不变 |
| 11 | 是否仍有阻塞 shard_id 的问题 | PASS（无已知阻塞） | Rank 查询 lowering/bitcode/runtime 全链路已验证；公共 API 属 Phase 2 |
| 12 | 是否引入其他后端回归风险 | NOT RUN（静态评估风险低） | 共享改动均保留旧默认值；本机无 NVIDIA/HCU 硬件，实机回归未执行 |

### 6.4 Phase 2 结论

**Go**。公共通信基础设施已打通：编译、链接、AS0 ABI、单卡/双卡设备函数调用均实际验证。进入 Phase 2 前需知悉：双卡验证依赖 `FLAGCX_P2P_BUFFER_SIZE=1048576`；多节点未验证；grid barrier、BAREX 远端 signal atomic 仍为后续专项。

## 7. 已知问题与后续任务

1. FlagCX `check_ir` 的 `alippu`/`ppu` triple 不一致需上游修复；当前 Registrar 绕过该目标并校验实际产物。
2. FlagCX 默认 64 MB P2P buffer 在 PPU 分配失败；需 FlagCX 或 SDK 侧确认推荐配置。
3. `tle.distributed_barrier` 的 PPU Lowering（Phase 5）与 BAREX 远端 signal atomic（Phase 4 前置）未实现。
4. 跨节点 `tle.remote`（Phase 6）依赖 FlagCX 网络路径，当前无多节点环境。
5. Lite-mode 的 `add_params_for_distribution` 已接入编译链，但未做运行期验证。
6. `FLAGCX_MODULE_PATH` 预装 bundle 场景仍要求 `triton/experimental/tle/language/flagcx_wrapper.py` 已存在（Python import 路径固定），bundling 文档需补充。
7. 非 editable wheel 的端到端安装未执行；已静态验证 `get_package_data()` 的 4 组 glob 均命中产物。

## 8. Phase 2：tle.shard_id

### 8.1 目标与参考

实现 `tle.shard_id` 的 PPU 公共 API，语义与 NVIDIA/Iluvatar 一致。参考：

- `ac89e408c [TLE] Support node axis in shard_id (#969)`（node 轴：`world_rank // n_pes`）。
- 原始引入提交 `3ab21a0c3 [TLE] Add initial distributed APIs (#395)`。
- 当前实现：`python/triton/experimental/tle/language/distributed.py::shard_id`。

### 8.2 调用链

```text
tle.shard_id(mesh, axis, device_dptr=ctx)
├─ axis == "device"/"chiplet" → _get_local_rank
│    → builder.get_device_id → tle.get_device_id
│    → GetDeviceIdToFlagCX.cpp → tle.get_local_rank
│    → getLocalPeFuncCall → flagcxDevCommGetIntraRank
├─ axis == "node" → _get_world_rank + n_pes
│    → builder.get_world_rank / get_n_pes
│    → tle.get_world_rank / tle.get_num_pes
│    → getWorldRankFuncCall → flagcxDevCommGetRank
│    → getNumPesFunCall → flagcxDevCommGetIntraSize
│    → node = world_rank // intra_size
└─ mesh axis (block/cluster) → program_id(0) 线性化，无 FlagCX 调用
```

PPU 注册点：`third_party/ppu/lib/TritonPPUGPUToLLVM/TritonGPUToLLVM.cpp` 的
`populateFlagCxOpToLLVMPatterns(..., /*ptrAddressSpace=*/0)`，覆盖
`GetDeviceId/GetLocalRank/GetNumPes/GetWorldRank`；Device Bitcode 按需链接。

### 8.3 代码修改

| 文件 | 修改内容 | 原因 |
|---|---|---|
| `third_party/ppu/backend/tle_supported.py` | 白名单新增 `"shard_id"` | 打开已验证的公共原语 |
| `third_party/ppu/python/test/unit/tle/test_tle_shard_id.py` | 单进程测试：白名单、device/node 编译 lowering、mesh 轴运行、非法轴 | PPU 专属回归 |
| `third_party/ppu/python/test/unit/tle/test_tle_shard_id_distributed.py` | 多 Rank 运行测试（device/node 轴可观测校验） | 双卡/多卡验证 |
| `third_party/ppu/python/test/unit/tle/test_tle_shard_id_distributed.sh` | torchrun 启动脚本（含 PPU P2P buffer 默认值） | 可复现运行 |

未新增 PPU 专属拓扑抽象或 lowering；Node/Device 轴全部复用共享 FlagCX 实现。

### 8.4 测试结果

| 测试项 | 命令/范围 | 结果 |
|---|---|---|
| 编译探针（mesh 轴） | `shard_id_compile_probe.py` | PASS |
| 编译探针（device/node 轴） | 同上；TTGIR 含 `get_device_id/get_world_rank/get_num_pes`，LLIR 含 AS0 `inttoptr` + `PpuPlatform` | PASS |
| 新增单进程测试 | `pytest -q third_party/ppu/python/test/unit/tle/test_tle_shard_id.py` | 8 passed（含编译期非法轴） |
| 通用 mesh 轴运行测试 | `pytest -q python/test/tle/unit/test_tle_shard_id_runtime.py` | 4 passed（4 种 shape） |
| 通用前端/非法轴测试 | `pytest -q python/test/tle/unit/test_tle_distributed.py` | 37 passed, 8 skipped |
| 单节点双卡 | `NPROC_PER_NODE=2 test_tle_shard_id_distributed.sh` | PASS：world=0/1，intra=2，device=0/1，node=0/0 |
| 单节点四卡 | `NPROC_PER_NODE=4 test_tle_shard_id_distributed.sh` | PASS：world=0/1/2/3，intra=4，device=0/1/2/3，node=0 |
| Node 轴（真实多节点，node>0） | 需要第二台主机 | **BLOCKED** |
| PPU TLE 全量回归 | `pytest -q third_party/ppu/python/test/unit/tle/` | 145 passed, 1 skipped, 1 xfailed |
| 普通 Triton 基线 | vector add + row reduction | PASS |

观测结果示例（双卡）：

```text
[rank 1] world=1 intra=2 device=1 node=0 (expected 1/0)
[rank 0] world=0 intra=2 device=0 node=0 (expected 0/0)
```

world rank 与 intra size 同时校验，确保 node=0 不是由“全 0”假象通过；device 轴与 `LOCAL_RANK` 的映射、node 轴 `world // intra` 的公共公式均得到确认。

### 8.5 结论与遗留

- `tle.shard_id` 的 PPU 编译、lowering、Device Bitcode 链接与运行链路已打通；device 轴与 node 轴（单节点 node=0）均返回与逻辑拓扑一致的值。
- 多节点 `node>0` 的 Rank 映射未验证；环境不具备第二台主机，标记 BLOCKED，不据此宣称跨节点能力完成。
- `chiplet` 轴在共享前端映射为 intra-node rank；PPU 无 chiplet 拓扑，本阶段未将其作为验收项。
- 双卡/四卡仍依赖 `FLAGCX_P2P_BUFFER_SIZE=1048576` / `FLAGCX_P2P_CHUNK_SIZE=1048576`；`FLAGCX_P2P_DISABLE=1` 未能绕过 `preconnectFullMesh` 失败。
- 下一阶段：Phase 3 `tle.remote(intra_node)`，等待确认。

## 9. Phase 3：tle.remote(intra_node)

### 9.1 目标与参考

实现节点内远端读写：`tle.remote(..., space="device")`。参考：

- `c5a928da9 [TLE] Add optional offset parameter to tle.remote op (#732)`（device space 的 offset 语义）。
- 当前共享 Lowering：`third_party/tle/dialect/lib/Conversion/TleToLLVM/LocalPointersOpToLLVM.cpp::lowerDeviceSpace`。
- FlagCX Device IR：`libflagcx_device.bc` 中的 `flagcxGetIntraPointerC(i8*, i64, i32) -> i8*`。

### 9.2 调用链

```text
tle.remote(ctx, space="device", dtype=T, shard_id=peer, offset=off)
→ _remote_pointer → _create_remote_pointers_tensor
→ builder.create_remote_pointers(resultTy, mem_ptr, shard_id, "device", offset)
→ tle.remote_pointers (shard_id i32, offset i64)
→ RemotePointersOpConversion → lowerDeviceSpace
   → inttoptr(mem_ptr i64) → AS0
   → flagcxGetIntraPointerC(comm_mem, offset*elem_bytes, peer) → AS0 i8*
   → addrspacecast AS0 → AS1
→ tt.load / tt.store on peer global memory
```

`mem_ptr` 来自 `DistributedRtContext[0]`（`create_dist_tensor` 的 DevMem 指针）；
device bitcode 提供 `flagcxGetIntraPointerC` 定义，编译期完成链接/内联。

### 9.3 地址空间依据

- PPU kernel 的全局指针为 AS1（kernel 参数与 `ppu.st.global` 均为 `ptr addrspace(1)`）。
- FlagCX Device C ABI 返回 generic AS0 指针；共享 lowering 在调用边界执行一次
  `addrspacecast ptr to ptr addrspace(1)`。
- 该 cast 是有 ABI 依据的边界转换，不是无依据替换；ppu-llc 接受该 cast，双卡运行期已验证 load/store 正确。

### 9.4 代码修改

| 文件 | 修改内容 | 原因 |
|---|---|---|
| `third_party/ppu/backend/tle_supported.py` | 白名单新增 `"remote"` | 打开已验证的 intra-node 路径 |
| `third_party/ppu/backend/compiler.py` | `PPU_UNSUPPORTED_TLE_OPS` 移除整体拒绝；`reject_unsupported_tle` 按 `space` 判断，仅放行 `device` | Phase 3 只开放 intra-node；node/cluster/chiplet 继续显式拒绝 |
| `third_party/ppu/python/test/unit/tle/test_tle_remote_compile.py` | 编译测试：lowering/ABI、offset 必填、node space 拒绝 | 编译级回归 |
| `third_party/ppu/python/test/unit/tle/test_tle_remote_intra_node.py` | 多 Rank 读写矩阵（含 1M 元素） | 运行级回归 |
| `third_party/ppu/python/test/unit/tle/test_tle_remote_intra_node.sh` | torchrun 启动与 PPU FlagCX 环境 | 可复现 |

未修改共享 Lowering、FlagCX pin 或 Device Bitcode。

### 9.5 测试结果

| 测试项 | 命令/范围 | 结果 |
|---|---|---|
| 编译 lowering/ABI | `pytest -q .../test_tle_remote_compile.py` | **4 passed** |
| node space 拒绝 | 同上（`tle.remote(space='node')` → `NotImplementedError`） | **PASS** |
| 通用 D2D 远端读 | `torchrun --nproc_per_node=2 python/test/tle/unit/test_tle_distributed_d2d.py` | **PASS**（两 Rank 读到对端数据） |
| 双卡读写矩阵 | `NPROC_PER_NODE=2 test_tle_remote_intra_node.sh` | **PASS**（N=4096，large N=1M） |
| 四卡读写矩阵 | `NPROC_PER_NODE=4 REMOTE_N=1024 test_tle_remote_intra_node.sh` | **PASS**（ring 邻近 Rank） |
| 八卡读写矩阵 | `NPROC_PER_NODE=8 REMOTE_N=1024 REMOTE_LARGE_N=65536 ...` | **PASS**（8 Rank ring） |
| 不同 offset | `offset=N-8, BLOCK=8` | PASS |
| 非连续访问 | `remote + offs*2` | PASS |
| Masked load/store | 半区间掩码，未写区域保持哨兵值 | PASS |
| 不同 dtype | fp32、int32、fp16（按元素宽度寻址） | PASS |
| 大数据量 | 1,048,576 × fp32（4 MB/方向），grid=1024；另测 1,048,579（尾部 mask） | PASS |
| PPU TLE 全量回归 | `pytest -q third_party/ppu/python/test/unit/tle/` | **150 passed, 2 skipped, 1 xfailed** |
| 普通 Triton 基线 | vector add + row reduction | PASS |

### 9.6 结论与遗留

- `space="device"` 的远端读、写、offset、strided、masked、多 dtype、多 Rank 与大数据量均实际执行通过。
- `space="node"` 在 PPU 编译期被显式拒绝，保持 Phase 6 前不开放；`cluster`/`chiplet` 同样拒绝。
- 读写正确性验证使用 Host 侧 `torch.cuda.synchronize()` + `dist.barrier()` 分隔 producer/consumer；设备侧 signal/wait 的内存顺序与可见性属 Phase 4，本阶段不宣称。
- 单个 communicator 一次注册一块 buffer 的路径已验证；多 buffer 交替注册未覆盖。
- 双卡/四卡仍依赖 `FLAGCX_P2P_BUFFER_SIZE=1048576` / `FLAGCX_P2P_CHUNK_SIZE=1048576`。
- 下一阶段：Phase 4 `tle.signal_wait`，等待确认。

## 10. Phase 4 前置验证：tle.signal AS0 lowering 与双卡可见性

本步骤只做验证，不修改 PPU 白名单、不回写代码；探针通过进程内临时扩展
`TLE_SUPPORTED_PRIMITIVES` 来编译 `signal`/`signal_wait`。

### 10.1 编译与 ABI

| 项 | 结果 |
|---|---|
| TTGIR | 含 `tle.signal`、`tle.signal_wait` |
| LLIR 符号 | `flagcxDevSignalInc`、`flagcxDevSignalAdd`、`flagcxDevWaitSignal` 均完成链接/内联 |
| 地址空间 | comm 指针为 AS0 `inttoptr`；无 `addrspacecast` |
| 原子指令 | 内联 FlagCX 代码使用 `ppu.atom.add.release.sys.u64` / `acq_rel.sys.u64`，操作 generic `ptr` |
| ppu-llc | `hgbin` 生成成功 |

命令：`/data/wenxuekun/flagtree-phase1/tests/signal_compile_probe.py`（PASS）。

### 10.2 双卡可见性

探针：producer 在同一 kernel 内写本地注册 buffer 后执行 `tle.signal`；
consumer 执行 `tle.signal_wait(order="acquire")` 后通过
`tle.remote(space="device")` 读取 sender 数据。Host 仅在 phase 之间做栅栏，
phase 内 producer/consumer 由设备侧 wait 同步。

| 场景 | 结果 |
|---|---|
| `inc` 累计 1/2/3 次，wait target 1/2/3 | **PASS**（2 Rank） |
| `add value=3`，wait target 3 | **PASS**（2 Rank） |
| 4 Rank ring（signal 下一个 Rank，wait 上一个 Rank） | **PASS** |
| 2 Rank 重复 3 次 | **PASS** |

观测：每轮 consumer 读到的都是 sender 本轮写入的数据（`rank*1000+value`），
证明 signal 送达与 release/acquire 可见性成立。

### 10.3 结论与限制

- `tle.signal` 的 AS0 lowering、设备侧原子与双卡可见性已实际验证，可作为 Phase 4 的基础。
- 仅验证 `wait_kind="signal"`；`counter`/`shadow` 未执行。
- 仅验证 `space="intra_node"`、`group_kind="block"`；inter_node/world、thread/warp 未执行。
- `signal`/`signal_wait` 仍未加入 PPU 白名单；Phase 4 完成完整语义与测试后再决定开放。
- 双卡仍依赖 `FLAGCX_P2P_BUFFER_SIZE=1048576` / `FLAGCX_P2P_CHUNK_SIZE=1048576`。
- 下一步：Phase 4 `tle.signal_wait` 完整实现与测试，等待确认。

## 11. Phase 4：tle.signal_wait

### 11.1 目标与参考

实现并开放 `tle.signal` / `tle.signal_wait`，验证信号生产、等待、内存顺序与数据可见性。参考：

- `d1a0467b0 [TLE] Add signal and signal_wait operators (#863)`。
- 共享 Lowering：`SignalOpToLLVM.cpp`（`tle.signal` → `tle.flagcx signal`）、
  `FlagCxSignalOpToLLVM.cpp`、`FlagcxUtils.cpp::getSignalFuncCall/getDevNetWaitFuncCallByKind`。
- 前置验证：本文件第 10 节（AS0 lowering + 双卡可见性）。

### 11.2 调用链与语义

```text
tle.signal(ctx, peer, slot, op=inc/add, space, group_kind, context_id, scope)
→ tle.signal → tle.flagcx signal
→ flagcxDevSignalInc / flagcxDevSignalAdd
→ inlined ppu.atom.add.release.sys / acq_rel.sys（generic AS0 ptr）

tle.signal_wait(ctx, slot, wait_kind, target, group_kind, context_id, order)
→ tle.signal_wait → tle.flagcx signal_wait
→ flagcxDevWaitSignal / WaitCounter / WaitSignalMeetShadow
→ 本地槽位轮询（CompletionStorage::waitBefore + spinBackoff），
   signal 语义为等待 slot >= target（bits=64）。
```

`wait_kind` 映射：

| wait_kind | Device API | target | 备注 |
|---|---|---|---|
| `signal` | `flagcxDevWaitSignal` | 必填 | bits=64，本阶段运行验证 |
| `counter` | `flagcxDevWaitCounter` | 必填 | bits=56；无公共 counter 生产端，仅编译验证 |
| `shadow` | `flagcxDevWaitSignalMeetShadow` | 禁止 | 依赖 shadow 维护路径，仅编译验证 |

### 11.3 代码修改

| 文件 | 修改内容 | 原因 |
|---|---|---|
| `third_party/ppu/backend/tle_supported.py` | 白名单新增 `"signal"`、`"signal_wait"` | 前置验证通过后开放 |
| `third_party/ppu/python/test/unit/tle/test_tle_signal_compile.py` | 编译测试：全部 wait kind、space、group、scope | 编译级回归 |
| `third_party/ppu/python/test/unit/tle/test_tle_signal_wait.py` | 多 Rank 运行矩阵 | 运行级回归 |
| `third_party/ppu/python/test/unit/tle/test_tle_signal_wait.sh` | torchrun + PPU FlagCX 环境 + `RUN_TIMEOUT` | 可复现与死锁超时 |

未修改共享 Lowering、FlagCX pin 或 Device Bitcode。

### 11.4 测试结果

| 测试项 | 命令/范围 | 结果 |
|---|---|---|
| 编译：AS0/全部 wait kind | `pytest -q .../test_tle_signal_compile.py` | **PASS**：2 passed |
| AS0 证据 | LLIR | `inttoptr` AS0、无 `addrspacecast`；`ppu.atom.add.release.sys`/`acq_rel.sys` |
| 双卡 inc 1/2/3 + add 3 + world inc | `NPROC_PER_NODE=2 test_tle_signal_wait.sh` | **PASS** |
| 双卡多生产者（3 个 Rank 信号 Rank 0） | 同上 | **PASS**（Rank 0 等待 2 个信号并读取全部生产者数据） |
| 双卡 1M 元素 payload | 同上 | **PASS** |
| 双卡 256 轮 / 4096 轮连续 signal | 同上（`SIGNAL_ROUNDS=4096`） | **PASS** |
| 四卡矩阵 | `NPROC_PER_NODE=4` | **PASS** |
| 八卡矩阵 | `CUDA_VISIBLE_DEVICES=1,3,4,5,6,7,8,9 NPROC_PER_NODE=8` | **PASS** |
| PPU TLE 全量回归 | `pytest -q third_party/ppu/python/test/unit/tle/` | **152 passed, 3 skipped, 1 xfailed** |
| 普通 Triton 基线 | vector add + row reduction | PASS |

说明：首次 8 卡尝试因设备 0/2 被外部负载占用（96 GB/94 GB）触发 OOM；改用空闲设备后通过，属环境问题，不是代码缺陷。

### 11.5 结论与遗留

- `signal`/`signal_wait` 的 AS0 lowering、System scope 原子、release/acquire 可见性、多轮/多生产者/大数据量均实际验证。
- `wait_kind="counter"`/`"shadow"` 仅编译验证；无公共 counter 生产端，shadow 依赖内部维护路径，运行期未执行。
- `group_kind="block"` 运行验证；`thread`/`warp` 仅编译验证。
- `space="intra_node"`/`"world"` 单节点验证；`inter_node` 需多节点，仅编译验证。
- 多消费者同一槽位受“一进程一设备”限制，未验证。
- 死锁检测由 `test_tle_signal_wait.sh` 的 `RUN_TIMEOUT`（默认 600s）提供；未执行“故意不发送信号”的负向挂起测试，避免影响共享设备。
- 双卡/多卡仍依赖 `FLAGCX_P2P_BUFFER_SIZE=1048576` / `FLAGCX_P2P_CHUNK_SIZE=1048576`。
- 下一阶段：Phase 5 `tle.distributed_barrier`，等待确认。

## 12. Phase 4.5：通信语义收敛与工程加固

### 12.1 审查发现核实结果

| # | 审查发现 | 核实结果 | 状态 |
|---|---|---|---|
| 1 | block/warp 组 signal 的数据可见性未覆盖 | **确认**：per-element ping-pong 在 block/warp 组失败，thread 组通过；根因是 FlagCX collective signal 只由 leader thread/lane 发 release atomic，其他线程的 payload store 未被 system fence 覆盖；consumer 侧普通 `tl.load` 也会命中旧缓存 | **已修复/已补测** |
| 2 | stress 测试不是真正的每轮发布/消费 | **确认**：原 `_produce_stress` 只写一次数据再连续 signal；新增设备侧 ping-pong，每轮独立 payload + ACK + 校验 | **已修复/已补测** |
| 3 | Runtime 环境配置在模块导入时生效 | **确认**：`FlagcxRuntimeConfig.__init__` 曾 `setdefault(FLAGCX_VMM_ENABLE)`，compile-only 导入有副作用 | **已修复** |
| 4 | Bundle 版本/ABI 一致性不足 | **确认**：metadata 缺 hashes/triple/SDK；`file_hash` 被 `lru_cache(path)` 缓存，文件原地替换后 compile cache 不失效 | **已修复** |
| 5 | 64MB P2P buffer 是否硬限制 | **不成立为固定限制**：空闲设备上默认 64MB/16MB 在 2/4/8 卡均稳定通过；设备 0/2 被外部负载占用时失败；chunk=1MB+buffer=64MB 可复现分配失败 | **已定位为环境/配置相关** |
| 6 | Lite-mode 是否可用 | PPU driver 缺隐式参数插入且 `PyArg` format 在建签名前生成；共享 JIT launch 使用新建空 `DistributedRtContext`（指针为 None） | **PPU plumbing 已修复；共享 JIT 空指针为遗留** |
| 7 | Grid barrier 的 scratch/PTX 依赖 | 共享 grid lowering 使用 `NVVM` IDs、inline PTX `atom.add.release.gpu`/`ld.acquire.gpu`，并依赖 `ttg.global_scratch_memory_offset`；TLE 管线当前无 pass 设置该 attr | **Phase 5 重点** |
| 8 | BAREX 远端 signal atomic 缺口 | `barex_adaptor.cc` 明确 `iputSignal = NULL`（BAREX 无 remote atomic primitive）；one-sided put/get 存在 | **Phase 6 阻塞项（源码确认）** |

### 12.2 已修复问题

1. **Collective signal/wait 的系统可见性**
   - `FlagcxUtils.cpp` 新增 `emitPpuSystemFence`：PPU 对 block/warp 组的 `flagcxDevSignal*` 前、`flagcxDevWait*` 后，为每个线程插入 `ppu.fence.sc.sys`。
   - `FlagCxSignalOpToLLVM.cpp` 将 `collectiveFence` 传入 signal/wait lowering；PPU 注册时传 `true`，NVIDIA/HCU 默认 `false`，不改变其行为。
   - thread 组不插入额外 fence（每线程自身 release/acquire 已覆盖）。
   - 验证：同一 kernel 内 store→signal、wait→remote load，block/warp/thread 均通过。

2. **Peer 轮询读取必须使用 volatile load**
   - 远端内存被对端在 kernel 运行期间更新时，普通 `tl.load` 会命中本地缓存；测试中的 remote load 全部改为 `volatile=True`。
   - 这是 TLE peer-polling 的使用约束，需在公共文档/后续原语中明确；Phase 5 的 barrier 后数据读取同样需要。

3. **真正的设备侧 Ping-Pong**
   - 新增 `test_tle_signal_pingpong.py/.sh`：ring 内每轮写不同 payload、signal、wait、校验、ACK、等 ACK；Host 只在配置之间同步。
   - 覆盖 block/warp/thread、1M payload、4096 轮、多生产者 star。

4. **Runtime 配置延迟与 VMM/Window**
   - `FlagcxRuntimeConfig.__init__` 不再修改环境；新增 `apply_runtime_env()`，由 `communication.py::init_communicator()` 在 `flagcxCommInitRank` 前调用。
   - 用户显式 `FLAGCX_VMM_ENABLE` 优先；显式 `=1` 给出警告；`FLAGCX_PPU_WINDOW_FLAGS=1`（symmetric）明确报错。
   - compile-only 导入不再有环境副作用；单进程重复初始化仍受共享 `_init_communicator_` latch 限制（见 12.5）。

5. **Bundle ABI 一致性**
   - Registrar metadata 增加 `metadata_version`、`target_triple`、`target_datalayout`、`sdk_version`、`build_flags`、host/bitcode/headers/wrapper SHA-256。
   - Runtime `validate_bundle()` 在通信初始化前校验 hashes、triple、arch；多 metadata 文件 commit/hash 不一致时报错；缺失 metadata 警告。
   - `file_hash` 改为按 `(realpath, mtime_ns, size, inode)` 缓存，原地重建 bitcode 会使 `HGGCOptions.hash()` 变化并失效 compile cache。
   - 测试覆盖 hash mismatch、triple 错误、缺失 metadata、混合 bundle root、extern lib 内容替换。

6. **Lite-mode PPU plumbing**
   - `third_party/ppu/backend/driver.py` 在 lite 模式插入两个 `*i64` 隐式参数，并在插入之后生成 `PyArg_ParseTuple` format（修复 20 vs 18 参数错误）。
   - `HGGCOptions.lite_dist` 参与 compile cache key；`make_ttgir` 使用该字段。
   - `lite_mode_probe.py/.sh` 验证 plain kernel 的隐式参数 ABI/launch 链路（2 卡）。

### 12.3 新增及修改的测试

| 文件 | 内容 |
|---|---|
| `third_party/ppu/python/test/unit/tle/test_tle_signal_pingpong.py/.sh` | 设备侧 ring/star ping-pong；block/warp/thread、1M payload、4096 轮、多生产者、timeout |
| `third_party/ppu/python/test/unit/tle/test_tle_signal_wait.py` | 增加 block nw1/nw2、warp nw4、thread nw1；remote load 改 volatile；单节点 guard |
| `third_party/ppu/python/test/unit/tle/test_tle_signal_compile.py` | 增加 collective fence 断言（block/warp 有、thread 无） |
| `third_party/ppu/python/test/unit/tle/test_tle_runtime_config.py` | VMM 默认/显式/非法组合、bundle hash/triple/缺失/混合 root、extern lib cache key |
| `third_party/ppu/python/test/unit/tle/test_tle_remote_compile.py` / `test_tle_remote_intra_node.py` | remote load 改 volatile |
| `third_party/ppu/python/test/unit/tle/lite_mode_probe.py/.sh` | Lite-mode 最小 launch 验证（非 pytest 收集） |
| `third_party/ppu/backend/compiler.py`、`driver.py`、`distributed.py` | 上述修复 |

### 12.4 实际测试结果

| 测试项 | 命令 | 结果 |
|---|---|---|
| Clean build | `rm -rf build && pip install -e . --no-build-isolation`（`FLAGTREE_BACKEND=ppu`） | **PASS**：退出 0，`flagtree-0.7.0+ppu.gita09986a8` |
| FlagCX pin 一致性 | 全局 `c93438d...`；PPU `4879b88...`；metadata 同 commit | **PASS** |
| Signal 分组矩阵 | `NPROC_PER_NODE=2 test_tle_signal_wait.sh` | **PASS**：block nw1/nw2/nw4、warp nw4、thread nw1、add/world、多生产者、1M、stress |
| Ping-pong 2 卡 | `NPROC_PER_NODE=2 test_tle_signal_pingpong.sh`（4096 轮 + 1M + warp/thread + star） | **PASS** |
| Ping-pong 4 卡 | `NPROC_PER_NODE=4 PINGPONG_ROUNDS=1024 ...` | **PASS**（4/4 Rank） |
| Ping-pong 8 卡 | `NPROC_PER_NODE=8 PINGPONG_ROUNDS=512 ...` | **PASS**（8/8 Rank） |
| Runtime config / bundle | `pytest -q test_tle_runtime_config.py` | **9 passed** |
| Signal compile | `pytest -q test_tle_signal_compile.py` | **4 passed** |
| Lite-mode | `NPROC_PER_NODE=2 lite_mode_probe.sh` | **PASS**（plain kernel ABI；TLE op in lite mode 未验证） |
| shard_id 双卡 | `NPROC_PER_NODE=2 test_tle_shard_id_distributed.sh` | **PASS**（world/device/node 校验） |
| remote intra_node 双卡 | `NPROC_PER_NODE=2 test_tle_remote_intra_node.sh` | **PASS**（N=4096，large N=1M） |
| P2P 默认 64MB/16MB | 2/4/8 卡，空闲设备，3 次重复 | **PASS** |
| P2P 1MB/1MB | 2 卡，空闲设备，3 次重复 | **PASS** |
| P2P chunk=1MB + buffer=64MB | 2 卡 | **FAIL**（后续 `flagcxMemAlloc` 8MB 失败；配置/分配器相关） |
| 设备 0/2 高占用 | 8 卡首次尝试 | **FAIL/OOM**（外部负载占 96GB/94GB；环境问题） |
| PPU TLE 全量回归 | `pytest -q third_party/ppu/python/test/unit/tle/`（clean build 后） | **163 passed, 4 skipped, 1 xfailed** |
| 普通 Triton 基线 | vector add + row reduction | **PASS** |
| Package data（静态） | `get_package_data` 5 组 glob | **PASS**（5/5 命中） |

### 12.5 尚未解决问题及优先级

| 优先级 | 问题 | 说明 |
|---|---|---|
| P0 | Grid barrier scratch 管线 | `DistributedBarrierOpToLLVM` 读取 `ttg.global_scratch_memory_offset`，但当前 TLE pipeline 无 pass 设置；PPU grid 实现需补分配/初始化 |
| P0 | Grid barrier 的 inline PTX | 共享 lowering 使用 `atom.add.release.gpu`/`ld.acquire.gpu`；PPU 需 `ppu.atom.*.sys`/`ppu.ld.acquire.sys` |
| P0 | 多节点 inter_node | 无第二台主机；`space="inter"`/inter-node signal/put/get 运行期未验证 |
| P1 | 共享 Lite-mode JIT 空指针 | `DistributedRtContext()` 新实例的 `comm_ptr`/`mem_ptr` 为 None；TLE op 的 lite-mode 运行期仍不可用 |
| P1 | counter/shadow wait | 无公共 counter 生产端；shadow 依赖内部维护，仅编译验证 |
| P1 | 多消费者同槽位 | 一进程一设备，无法构造同槽位多消费者 |
| P2 | chunk=1MB/buffer=64MB 分配失败 | FlagCX/PPU GDR 分配器碎片/池限制，需上游定位 |
| P2 | wheel 端到端打包 | 仅静态验证 package_data；未构建 wheel |

### 12.6 Phase 5 详细实现方案（不本阶段实现）

**5.1 FlagCX 跨设备 barrier（优先）**
- 前端：`tle.distributed_barrier(space="device"/"inter"/"world", ...)` 已映射到 `tle.distributed_barrier` 的 `space` 属性。
- 编译器：
  - `reject_unsupported_tle` 对 `tle.distributed_barrier` 改为 space-aware：允许 `device`/`inter`/`world`；`grid`/`grid_axis`/`submesh`/cluster 继续显式拒绝，直到各自 lowering 验证。
  - PPU `TritonGPUToLLVM.cpp` 注册 `populateDistributedBarrierOpToLLVMPatterns`；`lowerFlagCxSpaceBarrier` 生成 `tle.flagcx barrier`，由已注册的 `populateFlagCxBarrierOpToLLVMPatterns(ptrAddressSpace=0)` 降到 `flagcxDevBarrierSync/Arrive/Wait`。
- 验证：
  - 编译级断言 `flagcxDevBarrierSync`/AS0/无 PTX；`space="device"` 2 卡 producer/consumer + barrier + remote read；`space="world"` 单节点；`space="inter"` 多节点 BLOCKED。
  - 检查 barrier 的 team/参与 Rank 一致性；`barrier_count`/`context_id` 与 `create_dist_tensor` 的 `barrierCount=8` 匹配。

**5.2 PPU Grid barrier（第二阶段）**
- 复用算法：global scratch 上 sense-reversing 计数器（高 bit flip），每 CTA thread 0 到达，其他线程 CTA barrier。
- PPU 原子/加载：用 inline asm `ppu.atom.add.release.sys.u64`、`ppu.ld.acquire.sys.b64`、`ppu.fence.sc.sys`；避免 NVIDIA PTXBuilder。
- Scratch：
  - 在 TLE TTGIR 阶段新增/复用 pass，为每个 grid/grid_axis barrier 分配 global scratch 并设置 `ttg.global_scratch_memory_offset`；
  - 确认 PPU `add_allocate_global_scratch_memory`、driver `global_scratch` 参数、metadata `global_scratch_size` 一致；
  - 确认 scratch 在 launch 前清零；若不清零，需在 kernel 首次使用前初始化或使用 generation。
- 启动/驻留：
  - PPU driver 已支持 `HG_LAUNCH_ATTRIBUTE_COOPERATIVE`；grid barrier 要求所有参与 CTA 并发驻留。
  - 通过 `hggcDevAttrCooperativeLaunch` + `hggcOccupancyMaxActiveBlocksPerMultiprocessor` 校验 grid size；不满足时给出明确错误而不是死锁。
  - `num_ctas>1`/cluster 与 grid barrier 的组合需显式拒绝或验证。
- 测试：1/2/4 CTA、不同 num_warps、重复 barrier 代际、barrier 前后远端数据、超时检测、grid > residency 的错误路径。
- 风险：TLE scratch 管线缺失、PPU cooperative launch 限制、grid barrier 与 cluster 混用。

**5.3 Cluster/submesh**
- 共享 lowering 使用 `NVVM::ClusterArrive/Wait` 与 submesh shared scratch；PPU cluster 语义未验证。
- Phase 5 对 `group_kind="submesh"`/cluster-only mesh 显式拒绝，待 PPU cluster 能力确认后再开放。

### 12.7 Phase 6 底层能力及阻塞条件

源码证据：

- `barex_adaptor.cc` 明确 `iputSignal = NULL`，注释“BAREX has no remote atomic primitive”；one-sided `iput`/`iget` 存在。
- Device bitcode 提供 `flagcxDevNetPutS`/`GetS`/`GetFromCommS` 及 `PutS_RSigInc/RSigAdd`；TLE NodeTransfer lowering 使用无 RSig 的 `flagcxDevNetPutS/GetS`。
- `default_comm_traits.h` 的 signal 在 direct peer 指针 + `useP2pSignals` 时走 direct atomic，否则 `enqueueFifoSignalAction`（proxy）；跨节点无 direct peer 指针，依赖 FIFO/proxy。
- PPU 构建 `USE_ACCL_BAREX=1, USE_UCX=0`，网络适配器为 BAREX/ACCL；跨节点需要真实 BAREX 网络与 MR 注册。

Phase 6 Go/No-Go 前置条件：

1. 至少 2 台物理 PPU 主机，BAREX/ACCL 网络互通；可创建多节点 `torch.distributed` 与 FlagCX inter communicator。
2. 验证 BAREX one-sided PUT/GET 与 `flagcxDevNet*` 设备端完成机制（net context、MR、fence）。
3. 确认 inter-node signal 的 proxy/FIFO 路径可用；若依赖 `iputSignal`（BAREX 为 NULL），则 **No-Go**，需 FlagCX/BAREX 提供远端 signal 或经证实的代理实现。
4. 验证 asymmetric window + VMM=0 在多节点注册/窗口生命周期下可用；跨节点不能假设远端指针可解引用。
5. 明确 multi-node 环境获取时间；在此之前 Phase 6 只能做主机侧/编译侧预研，不能标记完成。

### 12.8 Phase 5 Go/No-Go 结论

- **5.1 FlagCX space barrier：Go**。可直接复用现有 FlagCxBarrier lowering、AS0 参数化与双卡通信上下文；风险低。
- **5.2 PPU grid barrier：Conditional Go**。算法与 PPU 原子能力具备，但必须先补齐 TLE global scratch 分配/清零管线并验证 cooperative launch 驻留；未完成前不得开放 grid barrier。注：12.6 的部分预估已被 13.9 的实测修正（PPU 已有 scratch 分配 pass，共享 PTX 计数形式可被当前 SDK 接受；真正阻塞是 scratch 清零、cooperative 标志传播与 occupancy 校验）。
- **5.3 cluster/submesh：No-Go（本阶段）**。PPU cluster 语义未验证，显式拒绝。
- **Phase 6：No-Go**，直到满足 12.7 的多节点与 BAREX signal 条件；不得以单节点 world 测试替代跨节点验证。

## 13. Phase 5.1：PPU TLE Distributed Barrier — FlagCX Space Barrier

### 13.1 目标、范围与结论

本阶段只开放 `tle.distributed_barrier` 的 **FlagCX Space Barrier** 路径：

- `space="device"`（FlagCX INTRA team）与单节点 `space="world"`（FlagCX WORLD team）。
- 合法 barrier 形式：`sync` / `arrive` / `wait`。
- `group_kind="block"`；`group_kind="warp"` 仅在 `num_warps=1`（整个 CTA 只有一个 warp）时开放。
- `barrier_index` 0–7、`context_id` 0–3。
- 线程组 `thread` 因缺少单线程执行掩码，运行时 **UNVERIFIED**，编译期拒绝。
- 显式不开放：`space="inter"`、多节点 `space="world"`、grid/grid_axis、cluster/submesh，以及未经验证的 Lite-mode TLE。

结论：**Phase 5.1 Go**。2/4/8 卡、4096 轮设备侧 barrier、1M 元素 payload、sync/arrive/wait、index/context、错峰到达、重复轮次、normal/`.cg`/volatile 读取均通过；未进入 NVIDIA grid barrier 路径，未引入 NVIDIA PTX 依赖。

### 13.2 调用链与 lowering

```
tle.distributed_barrier(mesh, device_dptr=ctx, space="device"|"world",
                        barrier_kind="sync"|"arrive"|"wait",
                        group_kind="block"|"warp",
                        index=i, context_id=c, order=..., memory_scope=...)
  -> tle.distributed_barrier op
  -> tle.flagcx.barrier op (FlagCxBarrierOp)
  -> flagcxDevBarrierSync / flagcxDevBarrierArrive / flagcxDevBarrierWait
     (teamKind = INTRA(0) / WORLD(2), coopKind = BLOCK(2) / WARP(1))
  -> FlagCX device inline: intra epoch inbox；world 在单节点退化为 intra，
     多节点才走 inter FIFO/proxy signal（本阶段未开放）
```

关键点：

- PPU 的 FlagCX Device IR 使用**泛型 AS0 指针**。PPU lowering 调用 `populateFlagCx*OpToLLVMPatterns(..., ptrAddressSpace=0, collectiveFence=true)`；LLIR 中 barrier 参数由 `inttoptr (i64 ...)` 构造，**不出现** `addrspacecast`。
- Space barrier 不经过 `DistributedBarrierOpToLLVM` 的 grid/cluster 分支；该共享 pattern 只在 `FLAGCX_ENABLED` 下注册，space barrier 实际由 `FlagCxBarrierOp` 消费。
- `memory_scope` 与 `order` 被传递到 FlagCX Device API，但见 13.4 的有效语义说明。

### 13.3 代码修改

| 文件 | 修改 |
|---|---|
| `third_party/ppu/backend/tle_supported.py` | 将 `distributed_barrier` 加入 PPU 支持列表 |
| `third_party/ppu/backend/compiler.py` | `reject_unsupported_tle` 增加 barrier space/group/index/context 门控；`_runtime_is_multi_node` 拓扑判定；`HGGCOptions.dist_multi_node` 与 `tle_gate_version` 进入编译缓存 key；`file_hash` 对刚修改文件绕过 stat 缓存，避免同尺寸快速重写的陈旧 hash |
| `python/src/ir.cc` | 新增 `Operation.get_int_attr`，供门控读取 `barrier_index`/`context_id`/`team_kind` |
| `third_party/tle/dialect/include/Tools/FlagcxUtils.h`、`lib/Tools/FlagcxUtils.cpp` | `getUnifiedBarrierFuncCall` 增加 `collectiveFence`；非 THREAD group 在 arrive/sync 前、wait/sync 后发射 `ppu.fence.sc.sys` |
| `FlagCxBarrierOpToLLVM.h/.cpp`、`FlagCxOpToLLVM.cpp` | 将 `collectiveFence` 从 umbrellas 传入 barrier pattern；保留其他后端默认值（AS1、无 fence） |
| `third_party/ppu/lib/TritonPPUGPUToLLVM/TritonGPUToLLVM.cpp` | 注册 `populateDistributedBarrierOpToLLVMPatterns`；FlagCx patterns 以 AS0 + collective fence 调用 |
| `python/triton/experimental/tle/language/communication.py` | `cleanup_communicator` 幂等化：未初始化直接返回；销毁后清空 comm/dev_mem/dev_comm/win 并复位 `_init_communicator_` |
| `third_party/ppu/python/test/unit/tle/test_tle_signal_compile.py` | 适配 Phase 5.1 的 inter signal 门控：inter 编译用例改为拒绝用例 |
| `test_tle_barrier_compile.py`、`test_tle_barrier_space.py/.sh` | 新增编译级与 2/4/8 卡运行级覆盖 |

### 13.4 能力门控矩阵

| 条件 | 结果 | 依据 |
|---|---|---|
| `space="device"` + `group_kind="block"` | **SUPPORTED** | 2/4/8 卡运行验证，含 4096 轮与 1M payload |
| `space="device"` + `group_kind="warp"` + `num_warps=1` | **SUPPORTED** | 单 warp CTA 即一个 warp cooperative；运行验证 |
| `space="device"` + `group_kind="warp"` + `num_warps>1` | **UNVERIFIED，编译拒绝** | 多个 warp 会共享同一 epoch slot，FlagCX 未提供 warp 级隔离 |
| `space="device"` + `group_kind="thread"` | **UNVERIFIED，编译拒绝** | 需要“每个 barrier index 恰好一个线程执行”；TLE 当前无 thread-id 掩码 |
| `space="world"` + `group_kind="block"` + 单节点 | **SUPPORTED** | 2/4/8 卡运行验证（sync/arrive/wait、index/context） |
| `space="world"` + `group_kind="warp"/"thread"` | **UNVERIFIED，编译拒绝** | world barrier 组合 intra+inter；单节点时 `_nInterPeers==0` 会退化为 intra，但该 net 字段是运行期状态，无法在编译期证明不会走 inter。inter 的 signal increment/shadow 只对 block coop 有锁步保证，因此只开放已验证的 block |
| `space="world"` + 多节点 | **UNSUPPORTED，编译拒绝** | `WORLD_SIZE != LOCAL_WORLD_SIZE`；多节点环境未提供 |
| `space="inter"` | **UNSUPPORTED，编译拒绝** | 跨节点 FlagCX inter team 未验证 |
| `group_kind="grid"` / `grid_axis` / space=None | **UNSUPPORTED（Phase 5.1）** | grid barrier 未开放；TLE op verifier 对 `group_kind="grid"` 先报诊断，PPU 门控对 `space=None` 再拒绝 |
| `index` 不在 0–7 | **编译拒绝** | `create_dist_tensor` 申请 `barrierCount=8` |
| `context_id` 不在 0–3 | **编译拒绝** | `interContextCount=4` |
| FlagCX 不可用 | **编译拒绝** | `Distributed().is_available` |
| Lite-mode TLE | **UNVERIFIED** | 沿用 Phase 4.5 结论，不在本阶段开放 |

拓扑门控细节：

- `world` 仅当可证明单节点时允许：`WORLD_SIZE<=1`（仅编译/单 rank）或 `LOCAL_WORLD_SIZE==WORLD_SIZE`。
- 多 rank 运行若缺少 `LOCAL_WORLD_SIZE`，无法证明单节点，按多节点/UNVERIFIED 拒绝。
- `HGGCOptions.dist_multi_node` 参与 `hash()`，防止单节点编译产物在多节点启动时被缓存复用。
- Triton 的磁盘缓存查找发生在 `make_ttir`/`reject_unsupported_tle` **之前**；因此 `tle_gate_version` 也参与 `hash()`。修改 gate 策略时必须递增 `PPU_TLE_GATE_VERSION`，否则旧缓存仍可能被复用而绕过新门控。本次调试中一次“绕过 gate 的探针编译”即污染了 `_barrier_grid` 缓存，加版本号后测试恢复，说明该约束是实际存在的。

### 13.5 内存序与 FlagCX 行为核实

FlagCX `flagcxDevBarrier<Intra>` 的实际实现（`default_comm_traits.h`）：

- `arrive`：先 `_coop.sync()`，再由 **striped 线程**向每个 peer 的 inbox 写 `epoch+1`（release）。
- `wait`：striped 线程 acquire 轮询 peer inbox，更新 epoch，threadRank 0 写 epoch buffer，最后 `_coop.sync()`。
- `sync = arrive + wait`。

因此存在 leader/striped-only 的释放/获取缺口：非 striped 线程的 payload 写入/读取不由这些 release/acquire 排序。PPU 的修正：

- `collectiveFence && coopKind != THREAD` 时，在 `arrive`/`sync` 调用前发射每线程 `ppu.fence.sc.sys`；在 `wait`/`sync` 调用后再次发射。
- 该 fence 使每个线程的 payload store 在 barrier 发布前可见，并使 wait 返回后的 load 有序。
- `thread` group 在 FlagCX 中每个线程执行全部 peer 操作，本不需要额外 fence；但 barrier 的 thread group 已被 13.4 门控拒绝。

FlagCX API 参数的有效性：

- `flagcxDevBarrier*` 的 `scope` 形参被 `(void)scope` **忽略**。
- intra barrier 的 `order` 形参也未参与内部 atomic：`arrive` 固定 `Release` store，`wait` 固定 `Acquire` load；epoch buffer 使用 `Release`。
- world/inter barrier 的 signal 操作同样固定使用 release/acquire 语义；`order`/`scope` 不改变 wire 行为。
- 需要特别指出：`Barrier<World>` 组合 intra+inter；单节点 world team 的 `_nInterPeers==0`，`sync()` 实际退化为 intra barrier；多节点才会执行 inter signal/wait 后再做 intra broadcast。本阶段只开放单节点 world，但由于 `_nInterPeers` 是运行期 net 字段，门控仍只允许已验证的 block coop。
- 因此 PPU 上的有效语义是 **system-scope release/acquire（强于 relaxed）**，而不是用户传入的任意 `order`/`memory_scope`。本阶段保留参数兼容，但文档明确其不生效；需要 relaxed barrier 的用户不能依赖参数含义。

编译级证据（`test_tle_barrier_compile.py`）：

- 三个 device API 符号均出现：`flagcxDevBarrierSync/Arrive/Wait`。
- `inttoptr (i64 ...)` 存在，`addrspacecast` 不存在。
- 无 `atom.add.release.gpu.u32` / `ld.acquire.gpu.u32`（space 路径不走 NVIDIA grid PTX）。
- block/warp barrier 的 LLIR 有 `ppu.fence.sc.sys`；thread-only 用例在门控处被拒绝。
- 产物为 `hgbin`。

### 13.6 可见性：normal / `.cg` / volatile

运行测试在每轮 barrier 后读取上一 Rank 的 payload：

| 读取方式 | 配置 | 结果 |
|---|---|---|
| `volatile=True` | 2/4/8 卡，4096 轮 × 2 barrier，N=64 | PASS |
| `volatile=True` | 2/4/8 卡，1M 元素 payload，4 轮 | PASS |
| `volatile=True` | world block、arrive/wait、index/context、错峰到达 | PASS |
| `volatile=False`（normal） | 2/4/8 卡，4096 轮，N=64 | PASS |
| `volatile=False` + `cache_modifier=".cg"` | 2/4/8 卡，8 轮，N=256 | PASS |

结论：

- **barrier 路径未复现 normal load 陈旧问题**。原因是 intra `wait` 末尾有 `_coop.sync()`，随后所有线程执行 `ppu.fence.sc.sys`；获取由 striped 线程完成后经 CTA 同步传播到全部线程。Phase 4.5 的 signal 路径不同：`signal_wait` 的 leader-only 轮询返回后没有 group sync，非 leader 线程的 normal load 可能命中陈旧缓存，因此 signal/remote 仍要求 `volatile=True`。
- 因此不能把“normal load 一定失效”认定为 PPU 后端缺陷；更准确的结论是：**signal/wait 的 leader-only 路径需要 volatile 或额外的 group rendezvous，barrier 路径已具备 rendezvous**。为保守起见，跨 Rank 轮询仍推荐 `volatile=True`，尤其 signal；barrier 后读取在现有证据下 normal/`.cg` 也可用。
- 本阶段未修改 normal load 的 cache policy，也未实现 remote-load 自动 invalidate；这属于后续优化/后端语义工作。

### 13.7 测试矩阵与结果

编译级（`pytest -q test_tle_barrier_compile.py`）：**11 passed**

覆盖：device sync block、device arrive/wait warp（`num_warps=1`）、world sync block、index=7/context=3 上界、inter 拒绝、invalid group（verifier 诊断）、index=8 拒绝、context=4 拒绝、thread 拒绝、warp `num_warps=4` 拒绝、world warp 拒绝、world 多节点拓扑拒绝、grid/space=None 拒绝。

运行级（`test_tle_barrier_space.sh`，每 Rank 一个 CTA）：

| 卡数 | 参数 | 结果 | 耗时 |
|---|---|---|---|
| 2 | `BARRIER_ROUNDS=64 BARRIER_LONG_ROUNDS=4096 BARRIER_LONG_N=64` | **PASS**（2/2 Rank ALL PASS） | 59s |
| 4 | 同上 | **PASS**（4/4） | 65s |
| 8 | 同上 | **PASS**（8/8） | 93s |

配置覆盖：

- device block：`num_warps=1/2/4`。
- device warp：`num_warps=1`。
- device `arrive` + `wait` 分离。
- device `index=1/context=1`，`index=7/context=3`。
- world block sync、world arrive/wait、world `index=1/context=1`。
- 错峰到达（Rank 0 无延迟，Rank i 延迟 `i*2000` 次 volatile load）。
- normal load 8 轮观察、`.cg` 8 轮观察、normal load 4096 轮观察。
- 4096 轮纯设备侧循环，每轮两次 barrier，Host 只在配置之间同步。
- 1M float payload（4 轮）。
- 结束处重复调用 `cleanup_communicator()`，验证幂等释放。

全量回归：

| 测试 | 结果 |
|---|---|
| PPU TLE 单测 `pytest -q third_party/ppu/python/test/unit/tle/` | **175 passed, 5 skipped, 1 xfailed** |
| 2 卡 signal/wait 矩阵 | **PASS** |
| 2 卡 signal ping-pong（4096 轮、1M、warp/thread、star） | **PASS** |
| 4 卡 ping-pong（1024 轮） | **PASS** |
| 8 卡 ping-pong（512 轮） | **PASS** |
| 2 卡 remote intra-node | **PASS** |
| 2 卡 shard_id | **PASS** |
| 普通 Triton vector add + row reduction | **PASS** |
| Phase 5.1 clean build | 见下方构建记录 |

构建：

- Phase 5.1 C++ 增量构建 `build-phase51.log`：**EXIT=0**。
- 修改 FlagCX wrapper/metadata 后重新执行 `pip install -e . --no-build-isolation`：**EXIT=0**，ABI hash 刷新后编译测试通过。
- 最终 clean build：`rm -rf build && FLAGTREE_BACKEND=ppu pip install -e . --no-build-isolation`，日志 `/data/wenxuekun/flagtree-phase1/build-phase51-clean.log`，**EXIT=0**（541 个编译/链接目标；确认无增量缓存依赖）。
- clean build 后复跑：barrier 2/4/8 卡分别 **61s / 62s / 94s，全部 PASS**；全量 PPU TLE 单测 **175 passed, 5 skipped, 1 xfailed**；2 卡 signal/wait 与 ping-pong 均 **PASS**。

### 13.8 通信器 release / re-init 结论

- `cleanup_communicator()` 已改为幂等：重复调用直接返回；销毁后复位 `comm/dev_mem/dev_comm/win/_init_communicator_`。运行测试在末尾调用两次并检查 `dist.is_initialized()==False`，通过。
- **re-init 未开放**。调查结果：
  - 2 Rank、少量 barrier 轮次的最小探针有时可以通过；但在完整测试序列或 4 Rank 下，`cleanup -> flagcxCommInitRank` 会在 FlagCX/ACCL/NCCL bootstrap 阶段出现 `double free or corruption` / SIGABRT。
  - 触发与 barrier 轮数、world barrier 使用、Rank 数相关，但即使 device-only 4 Rank 也会复现；这是 FlagCX/NCCL bootstrap 与 ACCL 线程池 teardown 的竞争，不是 barrier lowering 的内存错误。
  - 尝试在 cleanup 中调用 `flagcxCommFinalize`（FlagCX 测试的 canonical teardown）改善了部分 2 Rank 场景，但未修复 4 Rank；且该 API 绑定位于被主仓库 ignore 的 FlagCX 生成 wrapper 中，无法以可复现方式提交，故已回退。
- 结论：Phase 5.1 只承诺 **release/重复 release 安全**；同进程 destroy -> re-init 标记为 **UNVERIFIED/外部限制**，默认测试不包含重初始化，避免把 FlagCX teardown 竞争误判为 barrier 回归。后续如需要，应单独与 FlagCX 上游协调 bootstrap/ACCL 生命周期。

### 13.9 Phase 5.2 预研（不实现）

本节为预研结论，**未修改 gate，也未开放 grid/grid_axis**。方法：在 `/tmp` 探针中临时绕过 PPU 门控（仅进程内 monkeypatch，不改源码），复用共享 `DistributedBarrierOpToLLVM` 的 grid 路径，运行最小 kernel。

#### 13.9.1 已有能力与实测

- **编译**：`tle.distributed_barrier(mesh)` 的 grid 路径在 PPU 上可以直接编译出 `hgbin`；LLIR 中保留共享 lowering 的 `atom.add.release.gpu.u32` / `ld.acquire.gpu.u32` inline asm，PPU `ppu-llc` 接受这些字符串。`global_scratch_size=4`、`global_scratch_align=4`。说明 12.6 中“必须新增 TLE scratch pass / 必须替换为 PPU 原子”的判断过于保守：PPU 已有 `add_allocate_global_scratch_memory`，共享 lowering 的 PTX 形式在当前 SDK 可被接受。
- **运行**：
  - 2 CTA，一次 grid barrier，barrier 后读取 peer：**PASS**。
  - 8 CTA，16 轮、每轮两次 grid barrier，non-cooperative 与 `launch_cooperative_grid=True`：均 **PASS**。
  - 64 CTA，8 轮，cooperative：**PASS**。
  - grid_axis：4 CTA / group=2、8 CTA / group=4，cooperative 与 non-cooperative 的 8 轮测试：均 **PASS**；两个 barrier 站点 scratch 为 2×groups×4 字节（例如 4 CTA/group2 为 16 字节）。
- **算法**：共享 lowering 使用 4 字节 sense-reversing 计数器（高 bit flip），每个 CTA 的 thread 0 到达，其余线程经 CTA barrier 等待；grid_axis 为每个 disjoint group 分配独立 4 字节计数器。4 字节足够覆盖 CTA 数量（上限 2^31），当前不需要 64 位原子；PPU 32 位原子可用。

#### 13.9.2 必须先解决的问题

1. **Cooperative launch 标志未传播到 metadata（阻塞性）**
   - 前端 `_apply_mesh_grid_launch` 在 semantic lowering 期间把 `options.launch_cooperative_grid=True`；但 `triton.compile` 的 metadata 字典在调用 `make_ir` **之前**已由 `options.__dict__` 快照生成。
   - 因此默认 grid barrier kernel 的 `metadata.launch_cooperative_grid` 实测为 **False**；只有显式传入该 option 时才为 True。
   - NVIDIA backend 在 `make_ttir` 中显式执行 `metadata["launch_cooperative_grid"] = opt.launch_cooperative_grid`（`third_party/nvidia/backend/compiler.py:474` 注释即说明 frontend 会修改该 option）；PPU 缺少这一步。Phase 5.2 必须补上。
2. **缺少驻留/occupancy 校验（阻塞性）**
   - 原子 grid barrier 要求所有参与 CTA 同时驻留；否则自旋等待会死锁。
   - PPU driver 已有 `HG_LAUNCH_ATTRIBUTE_COOPERATIVE` 路径，SDK 提供 `hggcOccupancyMaxActiveBlocksPerMultiprocessor`、`hggcOccupancyMaxPotentialBlockSize`、`hggcDevAttrCooperativeLaunch`；当前 driver 只有 cluster occupancy helper（固定 blockDim=128），没有 cooperative grid 的 `maxActiveBlocksPerSM × SM count` 检查。
   - Phase 5.2 应在 launcher 启动前计算并校验 `grid <= occupancy`，超限时报明确错误；`num_ctas>1` / cluster 组合显式拒绝。
3. **global scratch 必须清零（阻塞性）**
   - PPU driver 在每次 launch 用 `allocate_scratch` 调用户 allocator 分配 `grid_size * num_ctas * scratch_size`，但 **不做 memset**；用户 allocator 通常返回 `torch.empty`。
   - 预研中把 scratch 填成 `0x7f` 后，8 CTA / 8 轮探针 **超时挂起**；清零（或至少低 31 位为 0）后通过。说明共享 grid barrier 协议依赖初始计数器为 0（或等价状态），不能直接使用脏 scratch。
   - 方案：launch 前对 barrier scratch 做清零（例如 allocator 返回后 `zero_()`，或专用 memset）；或改造协议为带 generation/epoch、可容忍任意初值。前者最小、风险低，建议 Phase 5.2 先做。
4. **Allocator 前置条件**
   - 没有 `triton.set_allocator(...)` 时，带 global scratch 的 kernel 直接报错（`Kernel requires a runtime memory allocation...`）。Phase 5.2 需要文档化该要求，或为 PPU 提供默认 allocator（注意生命周期与 stream 顺序）。
5. **并发/多 stream**
   - 每次 launch 都调用 allocator 分配独立 scratch，理论上并发 stream 各自持有独立计数器；但要求 allocator 是 stream-ordered 且 buffer 在 kernel 完成前存活。尚未做同 kernel 双 stream 并发验证，Phase 5.2 应补。
6. **grid vs grid_axis 计划**
   - 先开放 whole-grid `grid`（无 group descriptor，scratch=4 字节/站点）；验证后再开放 `grid_axis`（`mesh.axis_group(...)`，scratch=groups×4/站点）。
   - 两者共用 `lowerGridBarrier`；风险主要在 scratch 清零、cooperative 驻留和 group descriptor 验证，而非 lowering 本身。

#### 13.9.3 最小实现计划（Phase 5.2）

1. 在 PPU `make_ttir` 中回填 `metadata["launch_cooperative_grid"]`（对齐 NVIDIA），确保 frontend 的 mesh grid 设置生效。
2. 在 PPU driver/launcher 增加 cooperative occupancy 校验：查询 `hggcOccupancyMaxActiveBlocksPerMultiprocessor` 与 SM 数，grid 超限时拒绝；cluster/num_ctas>1 组合拒绝。
3. launch 前清零 global scratch（或实现 generation 协议）；补 allocator 默认/文档；验证脏内存、连续 launch、双 stream。
4. 只对 PPU 重新开放 `group_kind="grid"` 的 gate，先写 1/2/4/8/64 CTA、1–4096 轮、payload 可见性、超时测试；稳定后再开放 `grid_axis`。
5. 更新能力矩阵与测试；cluster/submesh 继续 No-Go。

#### 13.9.4 Phase 5.2 Go/No-Go

- **Grid barrier：Conditional Go**。共享 lowering、PPU PTX 接受度、4 字节计数器、per-launch scratch 均已被探针证实；但必须完成 13.9.2 的 1–4 项（coop 元数据、occupancy、scratch 清零、allocator）后才可开放，否则存在死锁/挂起风险。
- **grid_axis：Conditional Go（第二阶段）**。探针已通过 group=2/4 的 4/8 CTA 测试，但应在 whole-grid 稳定后开放，并补 group descriptor/多 group 并发测试。
- **cluster/submesh：No-Go（本阶段）**。PPU cluster 语义未验证，继续保持拒绝。
- **Phase 6：No-Go**，条件不变（见 12.7）。

### 13.10 限制汇总

- Space barrier 是 **team barrier**，不是 grid barrier：一个 barrier index 在每个 Rank 上应只由一个 cooperative unit 使用；block group 要求每个 Rank 一个 CTA 参与，多 CTA 不会形成 grid-wide 同步（Phase 5.2 处理）。
- `group_kind="thread"` 与 `group_kind="warp"`（`num_warps>1`）在 PPU 上编译拒绝；warp 只在单 warp CTA 下验证。
- world 仅单节点、block group；inter/多节点 world 未验证并拒绝。
- FlagCX barrier 忽略 `order`/`scope`，有效为 system-scope release/acquire；不能依参数获得 relaxed 语义。
- barrier 后 normal/`.cg` load 在现有测试中可见，但 signal/remote 仍要求 `volatile=True`；跨 Rank 轮询建议继续使用 volatile。
- 同进程 destroy -> re-init 未开放（FlagCX/ACCL teardown 竞争）。
- 未验证：多 CTA per Rank、grid>1、cluster/submesh、多节点 inter/world、Lite-mode TLE barrier、与 CUDA Graph/多 stream 的组合。
- P2P buffer 环境约束、`FLAGCX_P2P_BUFFER_SIZE/CHUNK_SIZE` 与设备占用限制沿用 Phase 4.5。
