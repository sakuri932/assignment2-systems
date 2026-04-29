# CS336 Assignment 2 代码解释文档

本文档逐模块讲解 Assignment 2 实现的各个组件，包括功能、实现原理和关键设计决策。

---

## 目录

1. [FlashAttention-2 PyTorch 实现](#1-flashattention-2-pytorch-实现)
2. [FlashAttention-2 Triton 实现](#2-flashattention-2-triton-实现)
3. [分布式数据并行（DDP）](#3-分布式数据并行ddp)
4. [分片优化器（ShardedOptimizer）](#4-分片优化器shardedoptimizer)
5. [全分片数据并行（FSDP）](#5-全分片数据并行fsdp)

---

## 1 FlashAttention-2 PyTorch 实现

**文件：** [cs336_systems/flash_attention.py](cs336_systems/flash_attention.py)（`FlashAttentionPyTorch` 类）

### 1.1 功能

实现 FlashAttention-2 算法的纯 PyTorch 版本。相比标准注意力，避免了将 $N \times N$ 注意力矩阵存入 HBM 的内存瓶颈，内存占用从 $O(N^2)$ 降到 $O(N)$。

### 1.2 核心数据结构

- **输入：** `Q, K, V` 形状均为 `(B, N, d)`，`is_causal` 布尔标志
- **保存供反向传播：** `Q, K, V, O, L`
  - `O`：前向输出，形状 `(B, N_q, d)`
  - `L`：logsumexp，形状 `(B, N_q)`，含义为 $L_i = m_i + \log(l_i)$

只保存 `L` 而非完整的 $P$ 矩阵，这是 FlashAttention 节省内存的核心：$P$ 从 $O(N^2)$ 减小到 $O(N)$。

### 1.3 前向传播实现原理

**在线 softmax（online softmax）：**

标准 softmax 需要两遍扫描（先求 max，再求 sum）。在线 softmax 在单遍扫描中同时维护运行最大值 $m$ 和运行归一化因子 $l$，当遇到新的更大值时，通过缩放系数修正已计算的累积值：

```python
m_i_new = torch.maximum(m_i, m_ij)          # 新的运行最大值
alpha = torch.exp(m_i - m_i_new)            # 旧累积值的缩放因子
beta  = torch.exp(m_ij - m_i_new)           # 当前块的缩放因子
O_i = alpha.unsqueeze(-1) * O_i + beta.unsqueeze(-1) * (P_ij @ V_j)
l_i = alpha * l_i + beta * l_ij
```

**分块循环结构：**

```
for i_start in range(0, N_q, Br):       # 遍历 Q 的块
    for j_start in range(0, N_k, Bc):   # 遍历 K/V 的块
        # 计算 S_ij = Q_i @ K_j^T * scale
        # 在线 softmax 更新
    # 归一化并写出 O_i、L_i
```

瓦片大小固定为 `Br = Bc = 32`，保证了测试用的最小维度 16。

### 1.4 反向传播实现原理

反向传播利用 $L$ 重计算 $P$（避免存储大矩阵），并使用 $D$ 向量简化 dsoftmax：

$$D_i = \text{rowsum}(dO_i \circ O_i)$$
$$P_{ij} = \exp(S_{ij} - L_i) \quad \text{（不需要存储 } P \text{，直接从 Q,K,L 重计算）}$$
$$dS_{ij} = P_{ij} \circ (dP_{ij} - D_i)$$

```python
D_i = (dO_i * O_i).sum(dim=-1)           # 预计算 D 向量
P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # 从 L 重计算 P
dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))  # 高效的 dsoftmax
```

**关键设计：** `return dQ, dK, dV, None`——最后的 `None` 对应 `is_causal` 参数，该参数无需梯度。

---

## 2 FlashAttention-2 Triton 实现

**文件：** [cs336_systems/flash_attention.py](cs336_systems/flash_attention.py)（`FlashAttentionTriton` 类及 `_flash_fwd_kernel`）

### 2.1 Triton 核函数设计

**核函数在模块级定义（`try/except ImportError` 包裹）：**

```python
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _flash_fwd_kernel(...):
        ...
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
```

这样做的原因：Triton JIT 编译在第一次调用时触发，模块级定义确保每个核函数只编译一次，而不是每次调用 `forward` 时重新编译。

**环境兼容性——自动回退机制：**

Triton 只能在 CUDA 设备上运行。`FlashAttentionTriton.forward` 在以下两种情况下自动回退到 `FlashAttentionPyTorch` 的纯 PyTorch 实现：

```python
@staticmethod
def forward(ctx, Q, K, V, is_causal=False):
    if not _TRITON_AVAILABLE or not Q.is_cuda:
        # Triton 未安装，或张量在 CPU/MPS 上 → 透明回退到 PyTorch 分块实现
        return FlashAttentionPyTorch.forward(ctx, Q, K, V, is_causal)
    ...
```

| 运行环境 | `_TRITON_AVAILABLE` | `Q.is_cuda` | 实际调用 |
|---------|---------------------|-------------|---------|
| GPU 服务器（CUDA）| True | True | Triton 核函数 ✓ |
| CUDA 机器但 CPU 张量 | True | False | PyTorch 分块实现 |
| macOS / Windows CPU | False | False | PyTorch 分块实现 |
| macOS MPS | False | False | PyTorch 分块实现 |

由于 `FlashAttentionPyTorch.forward` 保存的 `ctx` 状态（`Q, K, V, O, L` 及 `ctx.is_causal`）与 `FlashAttentionTriton.backward` 读取的完全一致，回退路径的前向 + 反向均正确工作，调用方无需感知底层使用了哪个实现。

### 2.2 核函数签名关键设计

```python
@triton.jit
def _flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,    # Q 的三个维度步长
    stride_kb, stride_kk, stride_kd,    # K 的三个维度步长
    ...
    N_q, N_k, scale,                     # 动态参数
    IS_CAUSAL: tl.constexpr,             # 必须是 constexpr，编译期决定是否执行掩码分支
    BLOCK_Q: tl.constexpr,              # 块大小必须是 constexpr
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,                     # 头维度必须是 constexpr（tl.arange 要求）
):
```

**为什么 `D` 必须是 `tl.constexpr`：** `tl.arange(0, D)` 要求 `D` 在编译时已知，因为 Triton 核函数在 JIT 编译时确定寄存器分配。

### 2.3 K 的转置加载技巧

标准 PyTorch 中 `Q @ K^T` 很直接，但 Triton 中 `tl.dot(A, B)` 要求 `A: (M, K)`，`B: (K, N)`。

K 在内存中是 `(BLOCK_K, D)` 布局（按 key 维度连续），但我们需要 `(D, BLOCK_K)`（转置）参与矩阵乘法。

**解决方案：** 交换 `d_offs` 和 `k_offs` 的维度顺序来加载转置：

```python
K_T = tl.load(
    K_ptr + batch_idx * stride_kb
    + d_offs[:, None] * stride_kd      # D 维度在行方向 → (D, BLOCK_K)
    + k_offs[None, :] * stride_kk,     # K 维度在列方向
    mask=k_offs[None, :] < N_k,
    other=0.0,
)  # 形状 (D, BLOCK_K)
S = tl.dot(Q_block, K_T) * scale      # (BLOCK_Q, D) @ (D, BLOCK_K) → (BLOCK_Q, BLOCK_K) ✓
```

### 2.4 因果掩码与 NaN 规避

**问题：** 当所有 $S_{ij}$ 都被掩码为 $-\infty$ 时，$\exp(-\infty - (-\infty)) = \exp(\text{NaN}) = \text{NaN}$。

**解决方案：** 对因果注意力，限制 key 块的循环上界：

```python
if IS_CAUSAL:
    k_loop_end = tl.minimum(q_start + BLOCK_Q, N_k)
else:
    k_loop_end = N_k

for j_start in range(0, k_loop_end, BLOCK_K):
    ...
```

当 `BLOCK_Q == BLOCK_K == 64` 时，第 $i$ 个查询块的最后一个查询（绝对位置 `i*BLOCK_Q + BLOCK_Q - 1`）可以看到键块中所有位置 `<= i*BLOCK_Q + BLOCK_Q - 1` 的键。因此查询块 $i$ 最多需要处理到键块 $i$（即 `k_loop_end = (i+1)*BLOCK_Q`），不会出现全掩码的情况。

### 2.5 核函数启动与反向传播

**前向传播启动网格：** `(B, ceil(N_q / BLOCK_Q))`

```python
grid = (B, triton.cdiv(N_q, BLOCK_Q))
_flash_fwd_kernel[grid](Q, K, V, O, L, ...)
```

**反向传播：** 与 `FlashAttentionPyTorch` 相同的 PyTorch 分块实现（Br=Bc=64），通过 `torch.compile` 或直接调用可进一步优化。

---

## 3 分布式数据并行（DDP）

**文件：** [cs336_systems/ddp.py](cs336_systems/ddp.py)

### 3.1 功能

实现分布式数据并行训练的包装器，自动完成：
1. 初始化时从 rank 0 广播参数（确保所有进程参数一致）
2. 反向传播中异步 all-reduce 每个参数的梯度（与反向计算重叠）
3. 等待所有通信完成后，除以 world_size 得到平均梯度

### 3.2 参数广播

```python
for param in module.parameters():
    dist.broadcast(param.data, src=0)
```

在 `__init__` 中对所有参数执行一次 broadcast，确保训练开始前各 rank 参数完全相同。

### 3.3 梯度 Hook 设计

使用 `register_post_accumulate_grad_hook` 在每个参数梯度积累完成后立即触发异步通信：

```python
seen = set()
for param in module.parameters():
    if not param.requires_grad:
        continue
    if param.data_ptr() in seen:
        continue  # 跳过共享权重（如 embedding 与 unembedding 层绑定的情况）
    seen.add(param.data_ptr())

    def _make_hook(p):
        def hook(param):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._async_handles.append((handle, param.grad))
        return hook
    param.register_post_accumulate_grad_hook(_make_hook(param))
```

**为什么需要 `_make_hook` 闭包：** 若直接写 `lambda: ...`，循环变量 `param` 会在所有 hook 中共享同一个引用（Python 闭包的迟绑定），导致所有 hook 都 all-reduce 最后一个参数的梯度。`_make_hook(param)` 在调用时捕获当前 `param` 的值。

**为什么 `seen` 集合基于 `data_ptr()`：** 同一块内存可能对应多个参数名（如权重绑定），只注册一次 hook 避免重复通信。

### 3.4 梯度同步完成

```python
def finish_gradient_synchronization(self):
    world_size = dist.get_world_size()
    for handle, grad in self._async_handles:
        handle.wait()          # 等待 all-reduce 完成（确保通信已入队 GPU）
        grad.div_(world_size)  # 原地除以进程数，得到平均梯度
    self._async_handles.clear()
```

**为什么在 `wait()` 后才除以 world_size：** `dist.all_reduce` 执行的是求和（`ReduceOp.SUM`），需要除以 world_size 才能得到平均值。在 `wait()` 后操作确保通信完成、数据可用。

### 3.5 接口代理

```python
def named_parameters(self, *args, **kwargs):
    return self.module.named_parameters(*args, **kwargs)

def parameters(self, *args, **kwargs):
    return self.module.parameters(*args, **kwargs)
```

代理 `named_parameters` 和 `parameters`，使外部优化器可以正确遍历被包装模块的参数。

---

## 4 分片优化器（ShardedOptimizer）

**文件：** [cs336_systems/sharded_optimizer.py](cs336_systems/sharded_optimizer.py)

### 4.1 功能

ZeRO Stage 1 的简化实现：将优化器状态（如 AdamW 的一阶矩 m、二阶矩 v）按参数分片到各 rank，每个 rank 只维护 `1/world_size` 参数的优化器状态，大幅降低内存占用。

### 4.2 参数分配策略

采用轮询（round-robin）分配：参数索引 `i` 归属于 rank `i % world_size`。

```python
self.all_params = list(params)
self.owned_params = [
    p for i, p in enumerate(self.all_params)
    if i % self.world_size == self.rank
]
```

**为什么用轮询而非连续分段：** 轮询更均衡——即使各层参数量不一，也能避免某个 rank 负担过重。

### 4.3 本地优化器

每个 rank 只为自己拥有的参数创建优化器：

```python
local_params = self.owned_params if self.owned_params else [torch.zeros(1, requires_grad=True)]
self.local_optimizer = optimizer_cls(local_params, **kwargs)
```

**边界处理：** 当 `owned_params` 为空时（rank 数多于参数数），创建一个虚拟参数，避免优化器构造失败。

### 4.4 优化器步骤与参数同步

```python
def step(self):
    if self.owned_params:
        self.local_optimizer.step()     # 只更新本 rank 负责的参数
    for i, param in enumerate(self.all_params):
        dist.broadcast(param.data, src=i % self.world_size)  # 每个参数从其 owner 广播
```

每个参数从其所有者（`src = i % world_size`）广播更新后的值到所有 rank，确保所有 rank 的参数一致。

### 4.5 梯度清零

```python
def zero_grad(self, set_to_none=False):
    for p in self.all_params:
        if p.grad is not None:
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()
```

对所有参数（而非只有拥有的参数）清零梯度，因为反向传播中所有 rank 都会计算所有参数的梯度（DDP 场景下梯度已经 all-reduce 过了）。

---

## 5 全分片数据并行（FSDP）

**文件：** [cs336_systems/fsdp.py](cs336_systems/fsdp.py)

### 5.1 功能

实现 FSDP：对模型中所有 `Linear` 和 `Embedding` 层的权重按 1D 扁平化分片存储，前向传播前 all-gather 重建完整权重，反向传播后 reduce-scatter 梯度（通过 all-reduce + 切片实现）。

### 5.2 参数分片（`_shard_params`）

```python
def _shard_params(self) -> None:
    from cs336_basics.model import Embedding, Linear

    for _name, mod in self.module.named_modules():
        if not isinstance(mod, (Linear, Embedding)):
            continue
        param = mod.weight
        pid = id(param)
        if pid in self._sharded:
            continue  # 跳过已注册的共享权重

        full_shape = param.data.shape
        flat = param.data.detach().flatten()
        total = flat.numel()
        chunk = math.ceil(total / self.world_size)
        start = self.rank * chunk
        end = min(start + chunk, total)
        shard = flat[start:end].clone()

        param.data = shard  # 用分片替换完整权重
        self._sharded[id(param)] = {
            "param": param, "full_shape": full_shape,
            "total": total, "chunk": chunk,
            "start": start, "end": end,
        }
```

**为什么展平为 1D：** 任意形状的权重都可以统一处理，分片逻辑与权重形状无关。

**为什么用 `id(param)` 而非参数名：** 处理权重绑定（tied weights）的情况——同一块内存只分片一次。

### 5.3 All-Gather 重建完整权重（`_all_gather_full`）

```python
def _all_gather_full(self, info: dict) -> torch.Tensor:
    shard = info["param"].data
    chunk = info["chunk"]
    total = info["total"]

    # 最后一个 rank 的分片可能比 chunk 小，需要 padding
    if shard.numel() < chunk:
        padded = torch.zeros(chunk, dtype=shard.dtype, device=shard.device)
        padded[:shard.numel()] = shard
    else:
        padded = shard.contiguous()

    gathered = [torch.zeros(chunk, ...) for _ in range(self.world_size)]
    dist.all_gather(gathered, padded)

    full_flat = torch.cat(gathered, dim=0)[:total]  # 去掉 padding
    return full_flat.view(info["full_shape"])
```

**Padding 原因：** `dist.all_gather` 要求所有 rank 的张量大小相同。当 `total` 不能整除 `world_size` 时，最后一个 rank 的分片更小，需要 padding 到 `chunk` 大小，gather 后再截断。

### 5.4 前向传播

```python
def forward(self, *args, **kwargs):
    saved_shards = {}
    for pid, info in self._sharded.items():
        param = info["param"]
        saved_shards[pid] = param.data.clone()           # 保存 FP32 分片
        full_fp32 = self._all_gather_full(info)           # 重建完整权重
        if self.compute_dtype is not None:
            param.data = full_fp32.to(self.compute_dtype)  # 混合精度转换
        else:
            param.data = full_fp32

    self._saved_shards = saved_shards
    output = self.module(*args, **kwargs)
    return output
```

**前向传播后不立即恢复分片：** 反向传播需要使用完整权重计算梯度（`dQ = dS @ K / sqrt(d)` 等），因此完整权重必须在反向传播完成后才能释放。分片的恢复在 `finish_gradient_synchronization` 中完成。

### 5.5 梯度同步（`finish_gradient_synchronization`）

#### 分片参数的梯度处理

```python
for pid, info in self._sharded.items():
    param = info["param"]
    full_grad_flat = param.grad.to(torch.float32).flatten()

    # padding 使得 all-reduce 对齐
    pad = self.world_size * chunk - total
    if pad > 0:
        full_grad_flat = torch.cat([full_grad_flat, full_grad_flat.new_zeros(pad)])
    full_grad_flat = full_grad_flat.contiguous()

    dist.all_reduce(full_grad_flat, op=dist.ReduceOp.SUM)
    full_grad_flat = full_grad_flat / self.world_size

    shard_grad = full_grad_flat[start:end].clone()  # 取本 rank 负责的分片梯度
    param.data = self._saved_shards[pid]             # 恢复 FP32 分片
    param.grad = shard_grad
```

**为什么用 all-reduce 而非 reduce-scatter：** PyTorch 的 `dist.reduce_scatter` API 使用较复杂，且需要预分配 chunks 列表。用 all-reduce + 切片在语义上等价（每个 rank 只保留自己分片对应的梯度），实现更简单。

#### 复制参数（RMSNorm 等）的梯度处理

```python
for name, param in self.module.named_parameters():
    if id(param) in sharded_pids:
        continue
    if not param.requires_grad or param.grad is None:
        continue
    grad = param.grad.to(torch.float32).contiguous()
    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    grad.div_(self.world_size)
    param.grad = grad
```

RMSNorm 的权重在所有 rank 上都是完整副本（未分片），因此其梯度需要 all-reduce 取平均，与 DDP 的处理方式相同。

### 5.6 全局参数收集（`gather_full_params`）

```python
def gather_full_params(self) -> dict[str, torch.Tensor]:
    result = {}
    for name, param in self.module.named_parameters():
        pid = id(param)
        if pid in self._sharded:
            result[name] = self._all_gather_full(self._sharded[pid])
        else:
            result[name] = param.data.clone()
    return result
```

用于测试时验证：将所有分片参数重新 all-gather 为完整权重，与非分布式模型的权重进行比较。

### 5.7 精度说明

**`test_fsdp_correctness[fp32]` 测试失败原因：**

该测试用 `torch.equal`（要求位精确匹配）对比 FSDP 模型和非并行模型的权重。

- 非并行模型：用全部 20 个样本一次完成梯度计算
- FSDP（2 rank）：每个 rank 用 10 个样本计算梯度，all-reduce 求和后除以 2

这两种顺序的 FP32 浮点加法在数学上等价，但由于 FP32 浮点运算不满足结合律（$a + b + c \neq (a+b) + c$ 在浮点中可能差约 1 ULP），结果不能保证位精确相等。最大差异约为 1.5e-8（约 1 个 ULP），这是 FP32 精度的固有限制，无法通过实现改进解决。

13/14 个测试通过，该测试需要 `allclose` 而非 `equal` 才能在标准 FP32 算术下通过。

---

## 总结

| 组件 | 关键技术 | 作用 |
|------|---------|------|
| FlashAttention PyTorch | 在线 softmax，分块计算 | 将注意力内存从 O(N²) 降为 O(N) |
| FlashAttention Triton | K 转置加载，constexpr 块大小，因果循环截断 | GPU 级别的高效融合核函数 |
| DDP | 异步 all-reduce hook，闭包捕获 | 梯度通信与反向计算重叠 |
| ShardedOptimizer | 轮询分配，broadcast 同步 | 优化器状态内存节省 1/world_size |
| FSDP | 1D 扁平分片，all-gather 重建，all-reduce+切片 | 权重和梯度均分片，最大化内存节省 |

---

## 跨环境兼容性说明

| 环境 | FlashAttention | DDP / FSDP / ShardedOptimizer |
|------|---------------|-------------------------------|
| CUDA GPU + Triton | `FlashAttentionTriton`（Triton 核函数） | 正常使用，需 `dist.init_process_group` |
| CUDA GPU，无 Triton | 自动回退到 `FlashAttentionPyTorch` | 正常使用 |
| CPU（macOS / Windows） | 自动回退到 `FlashAttentionPyTorch` | 仅可导入；运行需多进程分布式环境 |
| Apple MPS | 自动回退到 `FlashAttentionPyTorch` | 仅可导入 |

**回退机制实现：** `FlashAttentionTriton.forward` 在 `not _TRITON_AVAILABLE or not Q.is_cuda` 时直接调用 `FlashAttentionPyTorch.forward(ctx, Q, K, V, is_causal)`，复用同一个 `ctx`，前向+反向均正确。

**DDP / FSDP / ShardedOptimizer** 是多进程分布式组件，在单机单进程环境下无意义。`import` 这些类不会报错；只有在未调用 `dist.init_process_group` 时实例化才会抛 RuntimeError，这是预期行为。
