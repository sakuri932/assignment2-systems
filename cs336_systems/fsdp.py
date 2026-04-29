from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn


class FSDP(nn.Module):
    """Fully-Sharded Data Parallel: shards Linear/Embedding weights across ranks."""

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Maps id(param) -> metadata dict for sharded parameters
        self._sharded: dict[int, dict] = {}

        self._shard_params()

    # ------------------------------------------------------------------
    # Sharding helpers
    # ------------------------------------------------------------------

    def _shard_params(self) -> None:
        from cs336_basics.model import Embedding, Linear  # type: ignore

        for _name, mod in self.module.named_modules():
            if not isinstance(mod, (Linear, Embedding)):
                continue
            param = mod.weight
            pid = id(param)
            if pid in self._sharded:
                continue  # skip tied weights already registered

            full_shape = param.data.shape
            flat = param.data.detach().flatten()
            total = flat.numel()
            chunk = math.ceil(total / self.world_size)
            start = self.rank * chunk
            end = min(start + chunk, total)
            shard = flat[start:end].clone()

            # Replace weight data with the fp32 shard (1-D tensor)
            param.data = shard

            self._sharded[pid] = {
                "param": param,
                "full_shape": full_shape,
                "total": total,
                "chunk": chunk,
                "start": start,
                "end": end,
            }

    def _all_gather_full(self, info: dict) -> torch.Tensor:
        """All-gather the shards from all ranks and return the full fp32 tensor."""
        param = info["param"]
        chunk = info["chunk"]
        total = info["total"]
        full_shape = info["full_shape"]
        shard = param.data  # current fp32 shard

        # Pad shard to chunk size if this is the last rank and total % world_size != 0
        if shard.numel() < chunk:
            padded = torch.zeros(chunk, dtype=shard.dtype, device=shard.device)
            padded[: shard.numel()] = shard
        else:
            padded = shard.contiguous()

        gathered = [torch.zeros(chunk, dtype=padded.dtype, device=padded.device) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded)

        full_flat = torch.cat(gathered, dim=0)[:total]
        return full_flat.view(full_shape)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        # Save fp32 shards and replace param.data with full weight (possibly cast)
        saved_shards: dict[int, torch.Tensor] = {}
        for pid, info in self._sharded.items():
            param = info["param"]
            saved_shards[pid] = param.data.clone()
            full_fp32 = self._all_gather_full(info)
            if self.compute_dtype is not None:
                param.data = full_fp32.to(self.compute_dtype)
            else:
                param.data = full_fp32

        self._saved_shards = saved_shards  # keep for finish_gradient_synchronization

        output = self.module(*args, **kwargs)
        # Keep full weight in param.data through backward (do NOT restore shard here)
        return output

    # ------------------------------------------------------------------
    # After-backward synchronization
    # ------------------------------------------------------------------

    def finish_gradient_synchronization(self) -> None:
        """Reduce-scatter gradients for sharded params; all-reduce for replicated params."""
        sharded_pids = set(self._sharded.keys())

        # --- Sharded params: reduce-scatter via all-reduce + slice ---
        for pid, info in self._sharded.items():
            param = info["param"]
            start = info["start"]
            end = info["end"]
            total = info["total"]
            chunk = info["chunk"]

            if param.grad is None:
                param.data = self._saved_shards[pid]
                continue

            full_grad_flat = param.grad.to(torch.float32).flatten()
            pad = self.world_size * chunk - total
            if pad > 0:
                full_grad_flat = torch.cat([full_grad_flat, full_grad_flat.new_zeros(pad)])
            full_grad_flat = full_grad_flat.contiguous()
            dist.all_reduce(full_grad_flat, op=dist.ReduceOp.SUM)
            full_grad_flat = full_grad_flat / self.world_size

            shard_grad = full_grad_flat[start:end].clone()
            param.data = self._saved_shards[pid]
            param.grad = shard_grad

        # --- Replicated params: all-reduce gradient so all ranks get the mean ---
        for name, param in self.module.named_parameters():
            if id(param) in sharded_pids:
                continue
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad.to(torch.float32).contiguous()
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(self.world_size)
            param.grad = grad

    # ------------------------------------------------------------------
    # Gather full params (for checkpointing / testing)
    # ------------------------------------------------------------------

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        pid_to_name: dict[int, list[str]] = {}

        for name, param in self.module.named_parameters():
            pid = id(param)
            pid_to_name.setdefault(pid, []).append(name)

        for name, param in self.module.named_parameters():
            pid = id(param)
            if pid in self._sharded:
                info = self._sharded[pid]
                full = self._all_gather_full(info)
                result[name] = full
            else:
                result[name] = param.data.clone()

        return result
