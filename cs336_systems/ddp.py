from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    """Distributed Data Parallel wrapper with async per-parameter gradient all-reduce."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        # Broadcast rank-0 parameters to all ranks
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        self._async_handles: list[tuple] = []

        # Register async all-reduce hook on each unique parameter
        seen = set()
        for param in module.parameters():
            if not param.requires_grad:
                continue
            ptr = param.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)

            def _make_hook(p: torch.nn.Parameter):
                def hook(param: torch.nn.Parameter) -> None:
                    handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                    self._async_handles.append((handle, param.grad))
                return hook

            param.register_post_accumulate_grad_hook(_make_hook(param))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for handle, grad in self._async_handles:
            handle.wait()
            grad.div_(world_size)
        self._async_handles.clear()

    # Forward named_parameters / parameters to the underlying module so that
    # validate_ddp_net_equivalence (which uses net.module) still works, and the
    # optimizer can iterate parameters directly on the DDP wrapper.
    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)
