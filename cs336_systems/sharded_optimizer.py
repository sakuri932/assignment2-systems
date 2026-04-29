from __future__ import annotations

import torch
import torch.distributed as dist


class ShardedOptimizer:
    """ZeRO Stage-1 sharded optimizer: each rank owns optimizer state for ~1/world_size params."""

    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Materialize the full param list once (params may be a generator)
        self.all_params: list[torch.nn.Parameter] = list(params)

        # Assign params round-robin by index
        self.owned_params = [
            p for i, p in enumerate(self.all_params) if i % self.world_size == self.rank
        ]

        # Local optimizer only manages owned params; fall back to a dummy tensor if empty
        local_params = self.owned_params if self.owned_params else [torch.zeros(1, requires_grad=True)]
        self.local_optimizer = optimizer_cls(local_params, **kwargs)

    def zero_grad(self, set_to_none: bool = False) -> None:
        for p in self.all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self) -> None:
        # Step only on owned params
        if self.owned_params:
            self.local_optimizer.step()

        # Broadcast each param from its owner so all ranks stay in sync
        for i, param in enumerate(self.all_params):
            owner = i % self.world_size
            dist.broadcast(param.data, src=owner)

    # Expose state_dict / load_state_dict so tests can inspect if needed
    def state_dict(self):
        return self.local_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.local_optimizer.load_state_dict(state_dict)
