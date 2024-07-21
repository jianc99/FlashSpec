import os
import torch
import torch.distributed as dist

def _get_global_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_local():
    return _get_global_rank() == 0

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


global_rank = _get_global_rank()
world_size = _get_world_size()
torch.cuda.set_device(global_rank)
dist.init_process_group(backend="nccl")
global_group = dist.group.WORLD
draft_group = dist.new_group([0,1])

inp = torch.full((2, 1, 4096), global_rank, dtype=torch.bfloat16, device="cuda")
dist.all_reduce(inp, group=global_group)
expect = sum(range(world_size))
assert inp.eq(expect).all()

if 0 <= global_rank < 2:
    inp = torch.full((2, 1, 2048), global_rank, dtype=torch.bfloat16, device="cuda")
    dist.all_reduce(inp, group=draft_group)
    expect = sum(range(2))
    assert inp.eq(expect).all()

torch.cuda.synchronize()
print("success")
dist.destroy_process_group()