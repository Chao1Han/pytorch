import itertools
import os
from contextlib import nullcontext
from unittest import skip, skipIf

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import (
    _fused_matmul_reduce_scatter_fallback,
    _fused_matmul_reduce_scatter,
    _test_mode,
    enable_symm_mem_for_group,
    restride_A_for_fused_matmul_reduce_scatter,
    restride_A_shard_for_fused_all_gather_matmul,
)
import os
import torch.multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description="test_symm")
parser.add_argument("M", type=int, default=8192, help="M value")
parser.add_argument("N", type=int, default=4096, help="N value")
parser.add_argument("K", type=int, default=7168, help="K value")

args = parser.parse_args()

print("M = ", args.M, flush=True)
print("N = ", args.N, flush=True)
print("K = ", args.K, flush=True)

BATCH = 1
M = args.M
N = args.N
K = args.K
Loop = 10
enable_profile = False

# Get rank from MPI environment
rank = int(os.environ.get('PMI_RANK', os.environ.get('OMPI_COMM_WORLD_RANK', 0)))
world_size = int(os.environ.get('PMI_SIZE', os.environ.get('OMPI_COMM_WORLD_SIZE', 1)))

# Set ZE_AFFINITY_MASK for device binding (critical for ISHMEM)
os.environ['ZE_AFFINITY_MASK'] = str(rank)

os.environ['RANK'] = str(rank)
os.environ['WORLD_SIZE'] = str(world_size)
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29803'

# Initialize process group - this triggers ISHMEM registration
dist.init_process_group(backend='xccl', rank=rank, world_size=world_size)

# Now we can set the backend
import torch.distributed._symmetric_memory as symm_mem
# symm_mem.set_backend("ISHMEM")
group_name = dist.group.WORLD.group_name
symm_mem.enable_symm_mem_for_group(group_name)

def test_matmul_reducescatter(rank, world_size):
    # Device is always 0 after ZE_AFFINITY_MASK filtering
    torch.xpu.set_device(0)

    group = dist.group.WORLD

    torch.manual_seed(42 + rank)
    A = torch.rand(M, K, device="xpu", dtype=torch.bfloat16)
    B = torch.rand(K, N, device="xpu", dtype=torch.bfloat16)
    scatter_dim = 0
    
    print(f"[Rank {rank}] Initialized, A.shape={A.shape}, B.shape={B.shape}", flush=True)

    begin_events_ref = [
        torch.xpu.Event(enable_timing=True) for _ in range(Loop)
    ]
    end_events_ref = [torch.xpu.Event(enable_timing=True) for _ in range(Loop)]

    begin_events = [
        torch.xpu.Event(enable_timing=True) for _ in range(Loop)
    ]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(Loop)]

    if enable_profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
    else:
        prof = nullcontext()


    # warm up fallback aten ops
    for i in range(Loop):
        output_0 = _fused_matmul_reduce_scatter_fallback(
            A, B, "sum", scatter_dim=scatter_dim, group_name=group.group_name
        )
    torch.xpu.synchronize()
    print(f"[Rank {rank}] Fallback warm-up completed, starting timed runs...", flush=True)
    for i in range(Loop):
        begin_events_ref[i].record()
        output_0 = _fused_matmul_reduce_scatter_fallback(
            A, B, "sum", scatter_dim=scatter_dim, group_name=group.group_name
        )
        end_events_ref[i].record()
    torch.xpu.synchronize()
    print(f"[Rank {rank}] Fallback timed runs completed", flush=True)
    
    # warm up symmetric ops
    print(f"[Rank {rank}] Starting warm-up for symmetric ops...", flush=True)
    for i in range(Loop):
        output_1 = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, "sum", scatter_dim=scatter_dim, group_name=group.group_name
        )
    torch.xpu.synchronize()
    print(f"[Rank {rank}] Symmetric warm-up completed, starting timed runs...", flush=True)
    with prof:
        for i in range(Loop):
            begin_events[i].record()
            output_1 = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                A, B, "sum", scatter_dim=scatter_dim, group_name=group.group_name
            )
            end_events[i].record()
        torch.xpu.synchronize()
        print(f"[Rank {rank}] Symmetric timed runs completed", flush=True)

    print(f"[Rank {rank}] Calculating latencies...", flush=True)
    latencies_ref = [b.elapsed_time(e) for b, e in zip(begin_events_ref, end_events_ref)]
    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]

    # Print results BEFORE cleanup to ensure we see them
    print(f"[Fallback time in rank {rank}]: average time = {sum(latencies_ref) / len(latencies_ref):.3f} ms, detail lists = {[f'{x:.3f}' for x in latencies_ref]}", flush=True)
    print(f"[Symm ops time in rank {rank}]: average time = {sum(latencies) / len(latencies):.3f} ms, detail lists = {[f'{x:.3f}' for x in latencies]}", flush=True)

    if enable_profile:
        if rank == 0:
            print(prof.key_averages().table(sort_by="self_xpu_time_total"))
        prof.export_chrome_trace("./profile_kineto_trace_reduce_scatter_M" + str(M) + "_N" + str(N) + "_K" + str(K) + "_rank" + str(rank) + ".json")

    #print(output_0)
    #print(output_1)
    # assert torch.allclose(output_0, output_1)
    # assert output_0.stride() == output_1.stride()

    import sys
    import gc
    import time
    
    # Cleanup tensors
    del A, B, output_0, output_1
    del begin_events_ref, end_events_ref, begin_events, end_events
    gc.collect()
    
    dist.destroy_process_group()
    print(f"[Rank {rank}] Process group destroyed", flush=True)
    sys.stdout.flush()
    time.sleep(0.1)
    
    # Use os._exit() to avoid ISHMEM/PyTorch cleanup conflicts
    os._exit(0)


rank = dist.get_rank()
world_size = dist.get_world_size()
test_matmul_reducescatter(rank, world_size)

