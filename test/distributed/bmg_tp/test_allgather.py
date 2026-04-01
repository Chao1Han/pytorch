import itertools
import os
from contextlib import nullcontext
from unittest import skip, skipIf

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._symmetric_memory import (
    _fused_all_gather_matmul_fallback,
    _fused_all_gather_scaled_matmul_fallback,
    _fused_matmul_reduce_scatter_fallback,
    _test_mode,
    enable_symm_mem_for_group,
    restride_A_for_fused_matmul_reduce_scatter,
    restride_A_shard_for_fused_all_gather_matmul,
)
import os
import torch.multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description="test_symm")
parser.add_argument("M", type=int, default=4096, help="M value")
parser.add_argument("N", type=int, default=1792, help="N value")
parser.add_argument("K", type=int, default=4096, help="K value")

args = parser.parse_args()

print("M = ", args.M)
print("N = ", args.N)
print("K = ", args.K)

BATCH = 1
M = args.M # 512 #4096
N = args.N # 128 #1792
K = args.K # 128 #4096
Loop = 20
enable_profile = False

# CRITICAL: Set environment variables BEFORE importing torch.distributed
# os.environ['TORCH_SYMMMEM'] = 'ISHMEM'

# Get rank from MPI environment
rank = int(os.environ.get('PMI_RANK', os.environ.get('OMPI_COMM_WORLD_RANK', 0)))
world_size = int(os.environ.get('PMI_SIZE', os.environ.get('OMPI_COMM_WORLD_SIZE', 1)))

# Set ZE_AFFINITY_MASK for device binding (critical for ISHMEM)
os.environ['ZE_AFFINITY_MASK'] = str(rank)

os.environ['RANK'] = str(rank)
os.environ['WORLD_SIZE'] = str(world_size)
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29503'

# Initialize process group - this triggers ISHMEM registration
dist.init_process_group(backend='xccl', rank=rank, world_size=world_size)

# symm_mem.set_backend("XPU")
# symm_mem.set_backend("ISHMEM")
print("Now register backend:", symm_mem.get_backend("xpu"))
group_name = dist.group.WORLD.group_name
symm_mem.enable_symm_mem_for_group(group_name)

def test_allgather_matmul(rank, world_size):
    # Device is always 0 after ZE_AFFINITY_MASK filtering
    torch.xpu.set_device(0)

    group = dist.group.WORLD

    torch.manual_seed(42 + rank)
    A_shard = torch.rand(M, K, device="xpu", dtype=torch.bfloat16)
    Bs = [torch.rand(K, N, device="xpu", dtype=torch.bfloat16) for _ in range(1)]

    begin_events_ref = [
        torch.xpu.Event(enable_timing=True) for _ in range(Loop-5)
    ]
    end_events_ref = [torch.xpu.Event(enable_timing=True) for _ in range(Loop-5)]

    begin_events = [
        torch.xpu.Event(enable_timing=True) for _ in range(Loop-5)
    ]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(Loop-5)]

    if enable_profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
    else:
        prof = nullcontext()

    with prof:
        # warm up fallback aten ops
        for i in range(10):
            ag_output_0, mm_outputs_0 = _fused_all_gather_matmul_fallback(
                A_shard, Bs, gather_dim=0, group_name=group.group_name
            )
        torch.xpu.synchronize()
        for i in range(Loop):
            if i >= 5:
                begin_events_ref[i-5].record()
            ag_output_0, mm_outputs_0 = _fused_all_gather_matmul_fallback(
                A_shard, Bs, gather_dim=0, group_name=group.group_name
            )
            if i >= 5:
                end_events_ref[i-5].record()
        torch.xpu.synchronize()

        # warm up symmetric ops
        try:
            for i in range(10):
                ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_matmul(
                    A_shard, Bs, gather_dim=0, group_name=group.group_name
                )
        except Exception as e:
            print(f"[Rank {rank}] ERROR during warm-up: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        torch.xpu.synchronize()
        print(f"[Rank {rank}] Warm-up completed, starting timed runs...", flush=True)
        for i in range(Loop):
            if i >= 5:
                begin_events[i-5].record()
            ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_matmul(
                A_shard, Bs, gather_dim=0, group_name=group.group_name
            )
            if i >= 5:
                end_events[i-5].record()
        
        torch.xpu.synchronize()

    print(f"[Rank {rank}] Calculating latencies...", flush=True)
    latencies_ref = [b.elapsed_time(e) for b, e in zip(begin_events_ref, end_events_ref)]
    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]

    # Print results BEFORE cleanup to ensure we see them
    print(f"[Fallback time in rank {rank}]: average time = {sum(latencies_ref) / len(latencies_ref):.3f} ms, detail lists = {[f'{x:.3f}' for x in latencies_ref]}", flush=True)
    print(f"[Symm ops time in rank {rank}]: average time = {sum(latencies) / len(latencies):.3f} ms, detail lists = {[f'{x:.3f}' for x in latencies]}", flush=True)

    if enable_profile:
        print(f"[Rank {rank}] Exporting profile...", flush=True)
        prof.export_chrome_trace("./profile_kineto_trace_allgather_M" + str(M) + "_N" + str(N) + "_K" + str(K) + "_rank" + str(rank) + ".json")
        print(f"[Rank {rank}] Profile exported", flush=True)

    # assert torch.allclose(ag_output_0, ag_output_1)
    # assert ag_output_0.stride() == ag_output_1.stride()
    # for mm_output_0, mm_output_1 in zip(mm_outputs_0, mm_outputs_1):
    #     assert torch.allclose(mm_output_0, mm_output_1)
    #     assert mm_output_0.stride(), mm_output_1.stride()

    import sys
    import gc
    import time
    
    # Cleanup tensors
    del A_shard, Bs, ag_output_0, mm_outputs_0, ag_output_1, mm_outputs_1
    del begin_events_ref, end_events_ref, begin_events, end_events
    gc.collect()
    
    dist.destroy_process_group()
    print(f"[Rank {rank}] Process group destroyed", flush=True)
    sys.stdout.flush()
    time.sleep(0.1)
    
    # Use os._exit() to avoid ISHMEM/PyTorch cleanup conflicts
    os._exit(0)


rank = dist.get_rank()
size = dist.get_world_size()
test_allgather_matmul(rank, size)

