import itertools
import os
from contextlib import nullcontext
from unittest import skip, skipIf

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import (
    _fused_scaled_matmul_reduce_scatter_fallback,
    _test_mode,
    enable_symm_mem_for_group,
)
import os
import time
import statistics
import torch.multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description="test_scaled_matmul_reducescatter")
parser.add_argument("M", type=int, default=8192, help="M value")
parser.add_argument("N", type=int, default=4096, help="N value")
parser.add_argument("K", type=int, default=7168, help="K value")
parser.add_argument("--rowwise", action="store_true", default=False,
                    help="Use row-wise scaling instead of tensor-wise scaling")

args = parser.parse_args()

print("M = ", args.M, flush=True)
print("N = ", args.N, flush=True)
print("K = ", args.K, flush=True)
print("rowwise = ", args.rowwise, flush=True)

BATCH = 1
M = args.M
N = args.N
K = args.K
rowwise = args.rowwise
Loop = 20
WARMUP = 10
enable_profile = False

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29823'
dist.init_process_group(backend='xccl')


# Now we can set the backend
import torch.distributed._symmetric_memory as symm_mem
group_name = dist.group.WORLD.group_name
symm_mem.enable_symm_mem_for_group(group_name)

# Determine FP8 type
e4m3_type = torch.float8_e4m3fn

def test_scaled_matmul_reducescatter(rank, world_size):
    torch.xpu.set_device(rank)
    torch.use_deterministic_algorithms(True, warn_only=True)

    group = dist.group.WORLD
    scatter_dim = 0

    torch.manual_seed(42 + rank)
    A = torch.rand(M, K, device="xpu", dtype=torch.bfloat16).to(e4m3_type)
    B = torch.rand(N, K, device="xpu", dtype=torch.bfloat16).to(e4m3_type).T

    if rowwise:
        A_scale = torch.full((M, 1), 0.1, device="xpu")
        B_scale = torch.full((1, N), 0.1, device="xpu")
    else:
        A_scale = torch.tensor(0.1, device="xpu")
        B_scale = torch.tensor(0.1, device="xpu")

    output_shape = [*A.shape[:-1], B.shape[1]]

    print(f"[Rank {rank}] Initialized, A.shape={A.shape}, B.shape={B.shape}, rowwise={rowwise}", flush=True)

    TIMED = Loop - WARMUP

    begin_events_ref = [torch.xpu.Event(enable_timing=True) for _ in range(TIMED)]
    end_events_ref = [torch.xpu.Event(enable_timing=True) for _ in range(TIMED)]
    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(TIMED)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(TIMED)]

    if enable_profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
    else:
        prof = nullcontext()

    # -------- Fallback (scaled_gemm + reduce_scatter) --------
    for i in range(WARMUP):
        output_0 = _fused_scaled_matmul_reduce_scatter_fallback(
            A, B, A_scale, B_scale,
            "sum", scatter_dim, scatter_dim, group.group_name,
            output_shape,
            out_dtype=torch.bfloat16,
        )
    torch.xpu.synchronize()
    call_times_ref = []
    for i in range(TIMED):
        begin_events_ref[i].record()
        output_0 = _fused_scaled_matmul_reduce_scatter_fallback(
            A, B, A_scale, B_scale,
            "sum", scatter_dim, scatter_dim, group.group_name,
            output_shape,
            out_dtype=torch.bfloat16,
        )
        end_events_ref[i].record()
    torch.xpu.synchronize()
    print("zl_debug start to call torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter", flush=True)

    # -------- Symm ops (scaled_gemm + reduce_scatter) --------
    for i in range(WARMUP):
        output_1 = torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
            A, B, A_scale, B_scale,
            "sum", scatter_dim, scatter_dim, group.group_name,
            output_shape,
            out_dtype=torch.bfloat16,
        )
    torch.xpu.synchronize()
    call_times_symm = []
    with prof:
        for i in range(TIMED):
            begin_events[i].record()
            output_1 = torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
                A, B, A_scale, B_scale,
                "sum", scatter_dim, scatter_dim, group.group_name,
                output_shape,
                out_dtype=torch.bfloat16,
            )
            end_events[i].record()
        torch.xpu.synchronize()

    # -------- Print results --------
    latencies_ref = [b.elapsed_time(e) for b, e in zip(begin_events_ref, end_events_ref)]
    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]

    print(f"[Fallback time in rank {rank}]: average time = {sum(latencies_ref) / len(latencies_ref)} detail lists = {latencies_ref} ms")
    print(f"[Symm ops time in rank {rank}]: average time = {sum(latencies) / len(latencies)} detail lists =  {latencies} ms")

    if enable_profile:
        if rank == 0:
            print(prof.key_averages().table(sort_by="self_xpu_time_total"))
        prof.export_chrome_trace("./profile_kineto_trace_scaled_reduce_scatter_M" + str(M) + "_N" + str(N) + "_K" + str(K) + "_rank" + str(rank) + ".json")

    dist.destroy_process_group()
    print(f"[Fallback time in rank {rank}]: average time = {sum(latencies_ref) / len(latencies_ref)} detail lists = {latencies_ref} ms")
    print(f"[Symm ops time in rank {rank}]: average time = {sum(latencies) / len(latencies)} detail lists =  {latencies} ms")



rank = dist.get_rank()
world_size = dist.get_world_size()
test_scaled_matmul_reducescatter(rank, world_size)
