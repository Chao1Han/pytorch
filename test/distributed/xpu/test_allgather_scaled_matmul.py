import itertools
import os
from contextlib import nullcontext
from unittest import skip, skipIf

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._symmetric_memory import (
    _fused_all_gather_scaled_matmul_fallback,
    _test_mode,
    enable_symm_mem_for_group,
    restride_A_shard_for_fused_all_gather_matmul,
)
import os
import time
import statistics
import torch.multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description="test_allgather_scaled_matmul")
parser.add_argument("M", type=int, default=4096, help="M value")
parser.add_argument("N", type=int, default=1792, help="N value")
parser.add_argument("K", type=int, default=4096, help="K value")
parser.add_argument("--scale_mode", type=str, default="tensor-wise",
                    choices=["tensor-wise", "row-wise-replicated", "row-wise-sharded"],
                    help="Scale mode for FP8 scaling")

args = parser.parse_args()

print("M = ", args.M, flush=True)
print("N = ", args.N, flush=True)
print("K = ", args.K, flush=True)
print("scale_mode = ", args.scale_mode, flush=True)

BATCH = 1
M = args.M
N = args.N
K = args.K
scale_mode = args.scale_mode
Loop = 20
enable_profile = False

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29823'
dist.init_process_group(backend='xccl')

print("Now register backend:", symm_mem.get_backend("xpu"))
group_name = dist.group.WORLD.group_name
symm_mem.enable_symm_mem_for_group(group_name)

# Determine FP8 type
e4m3_type = torch.float8_e4m3fn

def test_allgather_scaled_matmul(rank, world_size):
    torch.xpu.set_device(rank)
    torch.use_deterministic_algorithms(True, warn_only=True)

    group = dist.group.WORLD
    gather_dim = 0

    torch.manual_seed(42 + rank)
    A_shard = torch.rand(M, K, device="xpu", dtype=torch.bfloat16).to(e4m3_type)
    Bs = [torch.rand(N, K, device="xpu", dtype=torch.bfloat16).to(e4m3_type).T for _ in range(1)]

    if scale_mode == "tensor-wise":
        A_scale = torch.tensor(0.1, device="xpu")
        B_scales = [torch.tensor(0.1, device="xpu") for _ in range(len(Bs))]
        out_dtypes = [torch.bfloat16] * len(Bs)
    elif scale_mode == "row-wise-sharded":
        A_scale = torch.full((M, 1), 0.1, device="xpu")
        B_scales = [torch.full((1, N), 0.1, device="xpu") for _ in range(len(Bs))]
        out_dtypes = [torch.bfloat16] * len(Bs)
    elif scale_mode == "row-wise-replicated":
        A_scale = torch.full((M * world_size, 1), 0.1, device="xpu")
        B_scales = [torch.full((1, N), 0.1, device="xpu") for _ in range(len(Bs))]
        out_dtypes = [torch.bfloat16] * len(Bs)
    else:
        raise ValueError(f"Invalid scale_mode: {scale_mode}")

    print(f"[Rank {rank}] Initialized, A_shard.shape={A_shard.shape}, Bs[0].shape={Bs[0].shape}, scale_mode={scale_mode}", flush=True)

    TIMED = Loop - 5

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

    with prof:
        # -------- Fallback (allgather + scaled_gemm) --------
        for i in range(10):
            ag_output_0, mm_outputs_0 = _fused_all_gather_scaled_matmul_fallback(
                A_shard, Bs, A_scale, B_scales,
                gather_dim=gather_dim, group_name=group.group_name,
                biases=[None] * len(Bs),
                result_scales=[None] * len(Bs),
                out_dtypes=out_dtypes,
                use_fast_accum=[False] * len(Bs),
            )
        torch.xpu.synchronize()
        call_times_ref = []
        for i in range(Loop):
            if i >= 5:
                begin_events_ref[i - 5].record()
            t0 = time.perf_counter()
            ag_output_0, mm_outputs_0 = _fused_all_gather_scaled_matmul_fallback(
                A_shard, Bs, A_scale, B_scales,
                gather_dim=gather_dim, group_name=group.group_name,
                biases=[None] * len(Bs),
                result_scales=[None] * len(Bs),
                out_dtypes=out_dtypes,
                use_fast_accum=[False] * len(Bs),
            )
            if i >= 5:
                call_times_ref.append((time.perf_counter() - t0) * 1000)
                end_events_ref[i - 5].record()
        torch.xpu.synchronize()

        # -------- Symm ops (allgather + scaled_gemm) --------
        try:
            for i in range(10):
                ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
                    A_shard, Bs, A_scale, B_scales,
                    gather_dim=gather_dim, group_name=group.group_name,
                    biases=[None] * len(Bs),
                    result_scales=[None] * len(Bs),
                    out_dtypes=out_dtypes,
                    use_fast_accum=[False] * len(Bs),
                )
        except Exception as e:
            print(f"[Rank {rank}] ERROR during warm-up: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        torch.xpu.synchronize()
        print(f"[Rank {rank}] Warm-up completed, starting timed runs...", flush=True)
        call_times_symm = []
        for i in range(Loop):
            if i >= 5:
                begin_events[i - 5].record()
            ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
                A_shard, Bs, A_scale, B_scales,
                gather_dim=gather_dim, group_name=group.group_name,
                biases=[None] * len(Bs),
                result_scales=[None] * len(Bs),
                out_dtypes=out_dtypes,
                use_fast_accum=[False] * len(Bs),
            )
            if i >= 5:
                end_events[i - 5].record()

        torch.xpu.synchronize()


    # -------- Print results --------
    latencies_ref = [b.elapsed_time(e) for b, e in zip(begin_events_ref, end_events_ref)]
    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]

    if enable_profile:
        print(f"[Rank {rank}] Exporting profile...", flush=True)
        prof.export_chrome_trace("./profile_kineto_trace_allgather_scaled_matmul_M" + str(M) + "_N" + str(N) + "_K" + str(K) + "_rank" + str(rank) + ".json")
        print(f"[Rank {rank}] Profile exported", flush=True)

    dist.destroy_process_group()
    print(f"[Fallback time in rank {rank}]: average time = {sum(latencies_ref) / len(latencies_ref)} detail lists = {latencies_ref} ms")
    print(f"[Symm ops time in rank {rank}]: average time = {sum(latencies) / len(latencies)} detail lists =  {latencies} ms")



rank = dist.get_rank()
size = dist.get_world_size()
test_allgather_scaled_matmul(rank, size)
