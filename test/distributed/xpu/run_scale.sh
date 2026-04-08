#!/bin/bash
export TORCH_SYMMMEM=XPU

LLAMA_8B_M=(2048 2048)
LLAMA_8B_N=(1536 3584)
LLAMA_8B_K=(4096 4096)

SCALE_MODE="tensor-wise"  # 可选: tensor-wise, row-wise-replicated, row-wise-sharded

len=${#LLAMA_8B_M[@]}

for ((i=0; i<$len; i++)); do
    M=${LLAMA_8B_M[$i]}
    N=${LLAMA_8B_N[$i]}
    K=${LLAMA_8B_K[$i]}

    echo "Running fused_all_gather_scaled_matmul with M=$M, N=$N, K=$K, scale_mode=$SCALE_MODE"
    mpirun -np 4 --prepend-rank python test_allgather_scaled_matmul.py $M $N $K --scale_mode $SCALE_MODE
    sleep 1
done

LLAMA_8B_M_1=(8192 8192)
LLAMA_8B_N_1=(4096 4096)
LLAMA_8B_K_1=(1024 1792)

ROWWISE=""  # 设为 "--rowwise" 启用 row-wise scaling

len1=${#LLAMA_8B_M_1[@]}

for ((i=0; i<$len1; i++)); do
    M=${LLAMA_8B_M_1[$i]}
    N=${LLAMA_8B_N_1[$i]}
    K=${LLAMA_8B_K_1[$i]}

    echo "Running fused_scaled_matmul_reduce_scatter with M=$M, N=$N, K=$K, rowwise=${ROWWISE:-false}"
    mpirun -np 4 --prepend-rank python test_fused_scaled_matmul_reducescatter.py $M $N $K $ROWWISE
    sleep 1
done
