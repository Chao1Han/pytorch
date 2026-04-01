#!/bin/bash
export TORCH_SYMMMEM=XPU

# ============================================================
# fused_all_gather_scaled_matmul benchmark
# ============================================================

# 定义数组（注意 Bash 的数组语法）
#LLAMA_8B_M=(4096 4096)
#LLAMA_8B_N=(3072 14336)
#LLAMA_8B_K=(4096 4096)

# tp=4
#LLAMA_8B_M=(2048 2048)
#LLAMA_8B_N=(1536 7168)
#LLAMA_8B_K=(4096 4096)

# tp=4 with seq=4096
LLAMA_8B_M=(2048 2048 512 512 256 256 128 128 32 32 8 8 2 2) 
LLAMA_8B_N=(1536 7168 1536 7168 1536 7168 1536 7168 1536 7168 1536 7168 1536 7168) 
LLAMA_8B_K=(4096 4096 4096 4096 4096 4096 4096 4096 4096 4096 4096 4096 4096 4096)

# tp=8
# LLAMA_8B_M=(1024 1024)
# LLAMA_8B_N=(768 3584)
# LLAMA_8B_K=(4096 4096)

SCALE_MODE="tensor-wise"  # 可选: tensor-wise, row-wise-replicated, row-wise-sharded

# 获取数组长度
len=${#LLAMA_8B_M[@]}

# 遍历数组索引
for ((i=0; i<$len; i++)); do
    M=${LLAMA_8B_M[$i]}
    N=${LLAMA_8B_N[$i]}
    K=${LLAMA_8B_K[$i]}

    echo "Running fused_all_gather_scaled_matmul with M=$M, N=$N, K=$K, scale_mode=$SCALE_MODE"
    mpirun -np 4 --prepend-rank python test_allgather_scaled_matmul.py $M $N $K --scale_mode $SCALE_MODE
    sleep 1
done

# ============================================================
# fused_scaled_matmul_reduce_scatter benchmark
# ============================================================

# 8B, tp=2, matmul+reducescatter
#LLAMA_8B_M_1=(8192 8192)
#LLAMA_8B_N_1=(4096 4096)
#LLAMA_8B_K_1=(2048 7168)

# tp=4
# LLAMA_8B_M_1=(8192 8192)
# LLAMA_8B_N_1=(4096 4096)
# LLAMA_8B_K_1=(1024 3584)

# tp=4 with seq = 1024
LLAMA_8B_M_1=(8192 8192 2048 2048 1024 1024 512 512 128 128)
LLAMA_8B_N_1=(4096 4096 4096 4096 4096 4096 4096 4096 4096 4096)
LLAMA_8B_K_1=(1024 3584 1024 3584 1024 3584 1024 3584 1024 3584)

ROWWISE=""  # 设为 "--rowwise" 启用 row-wise scaling

# 获取第二个数组的长度
len1=${#LLAMA_8B_M_1[@]}

for ((i=0; i<$len1; i++)); do
    M=${LLAMA_8B_M_1[$i]}
    N=${LLAMA_8B_N_1[$i]}
    K=${LLAMA_8B_K_1[$i]}

    echo "Running fused_scaled_matmul_reduce_scatter with M=$M, N=$N, K=$K, rowwise=${ROWWISE:-false}"
    mpirun -np 4 --prepend-rank python test_fused_scaled_matmul_reducescatter.py $M $N $K $ROWWISE
    sleep 1
done
