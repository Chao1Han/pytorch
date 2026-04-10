#!/bin/bash

set -e

LOG_DIR="perf_logs"
mkdir -p "$LOG_DIR"

# M values: 8192 down to 1024, step -1024
M_VALUES=(8192 7168 6144 5120 4096 3072 2048 1024)

# Allgather configs: N K
AG_N=(1536 7168 2560 12800)
AG_K=(4096 4096 5120 5120)

# Reduce_scatter configs: N K
RS_N=(4096 4096 5120 5120)
RS_K=(1024 3584 2048 6400)

NUM_RANKS=4

run_allgather() {
    local barrier_mode=$1  # "signal_barrier" or "no_signal_barrier"
    local len=${#AG_N[@]}

    for ((i=0; i<$len; i++)); do
        N=${AG_N[$i]}
        K=${AG_K[$i]}
        for M in "${M_VALUES[@]}"; do
            local logfile="${LOG_DIR}/allgather_${barrier_mode}_M${M}_N${N}_K${K}.log"
            echo "=== Allgather [${barrier_mode}] M=$M, N=$N, K=$K ==="
            mpirun -np $NUM_RANKS --prepend-rank python test_fused_allgather_matmul.py $M $N $K 2>&1 | tee "$logfile"
            sleep 1
        done
    done
}

run_reducescatter() {
    local barrier_mode=$1
    local len=${#RS_N[@]}

    for ((i=0; i<$len; i++)); do
        N=${RS_N[$i]}
        K=${RS_K[$i]}
        for M in "${M_VALUES[@]}"; do
            local logfile="${LOG_DIR}/reducescatter_${barrier_mode}_M${M}_N${N}_K${K}.log"
            echo "=== ReduceScatter [${barrier_mode}] M=$M, N=$N, K=$K ==="
            mpirun -np $NUM_RANKS --prepend-rank python test_fused_matmul_reducescatter.py $M $N $K 2>&1 | tee "$logfile"
            sleep 1
        done
    done
}

echo "=========================================="
echo "  Phase 1: USE_SIGNAL_BARRIER=1"
echo "=========================================="
export USE_SIGNAL_BARRIER=1

run_allgather "signal_barrier"
run_reducescatter "signal_barrier"

echo "=========================================="
echo "  Phase 2: unset USE_SIGNAL_BARRIER"
echo "=========================================="
unset USE_SIGNAL_BARRIER

run_allgather "no_signal_barrier"
run_reducescatter "no_signal_barrier"

echo "=========================================="
echo "  All runs complete. Parsing results..."
echo "=========================================="

python parse_perf_logs.py

echo "Done! Results saved to perf_results.csv"
