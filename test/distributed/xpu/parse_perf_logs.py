#!/usr/bin/env python3
"""Parse perf log files and generate CSV summary."""

import os
import re
import csv
import glob


LOG_DIR = "perf_logs"
OUTPUT_CSV = "perf_results.csv"

# Pattern: [Fallback time in rank 0]: average time = 4.495 detail lists = [...] ms
# Pattern: [Symm ops time in rank 0]: average time = 3.570 detail lists = [...] ms
RE_FALLBACK = re.compile(
    r"\[Fallback time in rank (\d+)\]: average time = ([\d.]+)"
)
RE_SYMM = re.compile(
    r"\[Symm ops time in rank (\d+)\]: average time = ([\d.]+)"
)

# Log filename pattern:
# allgather_signal_barrier_M8192_N1536_K4096.log
# reducescatter_no_signal_barrier_M8192_N4096_K1024.log
RE_FILENAME = re.compile(
    r"(allgather|reducescatter)_(signal_barrier|no_signal_barrier)_M(\d+)_N(\d+)_K(\d+)\.log"
)


def parse_log(filepath):
    """Parse a single log file. Return dict with fallback/symm avg times for rank 0."""
    fallback_times = {}
    symm_times = {}
    with open(filepath, "r") as f:
        for line in f:
            m = RE_FALLBACK.search(line)
            if m:
                rank = int(m.group(1))
                avg_time = float(m.group(2))
                fallback_times[rank] = avg_time
            m = RE_SYMM.search(line)
            if m:
                rank = int(m.group(1))
                avg_time = float(m.group(2))
                symm_times[rank] = avg_time

    # Use rank 0 as representative
    return {
        "fallback_avg_ms": fallback_times.get(0, None),
        "symm_avg_ms": symm_times.get(0, None),
    }


def main():
    rows = []

    for logfile in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        basename = os.path.basename(logfile)
        m = RE_FILENAME.match(basename)
        if not m:
            print(f"Skipping unrecognized file: {basename}")
            continue

        op_type = m.group(1)  # allgather / reducescatter
        barrier_mode = m.group(2)  # signal_barrier / no_signal_barrier
        M = int(m.group(3))
        N = int(m.group(4))
        K = int(m.group(5))

        result = parse_log(logfile)

        rows.append({
            "op_type": op_type,
            "barrier_mode": barrier_mode,
            "M": M,
            "N": N,
            "K": K,
            "fallback_avg_ms": result["fallback_avg_ms"],
            "symm_avg_ms": result["symm_avg_ms"],
        })

    # Sort: op_type, N, K, barrier_mode, M descending
    rows.sort(key=lambda r: (r["op_type"], r["N"], r["K"], r["barrier_mode"], -r["M"]))

    # Write CSV
    fieldnames = ["op_type", "barrier_mode", "M", "N", "K", "fallback_avg_ms", "symm_avg_ms"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    # Also write a pivot-style CSV for easier comparison
    # Group by (op_type, N, K, M) with columns for signal_barrier and no_signal_barrier
    pivot_csv = "perf_results_pivot.csv"
    pivot_map = {}
    for r in rows:
        key = (r["op_type"], r["N"], r["K"], r["M"])
        if key not in pivot_map:
            pivot_map[key] = {}
        pivot_map[key][r["barrier_mode"]] = r

    pivot_fields = [
        "op_type", "M", "N", "K",
        "fallback_avg_ms(signal_barrier)", "symm_avg_ms(signal_barrier)",
        "fallback_avg_ms(no_signal_barrier)", "symm_avg_ms(no_signal_barrier)",
    ]
    pivot_rows = []
    for (op_type, N, K, M) in sorted(pivot_map.keys(), key=lambda k: (k[0], k[1], k[2], -k[3])):
        entry = pivot_map[(op_type, N, K, M)]
        sb = entry.get("signal_barrier", {})
        nsb = entry.get("no_signal_barrier", {})
        pivot_rows.append({
            "op_type": op_type,
            "M": M,
            "N": N,
            "K": K,
            "fallback_avg_ms(signal_barrier)": sb.get("fallback_avg_ms", "") if isinstance(sb, dict) else "",
            "symm_avg_ms(signal_barrier)": sb.get("symm_avg_ms", "") if isinstance(sb, dict) else "",
            "fallback_avg_ms(no_signal_barrier)": nsb.get("fallback_avg_ms", "") if isinstance(nsb, dict) else "",
            "symm_avg_ms(no_signal_barrier)": nsb.get("symm_avg_ms", "") if isinstance(nsb, dict) else "",
        })

    with open(pivot_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pivot_fields)
        writer.writeheader()
        writer.writerows(pivot_rows)

    print(f"Wrote {len(pivot_rows)} rows to {pivot_csv}")


if __name__ == "__main__":
    main()
