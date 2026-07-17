"""
Experiment 1: Warmup Stabilization

Part A: Shows that running without JMH warmup (-wi 0) and discarding the first
        measurement iteration produces results similar to a proper warmup run
        (JMH-level granularity).

Part B: Shows that IQR filtering applied to the raw invocation-level execution
        times (collected via JIB tracing) removes warm-up outliers and converges
        to a mean close to the agent+warmup JMH baseline.
"""

import glob
import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP1_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results", "exp1")
PART1_DIR = os.path.join(_EXP1_DIR, "part1")
PART2_DIR = os.path.join(_EXP1_DIR, "part2")
TRACES_DIR = os.path.join(PART2_DIR, "traces")


def load_raw_data(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data[0]["primaryMetric"]["rawData"]


def flatten(nested):
    return [v for fork in nested for v in fork]


def stats(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    cv = (std / mean) * 100 if mean != 0 else 0
    return {"mean": mean, "std": std, "cv": cv, "n": n}


def print_stats(label, s, unit="ns/op"):
    print(f"  {label}:")
    print(f"    N = {s['n']:,}, Mean = {s['mean']:.2f} {unit}, "
          f"Std = {s['std']:.2f}, CV = {s['cv']:.2f}%")


def exp1a():
    warmup_file = os.path.join(PART1_DIR, "with_warmup.json")
    no_warmup_file = os.path.join(PART1_DIR, "no_warmup.json")

    for f in [warmup_file, no_warmup_file]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found. Run the experiment first.",
                  file=sys.stderr)
            sys.exit(1)

    warmup_raw = load_raw_data(warmup_file)
    no_warmup_raw = load_raw_data(no_warmup_file)

    warmup_all = flatten(warmup_raw)
    no_warmup_all = flatten(no_warmup_raw)
    no_warmup_trimmed = flatten([fork[1:] for fork in no_warmup_raw])

    s_warmup = stats(warmup_all)
    s_no_warmup = stats(no_warmup_all)
    s_trimmed = stats(no_warmup_trimmed)

    print("=" * 65)
    print("Experiment 1, Part A: JMH-level Warmup Stabilization")
    print("=" * 65)
    print()

    print_stats("With warmup (3 wi, all 5 iterations)", s_warmup)
    print()
    print_stats("No warmup (all 5 iterations, raw)", s_no_warmup)
    print()
    print_stats("No warmup (first iteration removed, 4 remaining)", s_trimmed)
    print()

    pct_diff_raw = abs(s_no_warmup["mean"] - s_warmup["mean"]) / s_warmup["mean"] * 100
    pct_diff_trimmed = abs(s_trimmed["mean"] - s_warmup["mean"]) / s_warmup["mean"] * 100

    print("-" * 65)
    print("Comparison to warmup baseline:")
    print(f"  Raw no-warmup vs warmup:     {pct_diff_raw:.2f}% difference in mean")
    print(f"  Trimmed no-warmup vs warmup: {pct_diff_trimmed:.2f}% difference in mean")
    print()
    print(f"  Raw no-warmup CV:     {s_no_warmup['cv']:.2f}%")
    print(f"  Trimmed no-warmup CV: {s_trimmed['cv']:.2f}%")
    print(f"  Warmup CV:            {s_warmup['cv']:.2f}%")
    print()


def _parse_log_file(log_file):
    """
    Parse a single JIB trace log (hash or plain) and return a list of
    individual method invocation durations in nanoseconds.
    """
    times = []
    current_start = None

    with open(log_file, 'r', buffering=8 * 1024 * 1024) as f:
        for line in f:
            try:
                ts_end = line.index(']')
                timestamp = int(line[1:ts_end])
                event = line[ts_end + 2]
                if event == 'S':
                    current_start = timestamp
                elif event == 'E' and current_start is not None:
                    times.append(timestamp - current_start)
                    current_start = None
            except (ValueError, IndexError):
                continue

    return times


def load_jib_traces(results_dir, base_name):
    """
    Collect invocation times from all JIB trace files matching
    <base_name>_*.log in results_dir (one file per JMH fork).

    Returns (list[int] of durations in ns, int number of files loaded).
    """
    pattern = os.path.join(results_dir, f"{base_name}_*.log")
    log_files = sorted(glob.glob(pattern))

    all_times = []
    non_empty = 0
    for lf in log_files:
        fork_times = _parse_log_file(lf)
        if fork_times:
            all_times.extend(fork_times)
            non_empty += 1

    return all_times, non_empty


def apply_iqr_filter(values):
    """Return values with IQR outliers removed (standard 1.5xIQR rule)."""
    if len(values) < 4:
        return values
    sv = sorted(values)
    n = len(sv)
    q1 = sv[n // 4]
    q3 = sv[3 * n // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [v for v in values if lo <= v <= hi]


def exp1b():
    baseline_file = os.path.join(PART2_DIR, "with_warmup.json")
    no_warmup_file = os.path.join(PART2_DIR, "no_warmup.json")

    for fp in [baseline_file, no_warmup_file]:
        if not os.path.exists(fp):
            print(f"  [Part B] Skipping: {fp} not found. Run Part B first.",
                  file=sys.stderr)
            return

    # JMH iteration-level stats: agent + warmup (baseline)
    s_baseline = stats(flatten(load_raw_data(baseline_file)))

    # JMH iteration-level stats: agent + no warmup
    s_jmh_nw = stats(flatten(load_raw_data(no_warmup_file)))

    # Invocation-level trace: agent + no warmup
    trace_times, num_forks = load_jib_traces(TRACES_DIR, "nowarmup_trace")
    if not trace_times:
        print("  [Part B] Skipping: no trace files found. Run Part B first.",
              file=sys.stderr)
        return

    filtered = apply_iqr_filter(trace_times)
    s_raw = stats(trace_times)
    s_filt = stats(filtered)

    removed = len(trace_times) - len(filtered)
    removed_pct = removed / len(trace_times) * 100

    pct_jmh_nw = (abs(s_jmh_nw["mean"] - s_baseline["mean"])
                  / s_baseline["mean"] * 100)
    pct_raw = abs(s_raw["mean"] - s_baseline["mean"]) / s_baseline["mean"] * 100
    pct_filt = abs(s_filt["mean"] - s_baseline["mean"]) / s_baseline["mean"] * 100

    print("=" * 65)
    print("Experiment 1, Part B: Trace-based IQR Filtering")
    print("=" * 65)
    print()
    print_stats("JIB baseline  (agent + warmup,    -wi 3 -i 5 -f 3)", s_baseline)
    print()
    print_stats("JMH no-warmup (agent + no warmup, -wi 0 -i 5 -f 3)", s_jmh_nw)
    print()

    print(f"  Invocation-level trace  (agent + no warmup, {num_forks} fork(s)):")
    print(f"    Total invocations:   {len(trace_times):,}")
    print_stats("  Raw trace", s_raw, unit="ns")
    print_stats("  IQR-filtered trace", s_filt, unit="ns")
    print(f"    Removed by IQR:      {removed:,} ({removed_pct:.2f}% of total)")
    print()

    print("-" * 65)
    print("Comparison to agent+warmup JMH baseline:")
    print(f"  JMH no-warmup vs baseline:           {pct_jmh_nw:.2f}% diff in mean")
    print(f"  Trace raw vs baseline:               {pct_raw:.2f}% diff in mean")
    print(f"  Trace IQR-filtered vs baseline:      {pct_filt:.2f}% diff in mean")
    print()
    print(f"  Raw trace CV:        {s_raw['cv']:.2f}%")
    print(f"  IQR-filtered CV:     {s_filt['cv']:.2f}%")
    print(f"  JIB baseline CV:     {s_baseline['cv']:.2f}%")
    print()


def main():
    exp1a()
    exp1b()


if __name__ == "__main__":
    main()
