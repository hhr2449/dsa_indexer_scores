#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

from common import DOCUMENTED_CASES, Kernel, make_inputs, require_cuda, synchronize


WARMUP = 30
ITERATIONS = 100


def benchmark(kernel: Kernel, case_id: int):
    case = DOCUMENTED_CASES[case_id]
    inputs = make_inputs(case, 2000 + case_id)
    for _ in range(WARMUP):
        kernel(inputs, case)
    synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERATIONS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERATIONS)]
    for start, end in zip(starts, ends):
        start.record()
        kernel(inputs, case)
        end.record()
    synchronize()
    times_us = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]
    result = {
        "testcase": case_id,
        "batch": case.batch,
        "max_pages": case.max_pages,
        "max_seq_len": case.max_seq_len,
        "minimum_us": min(times_us),
        "median_us": statistics.median(times_us),
        "average_us": statistics.fmean(times_us),
    }
    del inputs, starts, ends
    torch.cuda.empty_cache()
    return result


def record_csv(path: str, implementation: str, results: list[dict]) -> None:
    output = Path(path)
    fieldnames = [
        "timestamp_utc", "implementation", "testcase", "batch", "max_pages",
        "max_seq_len", "minimum_us", "median_us", "average_us",
    ]
    existing = []
    if output.exists():
        with output.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    replaced = {(implementation, str(row["testcase"])) for row in results}
    existing = [
        row for row in existing
        if (row["implementation"], row["testcase"]) not in replaced
    ]
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for result in results:
        existing.append({
            "timestamp_utc": timestamp,
            "implementation": implementation,
            **{key: result[key] for key in fieldnames if key in result},
        })
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a DSA indexer CUDA library")
    parser.add_argument("--lib", required=True, help="path to shared library")
    parser.add_argument("--csv", default="benchmark.csv", help="result CSV path")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="benchmark all documented testcases")
    group.add_argument("--case", type=int, choices=range(1, 11), help="benchmark one testcase")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_cuda()
    kernel = Kernel(args.lib)
    case_ids = list(DOCUMENTED_CASES) if args.all else [args.case]
    results = []
    print(f"warmup={WARMUP} iterations={ITERATIONS}")
    for case_id in case_ids:
        result = benchmark(kernel, case_id)
        results.append(result)
        print(
            f"testcase {case_id}: B={result['batch']} MaxSeqLen={result['max_seq_len']} "
            f"min={result['minimum_us']:.3f} us "
            f"median={result['median_us']:.3f} us "
            f"avg={result['average_us']:.3f} us"
        )
    implementation = Path(args.lib).stem.removeprefix("lib")
    record_csv(args.csv, implementation, results)
    print(f"recorded {len(results)} result(s) in {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
