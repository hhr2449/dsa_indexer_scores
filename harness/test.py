#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback

import torch

from common import Case, DOCUMENTED_CASES, Kernel, make_inputs, reference, require_cuda, synchronize


def check_result(name: str, inputs, expected: torch.Tensor) -> None:
    actual = inputs.scores
    max_seq_len = actual.shape[1]
    positions = torch.arange(max_seq_len, device="cuda").reshape(1, -1)
    valid = positions < inputs.context_lens.reshape(-1, 1)
    invalid = ~valid

    if invalid.any():
        invalid_values = actual[invalid]
        exact_negative_inf = torch.isinf(invalid_values) & (invalid_values < 0)
        if not bool(exact_negative_inf.all().item()):
            bad = int((~exact_negative_inf).sum().item())
            raise AssertionError(f"{name}: {bad} invalid outputs are not exact -inf")

    if valid.any():
        actual_valid = actual[valid].float()
        expected_valid = expected[valid].float()
        close = torch.isclose(actual_valid, expected_valid, atol=0.01, rtol=0.01)
        if not bool(close.all().item()):
            diff = (actual_valid - expected_valid).abs()
            worst = int(diff.argmax().item())
            raise AssertionError(
                f"{name}: valid output mismatch: bad={(~close).sum().item()}, "
                f"max_abs={diff[worst].item():.6g}, "
                f"actual={actual_valid[worst].item():.6g}, "
                f"expected={expected_valid[worst].item():.6g}"
            )


def run_one(kernel: Kernel, case: Case, seed: int, context_lens=None) -> None:
    inputs = make_inputs(case, seed, context_lens)
    expected = reference(inputs, case)
    kernel(inputs, case)
    synchronize()
    check_result(case.name, inputs, expected)
    lens = inputs.context_lens.cpu().tolist()
    print(
        f"PASS {case.name}: B={case.batch} MaxPages={case.max_pages} "
        f"contexts=[{min(lens)}, {max(lens)}]"
    )
    del inputs, expected
    torch.cuda.empty_cache()


def run_boundary_cases(kernel: Kernel) -> None:
    boundary = Case("boundaries_and_mapping", 8, 3)
    run_one(kernel, boundary, 1000, [0, 1, 63, 64, 65, 127, 191, 192])

    # Repeated random cases make assumptions about contiguous or identity physical
    # pages very unlikely to survive validation.
    for i in range(3):
        random_case = Case(f"random_mapping_{i + 1}", 5, 5 + i)
        run_one(kernel, random_case, 1100 + i)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a DSA indexer CUDA library")
    parser.add_argument("--lib", required=True, help="path to shared library")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="run all local tests")
    group.add_argument("--case", type=int, choices=range(1, 11), help="run one documented testcase")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    kernel = Kernel(args.lib)
    if args.case is not None:
        run_one(kernel, DOCUMENTED_CASES[args.case], 2000 + args.case)
    else:
        run_boundary_cases(kernel)
        for case_id, case in DOCUMENTED_CASES.items():
            run_one(kernel, case, 2000 + case_id)
        print("PASS all correctness tests")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
