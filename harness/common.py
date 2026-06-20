#!/usr/bin/env python3
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import torch


HIDX = 64
DIDX = 128
PAGE_SIZE = 64


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    max_pages: int

    @property
    def max_seq_len(self) -> int:
        return self.max_pages * PAGE_SIZE


DOCUMENTED_CASES = {
    1: Case("testcase_1", 1, 8),
    2: Case("testcase_2", 2, 16),
    3: Case("testcase_3", 4, 32),
    4: Case("testcase_4", 8, 64),
    5: Case("testcase_5", 16, 128),
    6: Case("testcase_6", 32, 128),
    7: Case("testcase_7", 32, 256),
    8: Case("testcase_8", 64, 128),
    9: Case("testcase_9", 64, 256),
    10: Case("testcase_10", 64, 512),
}


@dataclass
class Inputs:
    q_idx: torch.Tensor
    k_idx_cache: torch.Tensor
    w_idx: torch.Tensor
    block_table: torch.Tensor
    context_lens: torch.Tensor
    scores: torch.Tensor


class Kernel:
    def __init__(self, path: str):
        lib_path = str(Path(path).resolve())
        self.library = ctypes.CDLL(lib_path)
        self.run = self.library.run_kernel
        ptr = ctypes.c_void_p
        self.run.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr] + [ctypes.c_int64] * 5
        self.run.restype = None

    def __call__(self, inputs: Inputs, case: Case) -> None:
        self.run(
            inputs.q_idx.data_ptr(),
            inputs.k_idx_cache.data_ptr(),
            inputs.w_idx.data_ptr(),
            inputs.block_table.data_ptr(),
            inputs.context_lens.data_ptr(),
            inputs.scores.data_ptr(),
            case.batch,
            HIDX,
            DIDX,
            case.max_pages,
            PAGE_SIZE,
        )


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-capable PyTorch installation and GPU are required")


def _unordered_block_table(
    batch: int, max_pages: int, num_pages: int, generator: torch.Generator
) -> torch.Tensor:
    rows = []
    for b in range(batch):
        row = torch.randperm(num_pages, generator=generator)[:max_pages]
        if max_pages > 1 and torch.equal(row, torch.arange(max_pages)):
            row = row.roll(1)
        rows.append(row.to(torch.int32))
    return torch.stack(rows).cuda()


def make_inputs(
    case: Case,
    seed: int,
    context_lens: list[int] | None = None,
) -> Inputs:
    require_cuda()
    cpu_gen = torch.Generator(device="cpu").manual_seed(seed)
    cuda_gen = torch.Generator(device="cuda").manual_seed(seed)
    max_seq_len = case.max_seq_len
    num_pages = case.max_pages + max(17, case.max_pages // 4)

    q_idx = (torch.randn(
        (case.batch, HIDX, DIDX), device="cuda", generator=cuda_gen
    ) * 0.125).to(torch.bfloat16)
    k_idx_cache = (torch.randn(
        (num_pages, PAGE_SIZE, DIDX), device="cuda", generator=cuda_gen
    ) * 0.125).to(torch.bfloat16)
    w_idx = torch.randn(
        (case.batch, HIDX), device="cuda", generator=cuda_gen
    ).to(torch.bfloat16)
    w_idx[:, 0] = -1.0
    w_idx[:, 1] = 1.0
    block_table = _unordered_block_table(
        case.batch, case.max_pages, num_pages, cpu_gen
    )

    if context_lens is None:
        # Include unequal lengths and incomplete pages while keeping official cases
        # representative of their maximum sequence length.
        lens = []
        for b in range(case.batch):
            if b == 0:
                visible = max_seq_len
            elif b == 1:
                visible = max_seq_len - 1
            else:
                low = max(1, max_seq_len // 2)
                visible = int(torch.randint(
                    low, max_seq_len + 1, (1,), generator=cpu_gen
                ).item())
                if visible % PAGE_SIZE == 0:
                    visible -= 1
            lens = lens + [visible]
        context_lens = lens
    if len(context_lens) != case.batch:
        raise ValueError("context_lens length must equal batch size")
    if any(v < 0 or v > max_seq_len for v in context_lens):
        raise ValueError("context length outside [0, MaxSeqLen]")

    lens_tensor = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
    scores = torch.randn(
        (case.batch, max_seq_len), device="cuda", generator=cuda_gen
    ).to(torch.bfloat16)
    return Inputs(q_idx, k_idx_cache, w_idx, block_table, lens_tensor, scores)


@torch.no_grad()
def reference(inputs: Inputs, case: Case) -> torch.Tensor:
    result = torch.full_like(inputs.scores, -float("inf"))
    for b in range(case.batch):
        visible = int(inputs.context_lens[b].item())
        if visible == 0:
            continue
        logical_positions = torch.arange(visible, device="cuda")
        logical_pages = torch.div(logical_positions, PAGE_SIZE, rounding_mode="floor")
        offsets = logical_positions.remainder(PAGE_SIZE)
        physical_pages = inputs.block_table[b].long()[logical_pages]
        keys = inputs.k_idx_cache[physical_pages, offsets, :].float()
        dots = inputs.q_idx[b].float() @ keys.T
        activated = torch.relu(dots)
        values = (activated * inputs.w_idx[b].float().reshape(HIDX, 1)).sum(dim=0)
        result[b, :visible] = values.to(torch.bfloat16)
    return result


def synchronize() -> None:
    torch.cuda.synchronize()
