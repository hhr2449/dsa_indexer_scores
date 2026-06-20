# DeepSeek Sparse Attention Indexer Score

Read `docs/task.md` as the source of truth.

Target hardware: NVIDIA H800.

Implement the CUDA submission in:

src/kernel.cu

Preserve the exact required:

extern "C" void run_kernel(...)

Evaluation constants are:

Hidx = 64
Didx = 128
PageSize = 64

For every valid logical token s:

logical_page = s / PageSize
page_offset = s % PageSize
physical_page = block_table[b * MaxPages + logical_page]

score[b, s] =
    sum over head j:
        float(w_idx[b, j])
        *
        max(
            dot(
                q_idx[b, j, :],
                k_idx_cache[physical_page, page_offset, :]
            ),
            0
        )

Use FP32 for dot products and score accumulation.
Write the final result as BF16.

For every invalid position:

scores[b, s] = BF16(-inf)

Requirements:

- Do not assume physical_page == logical_page.
- Support unordered and non-contiguous block_table.
- Support different context_lens for different batch items.
- Support context_lens[b] == 0.
- Do not read or write out of bounds.
- Do not print anything.
- Do not access external files.
- Do not change the public ABI.
- Optimize specifically for NVIDIA H800.
- The result must exceed 2x speedup over the OJ baseline except testcase 1.

There is no local evaluator.

The user will submit src/kernel.cu to the OJ and provide compilation,
correctness and performance feedback.

For each iteration:

1. Inspect the current src/kernel.cu.
2. Implement one complete performance-oriented candidate.
3. Locally compile it only to catch syntax and linking errors.
4. Do not build a local test harness.
5. Stop after producing the compilable submission file.
6. Summarize the kernel design and expected performance characteristics.
7. Wait for the user to provide OJ feedback before making the next candidate.
