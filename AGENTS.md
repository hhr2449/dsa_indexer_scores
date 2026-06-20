# DeepSeek Sparse Attention CUDA Kernel Task

## Objective

Implement and optimize the BF16 Indexer Scores operator for DeepSeek Sparse Attention on an NVIDIA H800 GPU.

The final implementation must preserve the required exported `run_kernel` C ABI, pass all correctness tests, and maximize measured performance using the official evaluator.

Read `docs/task.md` and all repository files before modifying code.

## Mathematical operation

For each batch item `b` and logical token position `s`:

```text
if s >= context_lens[b]:
    scores[b, s] = BF16(-inf)
else:
    logical_page  = s / PageSize
    page_offset   = s % PageSize
    physical_page = block_table[b, logical_page]

    key = k_idx_cache[physical_page, page_offset, :]

    scores[b, s] =
        sum over head j:
            float(w_idx[b, j])
            *
            max(
                dot(
                    float(q_idx[b, j, :]),
                    float(key)
                ),
                0
            )
```

Evaluation uses:

```text
Hidx     = 64
Didx     = 128
PageSize = 64
```

Inputs and output are BF16. Dot products and final score accumulation must use FP32.

## Correctness constraints

* Preserve the exact required `extern "C" run_kernel` signature.
* Explicitly write BF16 negative infinity to every invalid output position.
* Support `context_lens[b] == 0`.
* Do not assume all batch items have the same context length.
* Always access the key cache through `block_table`.
* Do not assume `physical_page == logical_page`.
* Support non-contiguous and unordered physical pages.
* Do not read or write out of bounds.
* Do not print from `run_kernel`.
* Do not access external files from `run_kernel`.
* Meet the required absolute or relative error tolerance.
* Do not hard-code `B` or `MaxSeqLen`.
* A specialized fast path for `(Hidx, Didx, PageSize) = (64, 128, 64)` is allowed, but the public interface and parameter handling must remain correct.

## Required first actions

1. Inspect every build, validation and benchmark script.
2. Identify the exact commands for:

   * compilation;
   * full correctness validation;
   * individual testcase execution;
   * full performance benchmarking.
3. Run the unmodified baseline once.
4. Save baseline correctness output under `runs/`.
5. Save baseline benchmark output under `runs/`.
6. Write the discovered commands into `docs/commands.md`.

Do not begin performance claims before recording the baseline.

## Implementation direction

Start from a performance-oriented page-tiled fused design rather than the baseline one-thread-per-score organization.

For one batch item and one logical page:

```text
Q          = 64 x 128 BF16
K_page     = 64 x 128 BF16
Dot matrix = Q x transpose(K_page)
           = 64 x 64 FP32 accumulators
```

Then fuse:

```text
ReLU
head weighting
reduction over 64 heads
BF16 conversion
score stores
```

Do not materialize the full intermediate dot matrix in global memory.

Investigate and measure:

* CTA, warp or warp-group mapping;
* query reuse across tokens;
* key reuse across heads;
* cached head weights;
* vectorized and coalesced BF16 loads;
* shared-memory layouts and bank conflicts;
* BF16 Tensor Core MMA or Hopper WGMMA;
* register pressure;
* shared-memory consumption;
* occupancy;
* instruction and memory throughput;
* final partial-page handling;
* invalid-position stores.

Do not preserve an optimization unless it is both correct and measurably faster.

## Workflow

Maintain:

```text
docs/draft.md
docs/plan.md
docs/commands.md
benchmark.csv
candidates.jsonl
runs/
profile/
candidates/
```

For every meaningful candidate:

1. Save or identify the source revision.
2. Compile it.
3. Run full correctness validation.
4. If correct, benchmark it.
5. Record timing and speedup in `benchmark.csv`.
6. Record design notes and status in `candidates.jsonl`.
7. Keep the candidate only if it improves a relevant measured result.

Use NCU on representative large cases, especially testcases 5, 7 and 10. Store reports and summaries under `profile/`.

Use the installed `ncu-report-skill` and `KernelWiki` skills when profiling or selecting CUDA optimization techniques.

## Performance priorities

Primary targets:

```text
testcase 5
testcase 7
testcase 10
```

The implementation must exceed the assignment's speedup threshold wherever required.

Use the official evaluator as the source of truth. Do not rely on synthetic microbenchmarks alone.

## Completion criteria

The task is complete only when:

* all correctness tests pass;
* invalid positions are exactly BF16 negative infinity;
* arbitrary `block_table` mappings pass;
* no CUDA error or out-of-bounds access occurs;
* measured benchmark results are recorded;
* testcases 5, 7 and 10 have final timing and speedup data;
* profiling evidence explains the final design;
* the final source retains the exact required interface;
* `docs/final-report-notes.md` summarizes:

  * kernel organization;
  * thread and block mapping;
  * memory hierarchy usage;
  * Tensor Core or WGMMA usage;
  * testcase 5, 7 and 10 results;
  * remaining bottlenecks.

Do not stop after writing a plan. Continue through implementation, validation, benchmarking and profiling.
