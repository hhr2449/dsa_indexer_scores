#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void dsa_indexer_scores_baseline_kernel(
    const __nv_bfloat16* __restrict__ q_idx,
    const __nv_bfloat16* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ w_idx,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens,
    __nv_bfloat16* __restrict__ scores,
    int64_t B,
    int64_t Hidx,
    int64_t Didx,
    int64_t MaxPages,
    int64_t PageSize
) {
    const int64_t MaxSeqLen = MaxPages * PageSize;
    const int64_t total = B * MaxSeqLen;

    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {
        const int64_t s = linear % MaxSeqLen;
        const int64_t b = linear / MaxSeqLen;

        const int64_t score_offset = b * MaxSeqLen + s;
        const int32_t visible = context_lens[b];

        if (s >= (int64_t)visible) {
            scores[score_offset] = __float2bfloat16(-INFINITY);
            continue;
        }

        const int64_t logical_page = s / PageSize;
        const int64_t page_offset = s - logical_page * PageSize;
        const int32_t physical_page = block_table[b * MaxPages + logical_page];

        float acc = 0.0f;

        for (int64_t j = 0; j < Hidx; ++j) {
            const int64_t q_base = (b * Hidx + j) * Didx;
            const int64_t k_base = ((int64_t)physical_page * PageSize + page_offset) * Didx;

            float dot = 0.0f;
            for (int64_t d = 0; d < Didx; ++d) {
                const float q = __bfloat162float(q_idx[q_base + d]);
                const float k = __bfloat162float(k_idx_cache[k_base + d]);
                dot += q * k;
            }

            if (dot > 0.0f) {
                const float w = __bfloat162float(w_idx[b * Hidx + j]);
                acc += w * dot;
            }
        }

        scores[score_offset] = __float2bfloat16(acc);
    }
}

extern "C" void run_kernel(
    const __nv_bfloat16* q_idx,
    const __nv_bfloat16* k_idx_cache,
    const __nv_bfloat16* w_idx,
    const int32_t* block_table,
    const int32_t* context_lens,
    __nv_bfloat16* scores,
    int64_t B,
    int64_t Hidx,
    int64_t Didx,
    int64_t MaxPages,
    int64_t PageSize
) {
    const int64_t MaxSeqLen = MaxPages * PageSize;
    const int64_t total = B * MaxSeqLen;
    if (total <= 0) {
        return;
    }

    const int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    if (blocks64 > 65535) {
        blocks64 = 65535;
    }
    const int blocks = (int)blocks64;

    dsa_indexer_scores_baseline_kernel<<<blocks, threads>>>(
        q_idx,
        k_idx_cache,
        w_idx,
        block_table,
        context_lens,
        scores,
        B,
        Hidx,
        Didx,
        MaxPages,
        PageSize
    );
}
