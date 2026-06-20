#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace {

constexpr int kHeads = 64;
constexpr int kDim = 128;
constexpr int kPageSize = 64;
constexpr int kTile = 16;

// One CTA computes one complete logical page.  The 16 warps form a 4x4
// (head, token) grid of 16x16 Tensor Core output tiles.
__global__ __launch_bounds__(512, 2)
void dsa_indexer_scores_page_kernel(
    const __nv_bfloat16* __restrict__ q_idx,
    const __nv_bfloat16* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ w_idx,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens,
    __nv_bfloat16* __restrict__ scores,
    int64_t MaxPages) {
    const int page_linear = blockIdx.x;
    const int b = page_linear / static_cast<int>(MaxPages);
    const int logical_page = page_linear - b * static_cast<int>(MaxPages);
    const int page_begin = logical_page * kPageSize;
    const int visible = context_lens[b];
    __nv_bfloat16* const out =
        scores + (static_cast<int64_t>(b) * MaxPages + logical_page) * kPageSize;

    // Crucially, this path does not read block_table for an invalid logical page.
    if (page_begin >= visible) {
        if (threadIdx.x < kPageSize) {
            out[threadIdx.x] = __float2bfloat16(-INFINITY);
        }
        return;
    }

    __shared__ __align__(16) __nv_bfloat16 q_s[kHeads * kDim];
    __shared__ __align__(16) __nv_bfloat16 k_s[kPageSize * kDim];
    __shared__ __align__(16) float dots[kHeads * kPageSize];

    const int physical_page =
        block_table[static_cast<int64_t>(b) * MaxPages + logical_page];
    const __nv_bfloat16* const q_g =
        q_idx + static_cast<int64_t>(b) * kHeads * kDim;
    const __nv_bfloat16* const k_g =
        k_idx_cache + static_cast<int64_t>(physical_page) * kPageSize * kDim;

    // CUDA tensor allocations and all page/batch strides are at least 16-byte
    // aligned. Each thread moves two 16-byte vectors from each operand.
    const uint4* const q_g16 = reinterpret_cast<const uint4*>(q_g);
    const uint4* const k_g16 = reinterpret_cast<const uint4*>(k_g);
    uint4* const q_s16 = reinterpret_cast<uint4*>(q_s);
    uint4* const k_s16 = reinterpret_cast<uint4*>(k_s);
#pragma unroll
    for (int i = threadIdx.x; i < (kHeads * kDim * 2) / 16; i += blockDim.x) {
        q_s16[i] = q_g16[i];
        k_s16[i] = k_g16[i];
    }
    __syncthreads();

    using namespace nvcuda;
    const int warp = threadIdx.x >> 5;
    const int head_tile = warp >> 2;
    const int token_tile = warp & 3;

    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
    wmma::fill_fragment(acc, 0.0f);

#pragma unroll
    for (int d = 0; d < kDim; d += kTile) {
        wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                       __nv_bfloat16,
                       wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                       __nv_bfloat16,
                       wmma::col_major> bt;
        wmma::load_matrix_sync(a, q_s + head_tile * kTile * kDim + d, kDim);
        // k_s is [token, d] row-major. The same bytes are a [d, token]
        // column-major matrix with leading dimension kDim.
        wmma::load_matrix_sync(bt, k_s + token_tile * kTile * kDim + d, kDim);
        wmma::mma_sync(acc, a, bt, acc);
    }
    wmma::store_matrix_sync(
        dots + head_tile * kTile * kPageSize + token_tile * kTile,
        acc, kPageSize, wmma::mem_row_major);
    __syncthreads();

    if (threadIdx.x < kPageSize) {
        const int token = threadIdx.x;
        if (page_begin + token < visible) {
            const __nv_bfloat16* const wb = w_idx + b * kHeads;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < kHeads; ++h) {
                const float dot = dots[h * kPageSize + token];
                if (dot > 0.0f) {
                    score = fmaf(__bfloat162float(wb[h]), dot, score);
                }
            }
            out[token] = __float2bfloat16(score);
        } else {
            out[token] = __float2bfloat16(-INFINITY);
        }
    }
}

}  // namespace

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
    int64_t PageSize) {
    (void)Hidx;
    (void)Didx;
    (void)PageSize;
    const int64_t blocks64 = B * MaxPages;
    if (blocks64 <= 0) {
        return;
    }
    dsa_indexer_scores_page_kernel<<<static_cast<unsigned int>(blocks64), 512>>>(
        q_idx, k_idx_cache, w_idx, block_table, context_lens, scores, MaxPages);
}
