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
constexpr int kOutputTiles = (kHeads / kTile) * (kPageSize / kTile);

__device__ __forceinline__ void cp_async_16(void* smem_ptr,
                                            const void* gmem_ptr) {
    const uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit_wait() {
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// A page is a 64x128 by 128x64 BF16 GEMM.  Eight warps each produce two
// output tiles.  This keeps the landing storage below the static shared-memory
// limit and permits four resident CTAs per SM.  Each warp consumes its own
// WMMA result immediately, so the full 64x64 FP32 matrix never crosses a
// CTA-wide synchronization point.
template <int kWarps, int kPagesPerCta>
__global__ __launch_bounds__(kWarps * 32, 32 / kWarps)
void dsa_indexer_scores_page_kernel(
    const __nv_bfloat16* __restrict__ q_idx,
    const __nv_bfloat16* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ w_idx,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens,
    __nv_bfloat16* __restrict__ scores,
    int max_pages) {
    const int groups_per_batch =
        (max_pages + kPagesPerCta - 1) / kPagesPerCta;
    const int group_linear = static_cast<int>(blockIdx.x);
    const int b = group_linear / groups_per_batch;
    const int group_in_batch = group_linear - b * groups_per_batch;
    const int first_page = group_in_batch * kPagesPerCta;
    const int pages_this_cta =
        min(kPagesPerCta, max_pages - first_page);
    const int visible = __ldg(context_lens + b);

    // Never touch block_table or the cache for a wholly masked page group.
    if (first_page * kPageSize >= visible) {
        __nv_bfloat16* const out =
            scores + (b * max_pages + first_page) * kPageSize;
        for (int i = threadIdx.x; i < pages_this_cta * kPageSize;
             i += kWarps * 32) {
            out[i] = __float2bfloat16(-INFINITY);
        }
        return;
    }

    __shared__ __align__(16) __nv_bfloat16 q_s[kHeads * kDim];
    __shared__ __align__(16) __nv_bfloat16 k_s[kPageSize * kDim];
    __shared__ __align__(16) __nv_bfloat16 w_s[kHeads];
    // One reusable 16x16 FP32 landing tile per producer warp.
    __shared__ __align__(16) float tile_s[kWarps][kTile * kTile];
    // Partial score after reducing each group of 16 heads.
    __shared__ __align__(16) float partial_s[kHeads / kTile][kPageSize];

    const __nv_bfloat16* const q_g = q_idx + b * kHeads * kDim;

    const uint4* const q_g16 = reinterpret_cast<const uint4*>(q_g);
    uint4* const q_s16 = reinterpret_cast<uint4*>(q_s);
    uint4* const k_s16 = reinterpret_cast<uint4*>(k_s);
    constexpr int kVectors = (kHeads * kDim * sizeof(__nv_bfloat16)) /
                             sizeof(uint4);
    for (int i = threadIdx.x; i < kVectors; i += kWarps * 32) {
        cp_async_16(q_s16 + i, q_g16 + i);
    }
    if (threadIdx.x < kHeads) {
        w_s[threadIdx.x] = w_idx[b * kHeads + threadIdx.x];
    }

    using namespace nvcuda;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // Four-page CTAs amortize the query copy, weight copy, block scheduling,
    // and one synchronization over four independent paged-cache lookups.
#pragma unroll
    for (int page_in_cta = 0; page_in_cta < kPagesPerCta; ++page_in_cta) {
        if (page_in_cta >= pages_this_cta) {
            break;
        }
        const int logical_page = first_page + page_in_cta;
        const int page_begin = logical_page * kPageSize;
        __nv_bfloat16* const out =
            scores + (b * max_pages + logical_page) * kPageSize;

        // Once one page is wholly masked, every following page in this group
        // is masked too.  Fill all of them without reading block_table.
        if (page_begin >= visible) {
            const int remaining = pages_this_cta - page_in_cta;
            for (int i = threadIdx.x; i < remaining * kPageSize;
                 i += kWarps * 32) {
                out[i] = __float2bfloat16(-INFINITY);
            }
            return;
        }

        const int physical_page =
            block_table[b * max_pages + logical_page];
        const __nv_bfloat16* const k_g =
            k_idx_cache + physical_page * (kPageSize * kDim);
        const uint4* const k_g16 = reinterpret_cast<const uint4*>(k_g);
        for (int i = threadIdx.x; i < kVectors; i += kWarps * 32) {
            cp_async_16(k_s16 + i, k_g16 + i);
        }
        cp_async_commit_wait();
        __syncthreads();

#pragma unroll
        for (int output_tile = warp; output_tile < kOutputTiles;
             output_tile += kWarps) {
            const int head_tile = output_tile >> 2;
            const int token_tile = output_tile & 3;

            wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
            wmma::fill_fragment(acc, 0.0f);

#pragma unroll
            for (int d = 0; d < kDim; d += kTile) {
                wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                               __nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                               __nv_bfloat16, wmma::col_major> bt;
                wmma::load_matrix_sync(
                    a, q_s + head_tile * kTile * kDim + d, kDim);
                wmma::load_matrix_sync(
                    bt, k_s + token_tile * kTile * kDim + d, kDim);
                wmma::mma_sync(acc, a, bt, acc);
            }

            float* const warp_tile = tile_s[warp];
            wmma::store_matrix_sync(warp_tile, acc, kTile,
                                    wmma::mem_row_major);
            __syncwarp();

            if (lane < kTile) {
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < kTile; ++h) {
                    const float dot = warp_tile[h * kTile + lane];
                    if (dot > 0.0f) {
                        score = fmaf(
                            __bfloat162float(w_s[head_tile * kTile + h]),
                            dot, score);
                    }
                }
                partial_s[head_tile][token_tile * kTile + lane] = score;
            }
            __syncwarp();
        }

        __syncthreads();
        if (threadIdx.x < kPageSize) {
            const int token = threadIdx.x;
            if (page_begin + token < visible) {
                float score = partial_s[0][token] + partial_s[1][token];
                score += partial_s[2][token] + partial_s[3][token];
                out[token] = __float2bfloat16(score);
            } else {
                out[token] = __float2bfloat16(-INFINITY);
            }
        }
        if (page_in_cta + 1 < pages_this_cta) {
            __syncthreads();
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

    const int max_pages = static_cast<int>(MaxPages);
    if (blocks64 >= 16384) {
        constexpr int kPages = 4;
        const int64_t groups_per_batch = (MaxPages + kPages - 1) / kPages;
        const unsigned int blocks =
            static_cast<unsigned int>(B * groups_per_batch);
        dsa_indexer_scores_page_kernel<8, kPages><<<blocks, 256>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    } else {
        const unsigned int blocks = static_cast<unsigned int>(blocks64);
        dsa_indexer_scores_page_kernel<8, 1><<<blocks, 256>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    }
}
