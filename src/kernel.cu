#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

constexpr int kHeads = 64;
constexpr int kDim = 128;
constexpr int kPageSize = 64;
constexpr int kThreads = 128;
constexpr int kVectorsPerMatrix = kHeads * kDim / 8;

__device__ __forceinline__ void cp_async_16(void* smem_ptr,
                                            const void* gmem_ptr) {
    const uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void prefetch_k_page(
    uint4* k_s, const __nv_bfloat16* k_idx_cache, int physical_page) {
    const uint4* const k_g = reinterpret_cast<const uint4*>(
        k_idx_cache + physical_page * kPageSize * kDim);
    for (int src = threadIdx.x; src < kVectorsPerMatrix;
         src += kThreads) {
        const int row = src >> 4;
        const int vec = src & 15;
        const int dst = (vec >> 1) * 128 + (vec & 1) * 64 + row;
        cp_async_16(k_s + dst, k_g + src);
    }
    cp_async_commit();
}

// Unsizzled GMMA Major-K descriptor.  A 64x16 BF16 tile is stored as two
// 8-element K planes, with 64 contiguous 16-byte row vectors per plane.
__device__ __forceinline__ uint64_t make_gmma_desc(const void* ptr) {
    const uint64_t addr =
        static_cast<uint64_t>(__cvta_generic_to_shared(ptr));
    constexpr uint64_t kLeading = 64;  // 1024 bytes / 16
    constexpr uint64_t kStride = 8;    // 128 bytes / 16
    return ((addr >> 4) & 0x3fffull) |
           (kLeading << 16) | (kStride << 32);
}

struct Accumulator {
    float x[32];
};

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_wait() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n64k16(
    Accumulator& d, uint64_t desc_a, uint64_t desc_b, int accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,p,1,1,0,0;\n"
        "}\n"
        : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3]),
          "+f"(d.x[4]), "+f"(d.x[5]), "+f"(d.x[6]), "+f"(d.x[7]),
          "+f"(d.x[8]), "+f"(d.x[9]), "+f"(d.x[10]), "+f"(d.x[11]),
          "+f"(d.x[12]), "+f"(d.x[13]), "+f"(d.x[14]), "+f"(d.x[15]),
          "+f"(d.x[16]), "+f"(d.x[17]), "+f"(d.x[18]), "+f"(d.x[19]),
          "+f"(d.x[20]), "+f"(d.x[21]), "+f"(d.x[22]), "+f"(d.x[23]),
          "+f"(d.x[24]), "+f"(d.x[25]), "+f"(d.x[26]), "+f"(d.x[27]),
          "+f"(d.x[28]), "+f"(d.x[29]), "+f"(d.x[30]), "+f"(d.x[31])
        : "l"(desc_a), "l"(desc_b), "r"(accumulate));
}

template <int kPagesPerCta>
__global__ __launch_bounds__(kThreads, 6)
void dsa_indexer_scores_wgmma(
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
    const int pages_this_cta = min(kPagesPerCta, max_pages - first_page);
    const int visible = __ldg(context_lens + b);

    if (first_page * kPageSize >= visible) {
        __nv_bfloat16* const out =
            scores + (b * max_pages + first_page) * kPageSize;
        for (int i = threadIdx.x; i < pages_this_cta * kPageSize;
             i += kThreads) {
            out[i] = __float2bfloat16(-INFINITY);
        }
        return;
    }

    __shared__ __align__(128) uint4 q_s[kVectorsPerMatrix];
    __shared__ __align__(128) uint4 k_s[kVectorsPerMatrix];
    __shared__ __align__(16) __nv_bfloat16 w_s[kHeads];
    __shared__ __align__(16) float partial_s[4][kPageSize];

    const uint4* const q_g = reinterpret_cast<const uint4*>(
        q_idx + b * kHeads * kDim);
    // Convert row-major [64,128] into eight GMMA K-major [64,16] tiles.
    for (int src = threadIdx.x; src < kVectorsPerMatrix; src += kThreads) {
        const int row = src >> 4;
        const int vec = src & 15;
        const int dst = (vec >> 1) * 128 + (vec & 1) * 64 + row;
        cp_async_16(q_s + dst, q_g + src);
    }
    cp_async_commit();
    if (threadIdx.x < kHeads) {
        w_s[threadIdx.x] = w_idx[b * kHeads + threadIdx.x];
    }

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

#pragma unroll
    for (int page_in_cta = 0; page_in_cta < kPagesPerCta; ++page_in_cta) {
        if (page_in_cta >= pages_this_cta) break;
        const int logical_page = first_page + page_in_cta;
        const int page_begin = logical_page * kPageSize;
        __nv_bfloat16* const out =
            scores + (b * max_pages + logical_page) * kPageSize;

        if (page_begin >= visible) {
            const int remaining = pages_this_cta - page_in_cta;
            for (int i = threadIdx.x; i < remaining * kPageSize;
                 i += kThreads) {
                out[i] = __float2bfloat16(-INFINITY);
            }
            return;
        }

        // Page zero is loaded here.  Every later valid page was prefetched by
        // the previous iteration while its accumulator epilogue was running.
        if (page_in_cta == 0) {
            const int physical_page =
                block_table[b * max_pages + logical_page];
            prefetch_k_page(k_s, k_idx_cache, physical_page);
        }
        cp_async_wait_all();
        __syncthreads();

        Accumulator acc;
#pragma unroll
        for (int i = 0; i < 32; ++i) acc.x[i] = 0.0f;

        wgmma_fence();
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            const uint64_t desc_q = make_gmma_desc(q_s + kt * 128);
            const uint64_t desc_k = make_gmma_desc(k_s + kt * 128);
            wgmma_m64n64k16(acc, desc_q, desc_k, kt != 0);
        }
        wgmma_commit_wait();

        // WGMMA has finished reading k_s.  Reuse that buffer immediately for
        // the next page and overlap the copy with this page's register
        // reduction, shared partial reduction, and output store.
        const int next_page_in_cta = page_in_cta + 1;
        if (next_page_in_cta < pages_this_cta &&
            page_begin + kPageSize < visible) {
            const int next_logical_page = logical_page + 1;
            const int next_physical_page =
                block_table[b * max_pages + next_logical_page];
            prefetch_k_page(k_s, k_idx_cache, next_physical_page);
        }

        const int row0 = warp * 16 + (lane >> 2);
        const float w0 = __bfloat162float(w_s[row0]);
        const float w1 = __bfloat162float(w_s[row0 + 8]);
#pragma unroll
        for (int ng = 0; ng < 8; ++ng) {
            const int r = ng * 4;
            float s0 = fmaxf(acc.x[r], 0.0f) * w0 +
                       fmaxf(acc.x[r + 2], 0.0f) * w1;
            float s1 = fmaxf(acc.x[r + 1], 0.0f) * w0 +
                       fmaxf(acc.x[r + 3], 0.0f) * w1;
#pragma unroll
            for (int delta = 16; delta >= 4; delta >>= 1) {
                s0 += __shfl_down_sync(0xffffffffu, s0, delta);
                s1 += __shfl_down_sync(0xffffffffu, s1, delta);
            }
            if (lane < 4) {
                const int col = ng * 8 + lane * 2;
                partial_s[warp][col] = s0;
                partial_s[warp][col + 1] = s1;
            }
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
        if (page_in_cta + 1 < pages_this_cta) __syncthreads();
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
    if (blocks64 <= 0) return;

    const int max_pages = static_cast<int>(MaxPages);
    if (blocks64 >= 16384) {
        constexpr int kPages = 16;
        const int64_t groups_per_batch = (MaxPages + kPages - 1) / kPages;
        const unsigned int blocks =
            static_cast<unsigned int>(B * groups_per_batch);
        dsa_indexer_scores_wgmma<kPages><<<blocks, kThreads>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    } else if (blocks64 >= 4096) {
        constexpr int kPages = 8;
        const int64_t groups_per_batch = (MaxPages + kPages - 1) / kPages;
        const unsigned int blocks =
            static_cast<unsigned int>(B * groups_per_batch);
        dsa_indexer_scores_wgmma<kPages><<<blocks, kThreads>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    } else if (blocks64 >= 2048) {
        constexpr int kPages = 4;
        const int64_t groups_per_batch = (MaxPages + kPages - 1) / kPages;
        const unsigned int blocks =
            static_cast<unsigned int>(B * groups_per_batch);
        dsa_indexer_scores_wgmma<kPages><<<blocks, kThreads>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    } else {
        const unsigned int blocks = static_cast<unsigned int>(blocks64);
        dsa_indexer_scores_wgmma<1><<<blocks, kThreads>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    }
}
