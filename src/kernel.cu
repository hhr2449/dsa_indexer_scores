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

__device__ __forceinline__ void cp_async_wait_one() {
    asm volatile("cp.async.wait_group 1;\n" ::);
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

struct QueryFragments {
    uint32_t x[8][4];
};

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_wait() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n64k16_rs(
    Accumulator& d, const uint32_t* a, uint64_t desc_b, int accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35},%36,p,1,1,0;\n"
        "}\n"
        : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3]),
          "+f"(d.x[4]), "+f"(d.x[5]), "+f"(d.x[6]), "+f"(d.x[7]),
          "+f"(d.x[8]), "+f"(d.x[9]), "+f"(d.x[10]), "+f"(d.x[11]),
          "+f"(d.x[12]), "+f"(d.x[13]), "+f"(d.x[14]), "+f"(d.x[15]),
          "+f"(d.x[16]), "+f"(d.x[17]), "+f"(d.x[18]), "+f"(d.x[19]),
          "+f"(d.x[20]), "+f"(d.x[21]), "+f"(d.x[22]), "+f"(d.x[23]),
          "+f"(d.x[24]), "+f"(d.x[25]), "+f"(d.x[26]), "+f"(d.x[27]),
          "+f"(d.x[28]), "+f"(d.x[29]), "+f"(d.x[30]), "+f"(d.x[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b), "r"(accumulate));
}

template <int kPagesPerCta>
__global__ __launch_bounds__(kThreads, 4)
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

    constexpr int kStages =
        kPagesPerCta == 1 ? 1 : (kPagesPerCta == 4 ? 2 : 3);
    __shared__ __align__(128) uint4 k_s[kStages][kVectorsPerMatrix];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = warp * 16 + (lane >> 2);
    const float w0 = __bfloat162float(w_idx[b * kHeads + row0]);
    const float w1 = __bfloat162float(w_idx[b * kHeads + row0 + 8]);

    // Register fragment mapping specified by PTX for m64nNk16.  Each thread
    // owns two adjacent columns from rows row0 and row0+8 in each 16-wide K
    // tile.  Keeping all eight tiles resident removes Q from shared memory.
    QueryFragments q_frag;
    const int col_pair = (lane & 3) * 2;
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
        const int col = kt * 16 + col_pair;
        const __nv_bfloat16* const q0 =
            q_idx + ((b * kHeads + row0) * kDim + col);
        const __nv_bfloat16* const q1 = q0 + 8 * kDim;
        q_frag.x[kt][0] = *reinterpret_cast<const uint32_t*>(q0);
        q_frag.x[kt][1] = *reinterpret_cast<const uint32_t*>(q1);
        q_frag.x[kt][2] = *reinterpret_cast<const uint32_t*>(q0 + 8);
        q_frag.x[kt][3] = *reinterpret_cast<const uint32_t*>(q1 + 8);
    }

    const int valid_pages_this_cta = min(
        pages_this_cta,
        (visible - first_page * kPageSize + kPageSize - 1) / kPageSize);

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
            prefetch_k_page(k_s[0], k_idx_cache, physical_page);
        }
        if constexpr (kStages == 3) {
            if (page_in_cta == 0 ||
                page_in_cta + 1 >= valid_pages_this_cta) {
                cp_async_wait_all();
            } else {
                cp_async_wait_one();
            }
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        const int stage = page_in_cta % kStages;
        if constexpr (kStages == 3) {
            // Large groups keep two future pages in flight to cover random
            // paged-cache latency.
            const int first_prefetch =
                page_in_cta == 0 ? 1 : page_in_cta + 2;
            const int prefetch_count = page_in_cta == 0 ? 2 : 1;
#pragma unroll
            for (int pf = 0; pf < prefetch_count; ++pf) {
                const int target = first_prefetch + pf;
                if (target < valid_pages_this_cta) {
                    const int target_logical_page = first_page + target;
                    const int target_physical_page =
                        block_table[b * max_pages + target_logical_page];
                    prefetch_k_page(k_s[target % kStages], k_idx_cache,
                                    target_physical_page);
                }
            }
        } else {
            // Four-page groups favor the fifth resident CTA enabled by the
            // smaller two-stage shared-memory footprint.
            const int target = page_in_cta + 1;
            if (target < valid_pages_this_cta) {
                const int target_logical_page = first_page + target;
                const int target_physical_page =
                    block_table[b * max_pages + target_logical_page];
                prefetch_k_page(k_s[target % kStages], k_idx_cache,
                                target_physical_page);
            }
        }

        Accumulator acc;
#pragma unroll
        for (int i = 0; i < 32; ++i) acc.x[i] = 0.0f;

        wgmma_fence();
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            const uint64_t desc_k =
                make_gmma_desc(k_s[stage] + kt * 128);
            wgmma_m64n64k16_rs(acc, q_frag.x[kt], desc_k, kt != 0);
        }
        wgmma_commit_wait();

        // WGMMA is done with the current K stage.  Reuse its first 1 KiB for
        // the four-warp score reduction, keeping total static shared memory
        // at exactly 48 KiB for the double-buffered specializations.
        float (*partial_s)[kPageSize] =
            reinterpret_cast<float (*)[kPageSize]>(k_s[stage]);
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

__device__ __forceinline__ void named_barrier_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void named_barrier_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}

// A single producer warp copies one page.  All 32 operations are consumed at
// the same wait point, so keep them in one group and commit once per page.
__device__ __forceinline__ void producer_load_k_page(
    uint4* k_s, const __nv_bfloat16* k_idx_cache, int physical_page,
    int lane) {
    const uint4* const k_g = reinterpret_cast<const uint4*>(
        k_idx_cache + physical_page * kPageSize * kDim);
#pragma unroll
    for (int group = 0; group < 4; ++group) {
#pragma unroll
        for (int op = 0; op < 8; ++op) {
            const int src = lane + (group * 8 + op) * 32;
            const int row = src >> 4;
            const int vec = src & 15;
            const int dst = (vec >> 1) * 128 + (vec & 1) * 64 + row;
            cp_async_16(k_s + dst, k_g + src);
        }
    }
    cp_async_commit();
    cp_async_wait_all();
}

template <int kPagesPerCta, bool kProducerEpilogue>
__global__ __launch_bounds__(160, 4)
void dsa_indexer_scores_wgmma_ws(
    const __nv_bfloat16* __restrict__ q_idx,
    const __nv_bfloat16* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ w_idx,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens,
    __nv_bfloat16* __restrict__ scores,
    int max_pages) {
    constexpr int kWsThreads = 160;
    constexpr int kStages = kPagesPerCta == 4 ? 2 : 3;
    __shared__ __align__(128) uint4 k_s[kStages][kVectorsPerMatrix];

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
             i += kWsThreads) {
            out[i] = __float2bfloat16(-INFINITY);
        }
        return;
    }

    const int valid_pages = min(
        pages_this_cta,
        (visible - first_page * kPageSize + kPageSize - 1) / kPageSize);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    if (warp == 4) {
        if constexpr (!kProducerEpilogue) {
            // Producer only: barriers 0..2 signal ready and 3..5 signal free.
            for (int p = 0; p < valid_pages; ++p) {
                const int stage = p % kStages;
                if (p >= kStages) {
                    named_barrier_sync(3 + stage, kWsThreads);
                }
                int physical_page = 0;
                if (lane == 0) {
                    physical_page =
                        block_table[b * max_pages + first_page + p];
                }
                physical_page =
                    __shfl_sync(0xffffffffu, physical_page, 0);
                producer_load_k_page(k_s[stage], k_idx_cache,
                                     physical_page, lane);
                named_barrier_arrive(stage, kWsThreads);
            }
        } else {
            // Producer + epilogue: load p, then drain p-(kStages-1).
            for (int step = 0;
                 step < valid_pages + kStages - 1; ++step) {
                if (step < valid_pages) {
                    const int stage = step % kStages;
                    int physical_page = 0;
                    if (lane == 0) {
                        physical_page =
                            block_table[b * max_pages + first_page + step];
                    }
                    physical_page =
                        __shfl_sync(0xffffffffu, physical_page, 0);
                    producer_load_k_page(k_s[stage], k_idx_cache,
                                         physical_page, lane);
                    named_barrier_arrive(stage, kWsThreads);
                }

                const int done = step - (kStages - 1);
                if (done >= 0 && done < valid_pages) {
                    const int stage = done % kStages;
                    named_barrier_sync(3 + stage, kWsThreads);
                    float (*partial_s)[kPageSize] =
                        reinterpret_cast<float (*)[kPageSize]>(k_s[stage]);
                    const int logical_page = first_page + done;
                    const int page_begin = logical_page * kPageSize;
                    __nv_bfloat16* const out =
                        scores + (b * max_pages + logical_page) * kPageSize;
#pragma unroll
                    for (int token_offset = 0; token_offset < kPageSize;
                         token_offset += 32) {
                        const int token = lane + token_offset;
                        if (page_begin + token < visible) {
                            float score = partial_s[0][token] +
                                          partial_s[1][token];
                            score += partial_s[2][token] +
                                     partial_s[3][token];
                            out[token] = __float2bfloat16(score);
                        } else {
                            out[token] = __float2bfloat16(-INFINITY);
                        }
                    }
                }
            }

            const int invalid_pages = pages_this_cta - valid_pages;
            if (invalid_pages > 0) {
                __nv_bfloat16* const out = scores +
                    (b * max_pages + first_page + valid_pages) * kPageSize;
                for (int i = lane; i < invalid_pages * kPageSize; i += 32) {
                    out[i] = __float2bfloat16(-INFINITY);
                }
            }
        }
        return;
    }

    const int row0 = warp * 16 + (lane >> 2);
    const float w0 = __bfloat162float(w_idx[b * kHeads + row0]);
    const float w1 = __bfloat162float(w_idx[b * kHeads + row0 + 8]);
    QueryFragments q_frag;
    const int col_pair = (lane & 3) * 2;
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
        const int col = kt * 16 + col_pair;
        const __nv_bfloat16* const q0 =
            q_idx + ((b * kHeads + row0) * kDim + col);
        const __nv_bfloat16* const q1 = q0 + 8 * kDim;
        q_frag.x[kt][0] = *reinterpret_cast<const uint32_t*>(q0);
        q_frag.x[kt][1] = *reinterpret_cast<const uint32_t*>(q1);
        q_frag.x[kt][2] = *reinterpret_cast<const uint32_t*>(q0 + 8);
        q_frag.x[kt][3] = *reinterpret_cast<const uint32_t*>(q1 + 8);
    }

    for (int p = 0; p < valid_pages; ++p) {
        const int stage = p % kStages;
        named_barrier_sync(stage, kWsThreads);

        Accumulator acc;
#pragma unroll
        for (int i = 0; i < 32; ++i) acc.x[i] = 0.0f;
        wgmma_fence();
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            const uint64_t desc_k =
                make_gmma_desc(k_s[stage] + kt * 128);
            wgmma_m64n64k16_rs(acc, q_frag.x[kt], desc_k, kt != 0);
        }
        wgmma_commit_wait();

        float (*partial_s)[kPageSize] =
            reinterpret_cast<float (*)[kPageSize]>(k_s[stage]);
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

        if constexpr (kProducerEpilogue) {
            named_barrier_arrive(3 + stage, kWsThreads);
        } else {
            constexpr int kEpilogueBarrier = 6;
            named_barrier_sync(kEpilogueBarrier, kThreads);
            if (threadIdx.x < kPageSize) {
                const int token = threadIdx.x;
                const int logical_page = first_page + p;
                const int page_begin = logical_page * kPageSize;
                __nv_bfloat16* const out =
                    scores + (b * max_pages + logical_page) * kPageSize;
                if (page_begin + token < visible) {
                    float score = partial_s[0][token] +
                                  partial_s[1][token];
                    score += partial_s[2][token] +
                             partial_s[3][token];
                    out[token] = __float2bfloat16(score);
                } else {
                    out[token] = __float2bfloat16(-INFINITY);
                }
            }
            named_barrier_sync(kEpilogueBarrier, kThreads);
            if (p + kStages < valid_pages) {
                named_barrier_arrive(3 + stage, kWsThreads);
            }
        }
    }

    if constexpr (!kProducerEpilogue) {
        const int invalid_pages = pages_this_cta - valid_pages;
        if (invalid_pages > 0) {
            __nv_bfloat16* const out = scores +
                (b * max_pages + first_page + valid_pages) * kPageSize;
            for (int i = threadIdx.x; i < invalid_pages * kPageSize;
                 i += kThreads) {
                out[i] = __float2bfloat16(-INFINITY);
            }
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
    if (blocks64 <= 0) return;

    const int max_pages = static_cast<int>(MaxPages);
    if (blocks64 >= 32768) {
        constexpr int kPages = 64;
        const int64_t groups_per_batch = (MaxPages + kPages - 1) / kPages;
        const unsigned int blocks =
            static_cast<unsigned int>(B * groups_per_batch);
        dsa_indexer_scores_wgmma_ws<kPages, true><<<blocks, 160>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    } else if (blocks64 >= 16384) {
        constexpr int kPages = 32;
        const int64_t groups_per_batch = (MaxPages + kPages - 1) / kPages;
        const unsigned int blocks =
            static_cast<unsigned int>(B * groups_per_batch);
        dsa_indexer_scores_wgmma_ws<kPages, true><<<blocks, 160>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    } else if (blocks64 >= 8192) {
        constexpr int kPages = 16;
        const int64_t groups_per_batch = (MaxPages + kPages - 1) / kPages;
        const unsigned int blocks =
            static_cast<unsigned int>(B * groups_per_batch);
        dsa_indexer_scores_wgmma_ws<kPages, true><<<blocks, 160>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    } else if (blocks64 >= 4096) {
        constexpr int kPages = 8;
        const int64_t groups_per_batch = (MaxPages + kPages - 1) / kPages;
        const unsigned int blocks =
            static_cast<unsigned int>(B * groups_per_batch);
        dsa_indexer_scores_wgmma_ws<kPages, true><<<blocks, 160>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    } else if (blocks64 >= 2048) {
        constexpr int kPages = 4;
        const int64_t groups_per_batch = (MaxPages + kPages - 1) / kPages;
        const unsigned int blocks =
            static_cast<unsigned int>(B * groups_per_batch);
        dsa_indexer_scores_wgmma_ws<kPages, true><<<blocks, 160>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    } else {
        const unsigned int blocks = static_cast<unsigned int>(blocks64);
        dsa_indexer_scores_wgmma<1><<<blocks, kThreads>>>(
            q_idx, k_idx_cache, w_idx, block_table, context_lens, scores,
            max_pages);
    }
}
