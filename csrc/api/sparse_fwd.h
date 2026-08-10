#pragma once

#include "common.h"

#include "params.h"

#include "sm90/prefill/sparse/phase1.h"
#include "sm100/prefill/sparse/fwd/head128/phase1.h"
#include "sm100/prefill/sparse/fwd/head64/phase1.h"
#include "sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.h"

enum class FwdFeatures : int {
    HEAD_64,
    HEAD_128,

    HEAD_DIM_576,
    HEAD_DIM_512,

    ATTN_SINK,
    SINK_LSE,
    TOPK_LENGTH,

    FP8_KVCACHE     // MODEL1-layout fp8 KV cache, consumed natively (dequantized on the fly in shared memory)
};

class FwdImplBase : public ImplBase<
    SparseAttnFwdParams,
    FwdFeatures
> {};

class Fwd_Sm90_Impl : public FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        FwdFeatures::HEAD_64,
        FwdFeatures::HEAD_128,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::HEAD_DIM_576,
        FwdFeatures::ATTN_SINK,
        FwdFeatures::SINK_LSE,
        FwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            DISPATCH_BOOLEAN_FLAG(params.topk_length != nullptr, HAVE_TOPK_LENGTH, [&]() {
                sm90::fwd::run_fwd_phase1_kernel<HEAD_DIM_QK, HAVE_TOPK_LENGTH>(params);
            });
        });
    }
};

class Fwd_Sm100_Head64_Impl : public FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        FwdFeatures::HEAD_64,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::HEAD_DIM_576,
        FwdFeatures::ATTN_SINK,
        FwdFeatures::SINK_LSE,
        FwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            sm100::fwd::head64::run_fwd_phase1_kernel<HEAD_DIM_QK>(params);
        });
    }
};

class Fwd_Sm100_Head128_Impl : public FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        FwdFeatures::HEAD_128,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::HEAD_DIM_576,
        FwdFeatures::ATTN_SINK,
        FwdFeatures::SINK_LSE,
        FwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            sm100::fwd::head128::run_fwd_phase1_kernel<HEAD_DIM_QK>(params);
        });
    }
};

class Fwd_Sm100_Head128_Small_TopK_Impl : public FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        FwdFeatures::HEAD_128,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::ATTN_SINK,
        FwdFeatures::SINK_LSE,
        FwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        sm100::fwd_for_small_topk::head128::run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::Prefill, 512>(params);
    }
};

class Fwd_Sm100_Head128_Fp8_Impl : public FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        FwdFeatures::HEAD_128,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::ATTN_SINK,
        FwdFeatures::SINK_LSE,
        FwdFeatures::TOPK_LENGTH,
        FwdFeatures::FP8_KVCACHE
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        sm100::fwd_for_small_topk::head128::run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::PrefillFp8, 512>(params);
    }
};

// Prefill that directly consumes the fp8 (MODEL1-layout) KV cache, without materializing a bf16 copy
// of the cache in global memory. The cache layout matches the one used by the sparse decoding kernel
// (see tests/quant.py, FP8KVCacheLayout.MODEL1_FP8Sparse), so prefill and decode can share one cache.
static std::vector<at::Tensor> sparse_attn_prefill_fp8_interface(
    const at::Tensor &q,        // [s_q, h_q, 512], bf16
    const at::Tensor &kv,       // [num_blocks, page_block_size, 1, 584], fp8_e4m3fn / int8 / uint8
    const at::Tensor &indices,  // [s_q, 1, topk], int32. indices[i][0][k] = block_idx * page_block_size + offset_in_block, or -1 / >= s_kv for invalid entries
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    TORCH_CHECK(arch.is_sm100f(), "FP8 sparse attention forward (prefill) is only supported on SM100f architectures.");

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);

    int s_q = q.size(0);
    int h_q = q.size(1);
    int d_qk = q.size(2);
    int num_blocks = kv.size(0);
    int page_block_size = kv.size(1);
    int h_kv = kv.size(2);
    int topk = indices.size(2);
    int s_kv = num_blocks * page_block_size;

    static constexpr int BYTES_PER_TOKEN = 448 + 64*2 + (448/64)*1 + 1;   // MODEL1 layout, 584 bytes per token
    TORCH_CHECK(d_qk == 512, "Only d_qk == 512 (the MODEL1 shape) is supported for fp8 sparse prefill, got ", d_qk);
    TORCH_CHECK(d_v == 512, "Invalid d_v: ", d_v);
    TORCH_CHECK(h_q == 128, "Only h_q == 128 is supported for fp8 sparse prefill, got ", h_q);
    TORCH_CHECK(h_kv == 1, "Only MQA (h_kv == 1) is supported for fp8 sparse prefill");
    TORCH_CHECK(topk > 0 && topk % 64 == 0, "topk must be a positive multiple of 64 for fp8 sparse prefill, got ", topk);

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);

    KU_CHECK_DTYPE(q, torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kInt8 || kv.dtype() == torch::kUInt8, "kv must have dtype fp8_e4m3fn, int8 or uint8");
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);

    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, BYTES_PER_TOKEN);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    TORCH_CHECK(kv.stride(1) == BYTES_PER_TOKEN, "The whole block must be contiguous for the fp8 KV cache");
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    at::Tensor out = torch::empty({s_q, h_q, d_v}, opts);
    at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    at::Tensor max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    KU_CHECK_CONTIGUOUS(out);
    KU_CHECK_CONTIGUOUS(lse);
    KU_CHECK_CONTIGUOUS(max_logits);

    SparseAttnFwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        ku::get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        0, 0,   // stride_kv_s_kv / stride_kv_h_kv are unused for the fp8 cache; use stride_kv_block below
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (bf16*)out.data_ptr(),
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),

        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    };
    params.num_blocks = num_blocks;
    params.page_block_size = page_block_size;
    {
        // The kernel derives TMA coordinates from stride/576 and asserts the
        // divisibility. When the cache has a single block, PyTorch may have
        // re-strided the size-1 leading dimension (e.g. a .view() of a sliced
        // padded buffer computes contiguous strides for size-1 dims), dropping
        // the 576-padding. The outer stride is never used to address a second
        // block in that case, so normalize it instead of rejecting the tensor.
        int stride_kv_block = int64_stride_to_int(kv.stride(0));
        if (num_blocks == 1) {
            stride_kv_block = (stride_kv_block + 575) / 576 * 576;
        }
        params.stride_kv_block = stride_kv_block;
    }

    std::vector<FwdFeatures> required_features = {
        FwdFeatures::HEAD_128,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::FP8_KVCACHE
    };
    if (attn_sink.has_value()) {
        required_features.push_back(FwdFeatures::ATTN_SINK);
    }
    if (topk_length.has_value()) {
        required_features.push_back(FwdFeatures::TOPK_LENGTH);
    }

    Fwd_Sm100_Head128_Fp8_Impl fwd_impl;
    fwd_impl.run(params, required_features);

    return {out, max_logits, lse};
}

static std::vector<at::Tensor> sparse_attn_prefill_interface(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length
) {
    using bf16 = cutlass::bfloat16_t;

    // The dtype of `kv` selects the path: bf16 means a plain [s_kv, h_kv, d_qk] KV tensor,
    // while fp8_e4m3fn / int8 / uint8 means the quantized (MODEL1-layout) paged KV cache.
    if (kv.dtype() != torch::kBFloat16) {
        return sparse_attn_prefill_fp8_interface(q, kv, indices, sm_scale, d_v, attn_sink, topk_length);
    }

    Arch arch = Arch();
    bool is_sm90a = arch.is_sm90a();
    bool is_sm100f = arch.is_sm100f();
    TORCH_CHECK(is_sm90a || is_sm100f, "Sparse Attention Forward Kernel is only supported on SM90a and SM100f architectures.");

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);
    bool have_topk_length = topk_length.has_value();

    TORCH_CHECK(d_qk == 576 || d_qk == 512, "Invalid d_qk: ", d_qk);
    TORCH_CHECK(d_v == 512, "Invalid d_v", d_v);
    
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);
    
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(kv, torch::kBFloat16);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    
    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);
    
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);
    
    // Allocate results and buffers
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    
    at::Tensor out = torch::empty({s_q, h_q, d_v}, opts);
    at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    at::Tensor max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    KU_CHECK_CONTIGUOUS(out);
    KU_CHECK_CONTIGUOUS(lse);
    KU_CHECK_CONTIGUOUS(max_logits);

    SparseAttnFwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        ku::get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (bf16*)out.data_ptr(),
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),

        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    };

    std::vector<FwdFeatures> required_features;
    if (h_q == 64) {
        required_features.push_back(FwdFeatures::HEAD_64);
    } else if (h_q == 128) {
        required_features.push_back(FwdFeatures::HEAD_128);
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (d_qk == 576) {
        required_features.push_back(FwdFeatures::HEAD_DIM_576);
    } else if (d_qk == 512) {
        required_features.push_back(FwdFeatures::HEAD_DIM_512);
    } else {
        TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }
    if (attn_sink.has_value()) {
        required_features.push_back(FwdFeatures::ATTN_SINK);
    }
    if (have_topk_length) {
        required_features.push_back(FwdFeatures::TOPK_LENGTH);
    }

    if (is_sm90a) {
        Fwd_Sm90_Impl fwd_impl;
        fwd_impl.run(params, required_features);
    } else if (is_sm100f) {
        if (h_q == 64) {
            Fwd_Sm100_Head64_Impl fwd_impl;
            fwd_impl.run(params, required_features);
        } else if (h_q == 128) {
            Fwd_Sm100_Head128_Small_TopK_Impl small_topk_impl;
            Fwd_Sm100_Head128_Impl regular_impl;
            bool use_small_topk_impl = false;
            if (
                (topk <= 1280 && small_topk_impl.check_if_all_features_are_supported(required_features)) ||
                !regular_impl.check_if_all_features_are_supported(required_features)
            ) {
                use_small_topk_impl = true;
            }
            if (use_small_topk_impl) {
                small_topk_impl.run(params, required_features);
            } else {
                regular_impl.run(params, required_features);
            }
        } else {
            TORCH_CHECK(false, "Unsupported h_q: ", h_q);
        }
    } else {
        TORCH_CHECK(false, "Unsupported architecture");
    }

    return {out, max_logits, lse};
}
