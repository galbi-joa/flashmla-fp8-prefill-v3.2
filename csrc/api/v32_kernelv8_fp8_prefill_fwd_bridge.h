#pragma once

// Public PyTorch bridge for the standalone production v8 translation unit.
// Validation, output allocation, trace allocation, and dispatch are all owned
// here; no earlier-version bridge is required to build or call v8.

#include <climits>
#include <cstdint>
#include <optional>
#include <vector>

#include "common.h"
#include "params.h"
#include "sm100/prefill/sparse/v8_trace_schema.h"

namespace sm100::fwd::head64::v8 {
void run_v8_fp8_prefill(
    const SparseAttnFwdParams& params,
    int output_h_q);
void run_v8_fp8_prefill_trace(
    const SparseAttnFwdParams& params,
    int output_h_q,
    std::uint64_t* timestamps,
    int trace_s_q_idx,
    int trace_num_tiles);
}  // namespace sm100::fwd::head64::v8

static std::vector<at::Tensor> v32_kernelv8_fp8_prefill_fwd_impl(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length,
    int trace_s_q_idx
) {
    using bf16 = cutlass::bfloat16_t;

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
    int bytes_per_token = kv.size(3);
    int topk = indices.size(2);
    int s_kv = num_blocks * page_block_size;

    static constexpr int V32_BYTES_PER_TOKEN = 656;
    TORCH_CHECK(d_qk == 576, "V3.2 native fp8 prefill expects d_qk == 576, got ", d_qk);
    TORCH_CHECK(d_v == 512, "V3.2 native fp8 prefill expects d_v == 512, got ", d_v);
    static constexpr int V8_HEAD_TILE = 64;
    TORCH_CHECK(
        h_q == V8_HEAD_TILE || h_q == 2 * V8_HEAD_TILE,
        "V3.2 kernelv8 native fp8 prefill expects h_q == 64 or 128, got ",
        h_q);
    TORCH_CHECK(h_kv == 1, "V3.2 native fp8 prefill expects h_kv == 1, got ", h_kv);
    TORCH_CHECK(bytes_per_token == V32_BYTES_PER_TOKEN,
                "V3.2 native fp8 prefill expects 656 bytes per token, got ", bytes_per_token);
    TORCH_CHECK(topk > 0 && topk % 64 == 0,
                "V3.2 native fp8 prefill expects topk to be a positive multiple of 64, got ", topk);

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);

    const auto q_device = q.device();
    auto require_q_device = [&](const at::Tensor& tensor, const char* name) {
        TORCH_CHECK(
            tensor.device() == q_device,
            name, " must be on the same CUDA device as q (", q_device,
            "), got ", tensor.device());
    };
    require_q_device(kv, "kv");
    require_q_device(indices, "indices");
    if (attn_sink.has_value()) {
        require_q_device(*attn_sink, "attn_sink");
    }
    if (topk_length.has_value()) {
        require_q_device(*topk_length, "topk_length");
    }

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    Arch arch = Arch();
    TORCH_CHECK(arch.is_sm100f(), "v32_kernelv8_fp8_prefill_fwd is only supported on SM100f architectures.");

    KU_CHECK_DTYPE(q, torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn ||
                    kv.dtype() == torch::kInt8 ||
                    kv.dtype() == torch::kUInt8,
                "kv must have dtype fp8_e4m3fn, int8, or uint8");
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);

    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, V32_BYTES_PER_TOKEN);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    TORCH_CHECK(num_blocks > 0 && page_block_size > 0,
                "V3.2 fp8 KV cache must contain at least one non-empty page");
    TORCH_CHECK(kv.stride(1) == V32_BYTES_PER_TOKEN,
                "V3.2 fp8 KV cache must be contiguous within each page block");
    TORCH_CHECK(kv.stride(0) >= page_block_size * V32_BYTES_PER_TOKEN,
                "V3.2 fp8 KV block stride is smaller than its page payload");
    TORCH_CHECK(kv.stride(0) % V32_BYTES_PER_TOKEN == 0,
                "V3.2 fp8 KV block stride must be a multiple of 656 bytes");
    TORCH_CHECK(
        static_cast<std::int64_t>(num_blocks) *
                (kv.stride(0) / V32_BYTES_PER_TOKEN) <= INT_MAX,
        "V3.2 fp8 KV cache is too large for the TMA coordinate domain");
    const auto kv_address = reinterpret_cast<std::uintptr_t>(kv.data_ptr());
    TORCH_CHECK(kv_address % 16 == 0,
                "V3.2 fp8 KV cache base must be 16-byte aligned");
    TORCH_CHECK((kv_address + 512) % 16 == 0 &&
                    (kv_address + 528) % 16 == 0,
                "V3.2 fp8 scale and RoPE regions must be 16-byte aligned");
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(indices.data_ptr()) % 16 == 0,
        "V3.2 indices base must be 16-byte aligned");
    TORCH_CHECK(indices.stride(0) % 4 == 0,
                "V3.2 indices row stride must preserve int4 alignment");
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);

    auto opts = q.options();
    at::Tensor out = torch::empty({s_q, h_q, d_v}, opts);
    at::Tensor max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    KU_CHECK_CONTIGUOUS(out);
    KU_CHECK_CONTIGUOUS(max_logits);
    KU_CHECK_CONTIGUOUS(lse);

    SparseAttnFwdParams params = {
        s_q, s_kv, V8_HEAD_TILE, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        ku::get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        0, 0,
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (bf16*)out.data_ptr(),
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),

        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    };
    params.num_blocks = num_blocks;
    params.page_block_size = page_block_size;
    params.stride_kv_block = int64_stride_to_int(kv.stride(0));

    const int num_head_tiles = h_q / V8_HEAD_TILE;
    auto params_for_head_tile = [&](int head_tile_idx) {
        const int head_base = head_tile_idx * V8_HEAD_TILE;
        SparseAttnFwdParams cur_params = params;
        cur_params.q += static_cast<std::int64_t>(head_base) *
                        params.stride_q_h_q;
        if (cur_params.attn_sink != nullptr) {
            cur_params.attn_sink += head_base;
        }
        cur_params.out += static_cast<std::int64_t>(head_base) *
                          int64_stride_to_int(out.stride(1));
        cur_params.max_logits +=
            static_cast<std::int64_t>(head_base) *
            int64_stride_to_int(max_logits.stride(1));
        cur_params.lse += static_cast<std::int64_t>(head_base) *
                          int64_stride_to_int(lse.stride(1));
        return cur_params;
    };

    if (trace_s_q_idx >= 0) {
        TORCH_CHECK(
            h_q == V8_HEAD_TILE,
            "V3.2 kernelv8 trace currently supports h_q == 64 only; "
            "use the untraced H128 entry point for dual-launch execution");
        TORCH_CHECK(trace_s_q_idx < s_q,
                    "trace_s_q_idx must be in [0, ", s_q, "), got ", trace_s_q_idx);
        const int trace_num_tiles = (topk + 63) / 64;
        const int trace_tracks = sm100::fwd::head64::v8::TRACE_TRACKS;
        const int trace_schema_version =
            sm100::fwd::head64::v8::TRACE_SCHEMA_VERSION;
        at::Tensor timestamps = torch::zeros(
            {
                trace_tracks,
                trace_num_tiles + 1,
                sm100::fwd::head64::v8::TRACE_POINTS,
            },
            opts.dtype(torch::kInt64));
        at::Tensor trace_schema = torch::full(
            {},
            trace_schema_version,
            opts.dtype(torch::kInt32));
        sm100::fwd::head64::v8::run_v8_fp8_prefill_trace(
            params_for_head_tile(0),
            h_q,
            reinterpret_cast<std::uint64_t*>(
                timestamps.data_ptr<std::int64_t>()),
            trace_s_q_idx,
            trace_num_tiles);
        return {out, max_logits, lse, timestamps, trace_schema};
    }

    for (int head_tile_idx = 0; head_tile_idx < num_head_tiles;
         ++head_tile_idx) {
        sm100::fwd::head64::v8::run_v8_fp8_prefill(
            params_for_head_tile(head_tile_idx), h_q);
    }
    return {out, max_logits, lse};
}

static std::vector<at::Tensor> v32_kernelv8_fp8_prefill_fwd_interface(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length
) {
    return v32_kernelv8_fp8_prefill_fwd_impl(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length, -1);
}

static std::vector<at::Tensor> v32_kernelv8_fp8_prefill_trace_interface(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v,
    int trace_s_q_idx,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length
) {
    return v32_kernelv8_fp8_prefill_fwd_impl(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length,
        trace_s_q_idx);
}
