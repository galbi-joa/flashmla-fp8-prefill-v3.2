#include <pybind11/pybind11.h>

#include "sparse_fwd.h"
#include "v32_kernelv8_fp8_prefill_fwd_bridge.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("v32_kernelv8_fp8_prefill_fwd", &v32_kernelv8_fp8_prefill_fwd_interface);
    m.def("v32_kernelv8_fp8_prefill_trace", &v32_kernelv8_fp8_prefill_trace_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
