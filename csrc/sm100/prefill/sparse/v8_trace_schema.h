#pragma once

// Trace contract owned by the standalone production v8 translation unit.
// Keeping it here avoids dependencies on earlier trace implementations.
namespace sm100::fwd::head64::v8 {

inline constexpr int TRACE_SCHEMA_VERSION = 5;
inline constexpr int TRACE_TRACKS = 16;
inline constexpr int TRACE_POINTS = 16;

// Only the 512-thread production topology is instantiated.  The WG2 aliases
// keep uninstantiated generic helper branches self-contained in this file.
enum TraceTrack : int {
    TRACE_WG0_WARP0 = 0,
    TRACE_DQ0_WARP4 = 4,
    TRACE_DQ1_WARP8 = 8,
    TRACE_PRODUCER_MMA = 12,
    TRACE_PRODUCER_META = 13,
    TRACE_PRODUCER_RAW = 14,
    TRACE_PRODUCER_ROPE = 15,

    TRACE_WG2_MMA = TRACE_PRODUCER_MMA,
    TRACE_WG2_META = TRACE_PRODUCER_META,
    TRACE_WG2_RAW = TRACE_PRODUCER_RAW,
    TRACE_WG2_ROPE = TRACE_PRODUCER_ROPE,
};

}  // namespace sm100::fwd::head64::v8
