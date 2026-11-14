#pragma once

// Trace contract owned by the standalone production v8 translation unit.
// Keeping it here avoids dependencies on earlier trace implementations.
namespace sm100::fwd::head64::v8 {

inline constexpr int TRACE_SCHEMA_VERSION = 5;
inline constexpr int TRACE_TRACKS = 16;
inline constexpr int TRACE_POINTS = 16;

// WG0 and dequant tracks use their warp indices (0..11).
enum TraceTrack : int {
    TRACE_PRODUCER_MMA = 12,
    TRACE_PRODUCER_META = 13,
    TRACE_PRODUCER_RAW = 14,
    TRACE_PRODUCER_ROPE = 15,
};

// Point numbers are part of the schema read by the Perfetto exporter.
// Summary points use the extra row at tile == num_tiles.
namespace trace_point {
namespace wg0 {
enum : int {
    TILE_BEGIN = 0,
    QK_AND_VALID_READY = 1,
    SOFTMAX_DONE = 2,
    PREV_SV_DONE = 3,
    S_PUBLISHED = 4,
    PAIR_READY = 5,
};
}

namespace wg0_summary {
enum : int {
    ENTER = 0,
    TILES_DONE = 1,
    FINAL_SV_DONE = 2,
    EXIT = 3,
    EPILOGUE_BEGIN = 4,
};
}

namespace dq {
enum : int {
    TILE_BEGIN = 0,
    RAW_READY = 1,
    SCALES_READY = 2,
    VALID_READY = 3,
    V_SLOT_FREE = 4,
    Q_STORAGE_FREE = 5,
    BF16_PUBLISHED = 6,
};
}

namespace dq_summary {
enum : int {
    ENTER = 0,
    EXIT = 1,
};
}

namespace mma {
enum : int {
    QK_BEGIN = 0,
    P_SLOT_FREE = 1,
    K_ROPE_READY = 2,
    ROPE_QK_ISSUED = 3,
    BF16_READY = 4,
    QK_ISSUED = 5,
    SV_BEGIN = 6,
    S_READY = 7,
    SV_ISSUED = 8,
    BF16_WAIT_BEGIN = 9,
};
}

namespace mma_summary {
enum : int {
    ENTER = 0,
    Q_ROPE_TMA_READY = 1,
    Q_ROPE_COPY_ISSUED = 2,
    Q_NOPE_TMA_READY = 3,
    Q_NOPE_COPY_ISSUED = 4,
    Q_ROPE_STORAGE_FREE = 5,
};
}

namespace meta {
enum : int {
    COORD_PUBLISH_BEGIN = 0,
    COORD_SLOT_FREE = 1,
    COORD_PUBLISHED = 2,
    LOAD_BEGIN = 3,
    LOAD_DONE = 4,
    SCALE_VALID_SLOTS_FREE = 5,
    SCALE_VALID_PUBLISHED = 6,
    SCALE_VALID_WAIT_BEGIN = 7,
};
}

namespace raw {
enum : int {
    TILE_BEGIN = 0,
    RAW_SLOT_FREE = 1,
    COORD_READY = 2,
    GATHER_ISSUED = 3,
};
}

namespace rope {
enum : int {
    TILE_BEGIN = 0,
    DEPS_READY = 1,
    GATHER_BEGIN = 2,
    GATHER_ISSUED = 3,
    COORD_RELEASED = 4,
};
}
}  // namespace trace_point

}  // namespace sm100::fwd::head64::v8
