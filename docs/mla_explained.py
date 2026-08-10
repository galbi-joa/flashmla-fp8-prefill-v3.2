"""
MLA (Multi-head Latent Attention) 계산 순서 — 코드로 설명
DeepSeek-V3 기준 치수 사용

HEAD_DIM_K = 576  (= d_c 512 + d_R 64)
HEAD_DIM_V = 512  (= d_c)
"""

import torch
import torch.nn.functional as F
import math

# ─────────────────────────────────────────────
# 치수 정의 (DeepSeek-V3 기준)
# ─────────────────────────────────────────────
D_MODEL   = 7168   # hidden size
N_HEADS   = 128    # query head 수
D_HEAD_C  = 128    # per-head compressed dim (K_C, V 각각)
D_R       = 64     # RoPE 전용 dim
D_C       = 512    # KV latent dim  (= N_HEADS * D_HEAD_C / some_factor, 실제론 512)
D_C_Q     = 1536   # Q latent dim

# 파생 치수
HEAD_DIM_K = D_C + D_R   # 576  ← csrc/sm90/decode/dense/config.h
HEAD_DIM_V = D_C          # 512  ← csrc/sm90/decode/dense/config.h

SEQ_Q  = 1     # decode: 보통 1
SEQ_KV = 4096  # KV 캐시 길이
B      = 2     # batch size

device = "cpu"  # GPU 있으면 "cuda"로 변경


# ─────────────────────────────────────────────
# 가중치 정의
# ─────────────────────────────────────────────
class MLAWeights:
    def __init__(self):
        # KV 압축
        self.W_DKV = torch.randn(D_MODEL, D_C,   device=device) * 0.01   # Down-proj KV
        self.W_UK  = torch.randn(D_C,   N_HEADS * D_HEAD_C, device=device) * 0.01  # Up-proj K
        self.W_UV  = torch.randn(D_C,   N_HEADS * D_HEAD_C, device=device) * 0.01  # Up-proj V
        self.W_KR  = torch.randn(D_MODEL, D_R,   device=device) * 0.01   # RoPE key

        # Q 압축
        self.W_DQ  = torch.randn(D_MODEL, D_C_Q,            device=device) * 0.01  # Down-proj Q
        self.W_UQ  = torch.randn(D_C_Q,  N_HEADS * D_HEAD_C, device=device) * 0.01  # Up-proj Q
        self.W_QR  = torch.randn(D_C_Q,  N_HEADS * D_R,     device=device) * 0.01  # RoPE query

        # 출력
        self.W_O   = torch.randn(N_HEADS * D_HEAD_C, D_MODEL, device=device) * 0.01

W = MLAWeights()


def apply_rope(x):
    """RoPE 적용 (단순화 버전 — 실제론 회전 행렬 적용)"""
    # x: (B, seq, n_heads, d_R) or (B, seq, d_R)
    return x  # 단순화를 위해 identity로 둠


# ══════════════════════════════════════════════════════════════════
# PART 1: 학습 시 MLA Forward (Weight Absorption 없음)
# ══════════════════════════════════════════════════════════════════
def mla_forward_training(x, x_kv):
    """
    x    : (B, SEQ_Q,  D_MODEL)  — 현재 토큰
    x_kv : (B, SEQ_KV, D_MODEL)  — 전체 context
    """
    B = x.shape[0]

    # ── Step 1: KV 압축 ────────────────────────────────────────────
    # c_KV: KV latent. 이게 KV 캐시에 저장될 핵심 정보
    c_KV = x_kv @ W.W_DKV                    # (B, SEQ_KV, 512)

    # k_R: RoPE 전용 key. c_KV와 함께 캐시에 저장됨
    k_R  = x_kv @ W.W_KR                     # (B, SEQ_KV, 64)
    k_R  = apply_rope(k_R)

    # ── Step 2: K, V 복원 (Up-projection) ─────────────────────────
    K_C  = c_KV @ W.W_UK                     # (B, SEQ_KV, N_HEADS*128)
    K_C  = K_C.view(B, SEQ_KV, N_HEADS, D_HEAD_C)

    # K = [K_C | k_R]  — k_R를 모든 head에 broadcast
    k_R_expand = k_R.unsqueeze(2).expand(-1, -1, N_HEADS, -1)
    K = torch.cat([K_C, k_R_expand], dim=-1) # (B, SEQ_KV, N_HEADS, 128+64=192)

    V    = c_KV @ W.W_UV                     # (B, SEQ_KV, N_HEADS*128)
    V    = V.view(B, SEQ_KV, N_HEADS, D_HEAD_C)

    # ── Step 3: Q 계산 ─────────────────────────────────────────────
    c_Q  = x @ W.W_DQ                        # (B, SEQ_Q, 1536)
    Q_C  = c_Q @ W.W_UQ                      # (B, SEQ_Q, N_HEADS*128)
    Q_C  = Q_C.view(B, SEQ_Q, N_HEADS, D_HEAD_C)

    q_R  = c_Q @ W.W_QR                      # (B, SEQ_Q, N_HEADS*64)
    q_R  = q_R.view(B, SEQ_Q, N_HEADS, D_R)
    q_R  = apply_rope(q_R)

    Q    = torch.cat([Q_C, q_R], dim=-1)     # (B, SEQ_Q, N_HEADS, 192)

    # ── Step 4: Attention ──────────────────────────────────────────
    # Q, K: (B, N_HEADS, SEQ, D)
    Q = Q.transpose(1, 2)   # (B, N_HEADS, SEQ_Q,  192)
    K = K.transpose(1, 2)   # (B, N_HEADS, SEQ_KV, 192)
    V = V.transpose(1, 2)   # (B, N_HEADS, SEQ_KV, 128)

    scale  = 1.0 / math.sqrt(D_HEAD_C + D_R)   # 1/sqrt(192)
    scores = (Q @ K.transpose(-1, -2)) * scale  # (B, N_HEADS, SEQ_Q, SEQ_KV)
    A      = F.softmax(scores, dim=-1)           # (B, N_HEADS, SEQ_Q, SEQ_KV)
    out    = A @ V                               # (B, N_HEADS, SEQ_Q, 128)

    # ── Step 5: 출력 projection ────────────────────────────────────
    out = out.transpose(1, 2).reshape(B, SEQ_Q, N_HEADS * D_HEAD_C)
    out = out @ W.W_O                            # (B, SEQ_Q, D_MODEL)

    return out


# ══════════════════════════════════════════════════════════════════
# PART 2: 추론 시 MLA Forward (Weight Absorption 적용)
# ══════════════════════════════════════════════════════════════════
#
# 핵심 아이디어:
#   Q_C @ K_C^T = Q_C @ (c_KV @ W_UK)^T
#               = (Q_C @ W_UK^T) @ c_KV^T
#                  ──────────────
#                  Q_absorbed_C
#
# 즉, W_UK를 Q에 미리 흡수시키면:
#   K 자리에 c_KV (512 dim)을 그대로 사용 가능
#   → KV 캐시: c_KV (512) + k_R (64) = 576 dim만 저장
#
# 마찬가지로 V도:
#   A @ V = A @ (c_KV @ W_UV) = (A @ c_KV) @ W_UV
#           → W_UV를 W_O에 흡수시킴
#

def precompute_absorbed_weights(W):
    """
    W_UK를 Q 쪽에, W_UV를 W_O 쪽에 흡수.
    모델 로드 시 1회만 실행.
    """
    # W_UK: (D_C=512, N_HEADS*D_HEAD_C=16384)
    # W_UQ: (D_C_Q=1536, N_HEADS*D_HEAD_C=16384)
    # W_UQ_absorbed: Q_C (D_C_Q → N_HEADS*D_HEAD_C) 이후 W_UK^T (→ D_C)를 한 행렬로 합침
    # 즉 c_Q → Q_absorbed_C는: c_Q @ W_UQ @ W_UK^T
    W_UQ_absorbed = W.W_UQ @ W.W_UK.T          # (D_C_Q, D_C) = (1536, 512)

    # W_UV: (D_C=512, N_HEADS*D_HEAD_C=16384)
    # W_O:  (N_HEADS*D_HEAD_C, D_MODEL)
    # W_O_absorbed: (A @ c_KV) → (A @ c_KV @ W_UV) @ W_O를 W_UV @ W_O로 합침
    W_O_absorbed  = W.W_UV @ W.W_O             # (D_C, D_MODEL) = (512, 7168)

    return W_UQ_absorbed, W_O_absorbed


def mla_forward_inference(x, kv_cache, W, W_UQ_absorbed, W_O_absorbed):
    """
    x         : (B, 1,      D_MODEL)  — 새 토큰 1개 (decode step)
    kv_cache  : (B, SEQ_KV, HEAD_DIM_K=576)  — 저장된 KV 캐시
                  = concat(c_KV[512], k_R[64])

    ┌─────────────────────────────────────────────────────────┐
    │  KV 캐시 구조 (per token, 576 dim)                      │
    │  [ c_KV (0:512) | k_R (512:576) ]                      │
    │    ↑ V 계산에 사용    ↑ RoPE key로만 사용              │
    └─────────────────────────────────────────────────────────┘
    """
    B       = x.shape[0]
    SEQ_KV  = kv_cache.shape[1]

    # KV 캐시에서 분리
    c_KV = kv_cache[:, :, :D_C]   # (B, SEQ_KV, 512) ← V 계산에 사용
    k_R  = kv_cache[:, :, D_C:]   # (B, SEQ_KV, 64)  ← RoPE key

    # ── Step 1: 새 토큰을 KV 캐시에 추가 ─────────────────────────
    new_c_KV = x @ W.W_DKV                    # (B, 1, 512)
    new_k_R  = apply_rope(x @ W.W_KR)         # (B, 1, 64)
    new_kv   = torch.cat([new_c_KV, new_k_R], dim=-1)  # (B, 1, 576)
    kv_cache = torch.cat([kv_cache, new_kv], dim=1)    # 캐시에 추가

    # ── Step 2: Absorbed Q 계산 ───────────────────────────────────
    c_Q = x @ W.W_DQ                           # (B, 1, 1536)

    # Q_absorbed_C = c_Q @ W_UQ @ W_UK^T = c_Q @ W_UQ_absorbed
    Q_absorbed_C = c_Q @ W_UQ_absorbed         # (B, 1, 512)   ← 흡수된 Q
    Q_absorbed_C = Q_absorbed_C.view(B, 1, N_HEADS, D_C // N_HEADS)
    # 참고: D_C // N_HEADS = 512 // 128 = 4 이지만 실제론 head별로 다름
    # 단순화를 위해 아래에선 head 구분 없이 flat으로 처리

    q_R = apply_rope(c_Q @ W.W_QR)             # (B, 1, N_HEADS*64)

    # Q_absorbed = [Q_absorbed_C (512) | q_R (N_HEADS*64)]
    # 실제 커널에서 Q shape: (SEQ_Q, N_HEADS, HEAD_DIM_K=576)
    Q_absorbed_flat = torch.cat([
        c_Q @ W_UQ_absorbed,                   # (B, 1, 512)
        q_R                                    # (B, 1, N_HEADS*64=8192) ← 실제론 per-head
    ], dim=-1)
    # 단순화를 위해 per-head로 분리하지 않고 설명용으로 표기

    # ── Step 3: Q @ K^T (Attention Score) ────────────────────────
    # 실제 커널에서 수행하는 것:
    #   Q_absorbed: (B, N_HEADS, SEQ_Q=1, HEAD_DIM_K=576)
    #   KV_cache:   (B, SEQ_KV, HEAD_DIM_K=576)
    #   → scores = Q_absorbed @ KV_cache^T   (576 dim 전부 사용)
    #
    # HEAD_DIM_K = 576 → 9개 TMA 타일(64씩)로 분할해서 비동기 로드
    # (warpgroup_cooperative_qkt_gemm 함수가 이 역할)

    # 단순화 버전:
    # c_KV_full = kv_cache[:, :, :D_C]   # (B, SEQ_KV, 512)
    # k_R_full  = kv_cache[:, :, D_C:]   # (B, SEQ_KV, 64)
    # Q_C_part  = c_Q @ W_UQ_absorbed    # (B, 1, 512)
    # q_R_part  = q_R[:, :, :D_R]       # (B, 1, 64) — 첫 head만 예시
    # score_C = Q_C_part @ c_KV_full.transpose(-1, -2)   # (B, 1, SEQ_KV)
    # score_R = q_R_part @ k_R_full.transpose(-1, -2)    # (B, 1, SEQ_KV)
    # scores  = score_C + score_R                          # 합산 (내적이라 분리 가능)

    print(f"[Score] Q shape: (B, N_HEADS, SEQ_Q=1, {HEAD_DIM_K})")
    print(f"[Score] K shape: (B, SEQ_KV={SEQ_KV}, {HEAD_DIM_K})  ← KV 캐시 전체")
    print(f"[Score] 사용 dim: {HEAD_DIM_K}  (= {D_C} + {D_R})")

    # ── Step 4: Softmax ───────────────────────────────────────────
    # FlashMLA에서는 online softmax로 처리:
    #   - running max (sM), running sum (rL) 유지
    #   - wg0_bunch_0() / wg1_bunch_0() 가 이 역할
    #   - exp2f(x * scale_softmax_log2 - max) 로 계산 (expf 대신 exp2f로 속도 향상)
    print(f"\n[Softmax] online softmax with scale = 1/sqrt({HEAD_DIM_K}) = {1/math.sqrt(HEAD_DIM_K):.5f}")

    # ── Step 5: A @ V (Output) ────────────────────────────────────
    # 핵심: V 계산에는 c_KV 부분(512 dim)만 사용. k_R(64 dim)은 버림!
    #   V_latent = KV_cache[:, :, :512]   (= c_KV)
    #   out      = A @ V_latent           (B, N_HEADS, SEQ_Q, 512)
    #
    # HEAD_DIM_V = 512 → 두 warpgroup이 256씩 나눠서 처리
    # (warpgroup_cooperative_pv_gemm_localP / _remoteP 함수가 이 역할)

    print(f"\n[P@V] V shape: (B, SEQ_KV={SEQ_KV}, {HEAD_DIM_V})  ← 512 dim만 (k_R 64는 버림)")
    print(f"[P@V] 사용 dim: {HEAD_DIM_V}  (= {D_C} only, k_R {D_R} 제외)")

    # ── Step 6: Absorbed Output Projection ───────────────────────
    # out @ W_UV @ W_O = out @ W_O_absorbed
    # (W_UV 없이 직접 D_C → D_MODEL 변환)
    print(f"\n[Output] out ({HEAD_DIM_V}) @ W_O_absorbed → D_MODEL ({D_MODEL})")
    print(f"         (W_UV가 W_O에 흡수됨: W_UV({D_C},{N_HEADS*D_HEAD_C}) @ W_O({N_HEADS*D_HEAD_C},{D_MODEL}))")

    return kv_cache  # 업데이트된 KV 캐시 반환


# ══════════════════════════════════════════════════════════════════
# PART 3: FlashMLA 커널과의 대응
# ══════════════════════════════════════════════════════════════════
def show_kernel_mapping():
    print("""
┌──────────────────────────────────────────────────────────────────┐
│          MLA 수식          │       FlashMLA 커널 코드            │
├──────────────────────────────────────────────────────────────────┤
│  KV cache 저장             │  params.k_ptr                       │
│  = [c_KV(512) | k_R(64)]  │  shape: (num_blocks, 64, 576)       │
│  = 576 dim                 │  HEAD_DIM_K = 576                   │
├──────────────────────────────────────────────────────────────────┤
│  Q_absorbed 로드           │  launch_q_copy()                    │
│  (SEQ_Q=64, HEAD_DIM_K=576)│  TMA로 sQ에 로드 (73728 bytes)     │
├──────────────────────────────────────────────────────────────────┤
│  KV 캐시 로드              │  launch_kv_tiles_copy_tma<0,9>()   │
│  576 = 9 × 64 타일         │  barriers_K0[9], barriers_K1[9]     │
├──────────────────────────────────────────────────────────────────┤
│  Q_absorbed @ KV^T         │  warpgroup_cooperative_qkt_gemm()  │
│  → P (SEQ_Q × SEQ_KV)      │  sQ (64×576) @ sK^T (576×64) = rP │
├──────────────────────────────────────────────────────────────────┤
│  softmax(P / sqrt(576))    │  wg0_bunch_0() / wg1_bunch_0()     │
│  online softmax            │  sM (max), rL (sum), scale_log2    │
├──────────────────────────────────────────────────────────────────┤
│  P @ V (V = c_KV 512 dim)  │  warpgroup_cooperative_pv_gemm_*() │
│  → O (SEQ_Q × 512)         │  sV = sK.data() 재해석 (앞 512)   │
│  HEAD_DIM_V = 512          │  get_half_V<0>: 0~256              │
│                            │  get_half_V<1>: 256~512            │
├──────────────────────────────────────────────────────────────────┤
│  O 저장                    │  store_o()                          │
│  IS_NO_SPLIT: FP16로 변환  │  TMA store → params.o_ptr          │
│  IS_SPLIT: FP32 그대로     │  bulk copy → params.oaccum_ptr     │
└──────────────────────────────────────────────────────────────────┘
    """)


# ══════════════════════════════════════════════════════════════════
# PART 4: KV 캐시 메모리 비교
# ══════════════════════════════════════════════════════════════════
def show_memory_comparison():
    SEQ = 32768
    LAYERS = 61   # DeepSeek-V3
    BYTES = 2     # BF16

    # MHA (GPT-style)
    N_HEADS_MHA = 128
    D_HEAD_MHA  = 128
    mha_kv = SEQ * 2 * N_HEADS_MHA * D_HEAD_MHA * LAYERS * BYTES
    print(f"MHA KV cache:  {mha_kv / 1e9:.1f} GB  (seq={SEQ}, {LAYERS} layers)")

    # MLA
    mla_kv = SEQ * HEAD_DIM_K * LAYERS * BYTES   # HEAD_DIM_K=576, KV 통합
    print(f"MLA KV cache:  {mla_kv / 1e9:.1f} GB  (seq={SEQ}, {LAYERS} layers, 576 dim)")
    print(f"압축률: {mha_kv / mla_kv:.1f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("PART 1: 학습 시 MLA Forward")
    print("=" * 60)
    x    = torch.randn(B, SEQ_Q,  D_MODEL, device=device)
    x_kv = torch.randn(B, SEQ_KV, D_MODEL, device=device)
    out_train = mla_forward_training(x, x_kv)
    print(f"Input:  {x.shape}")
    print(f"Output: {out_train.shape}")

    print("\n" + "=" * 60)
    print("PART 2: 추론 시 MLA Forward (Absorption 적용)")
    print("=" * 60)
    W_UQ_absorbed, W_O_absorbed = precompute_absorbed_weights(W)
    kv_cache = torch.randn(B, SEQ_KV, HEAD_DIM_K, device=device)
    x_new    = torch.randn(B, 1, D_MODEL, device=device)
    kv_cache = mla_forward_inference(x_new, kv_cache, W, W_UQ_absorbed, W_O_absorbed)

    print("\n" + "=" * 60)
    print("PART 3: FlashMLA 커널과의 대응")
    print("=" * 60)
    show_kernel_mapping()

    print("=" * 60)
    print("PART 4: KV 캐시 메모리 비교")
    print("=" * 60)
    show_memory_comparison()
