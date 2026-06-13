"""STUDENT FILE: implement the three block-sparse rung functions.

Implement these three functions from the spec in ALGORITHMS.md -- no reference
code is shipped:

  dsd_matmul             (A1) block-sparse (BCSR) A @ dense B -> dense C
  sparse_flash_forward   (A2) block-sparse flash attention forward
  sparse_flash_backward  (A3) block-sparse flash attention backward

Your functions must match the signatures below: the SHAPES and DTYPES of the
inputs and outputs (each docstring states them; ALGORITHMS.md sec 0.1 collects
them). EVERYTHING ELSE IS YOURS -- how many @triton.jit kernels you write, the
grid, the (B, H) flatten, strides, output allocation, and the launch/tuning. The
grader asserts the returned shapes and dtypes, then checks correctness against an
fp64 reference.

ALGORITHMS.md is the complete spec: the BCSR layout and its two transpose views,
what each output equals, and the five backward equations.

When `python sanity_check.py` passes all three rungs, you're done.
"""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


# =============================================================================
# A1 -- DSD: block-sparse (BCSR) A @ dense B -> dense C
# =============================================================================

@triton.jit
def _dsd_kernel(
    values_ptr, row_offsets_ptr, col_indices_ptr, B_ptr, C_ptr,
    M, K, N, block,
    stride_vb, stride_vm, stride_vk,        # values (nnz, block, block)
    stride_bk, stride_bn,                   # B (K, N)
    stride_cm, stride_cn,                   # C (M, N)
    BLK: tl.constexpr,                      # the BCSR block size (== `block`)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)               # which row-tile (sub-tile of a block-row)
    pid_n = tl.program_id(1)               # which N-tile

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # BLOCK_M divides `block`, so this row-tile lies entirely in one BCSR block-row.
    block_row = m_start // block
    m_local = m_start - block_row * block

    row_lo = tl.load(row_offsets_ptr + block_row).to(tl.int32)
    row_hi = tl.load(row_offsets_ptr + block_row + 1).to(tl.int32)

    offs_m = m_local + tl.arange(0, BLOCK_M)      # rows inside the value block
    offs_n = n_start + tl.arange(0, BLOCK_N)      # columns of B / C

    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for idx in range(row_lo, row_hi):
        kblk = tl.load(col_indices_ptr + idx).to(tl.int32)
        # Contract the BLK-wide block in BLOCK_K-sized chunks so the on-chip
        # operand tiles stay within shared memory for large block sizes.
        for kk in range(0, BLK, BLOCK_K):
            offs_k = kk + tl.arange(0, BLOCK_K)

            v_ptrs = (values_ptr + idx * stride_vb
                      + offs_m[:, None] * stride_vm + offs_k[None, :] * stride_vk)
            a = tl.load(v_ptrs)                     # (BLOCK_M, BLOCK_K) fp32

            b_rows = kblk * block + offs_k
            b_ptrs = B_ptr + b_rows[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)   # (BLOCK_K, BLOCK_N)

            acc += tl.dot(a, b, allow_tf32=False)

    c_rows = m_start + tl.arange(0, BLOCK_M)
    c_ptrs = C_ptr + c_rows[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (c_rows[:, None] < M) & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    """A1 -- block-sparse C = A @ B. See ALGORITHMS.md sec 1-2.

    Inputs:
      values         (nnz, block, block)  fp32   A's live blocks, row-major
      row_offsets    (M//block + 1,)      int32  per block-row prefix sum of nnz
      column_indices (nnz,)               int32  K-block of each live block
      B              (K, N)               fp32   dense right operand
      M, K, N, block                      ints   dims and block size
    Returns:
      C              (M, N)               fp32

    fp32 throughout, allow_tf32=False.
    """
    assert values.dtype == torch.float32 and B.dtype == torch.float32
    values = values.contiguous()
    B = B.contiguous()
    row_offsets = row_offsets.contiguous()
    column_indices = column_indices.contiguous()

    C = torch.zeros((M, N), device=B.device, dtype=torch.float32)

    # BLOCK_M must divide `block` so each row-tile maps to one block-row.
    BLOCK_M = block
    while BLOCK_M > 128:
        BLOCK_M //= 2
    BLOCK_N = 64
    # Inner contraction tile: a divisor of `block`, kept small so the on-chip
    # operand tiles fit in shared memory even for block=256.
    BLOCK_K = min(block, 64)

    grid = (M // BLOCK_M, triton.cdiv(N, BLOCK_N))
    _dsd_kernel[grid](
        values, row_offsets, column_indices, B, C,
        M, K, N, block,
        values.stride(0), values.stride(1), values.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLK=block,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C


# =============================================================================
# A2 -- sparse flash forward
# =============================================================================

@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    q_row_offsets, q_col_indices,
    qk_scale,                               # sm_scale * LOG2E
    T,
    stride_h, stride_t, stride_d,           # Q/K/V/O  (BH, T, d)
    stride_lh, stride_lt,                   # L        (BH, T)
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    q_start = pid_q * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    q_mask = offs_q < T

    qh = Q_ptr + pid_bh * stride_h
    kh = K_ptr + pid_bh * stride_h
    vh = V_ptr + pid_bh * stride_h
    oh = O_ptr + pid_bh * stride_h

    q_ptrs = qh + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)

    m_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)

    lo = tl.load(q_row_offsets + pid_q).to(tl.int32)
    hi = tl.load(q_row_offsets + pid_q + 1).to(tl.int32)

    for idx in range(lo, hi):
        kblk = tl.load(q_col_indices + idx).to(tl.int32)
        offs_k = kblk * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < T

        k_ptrs = kh + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
        v_ptrs = vh + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
        k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(q, tl.trans(k))                       # (BQ, BK) fp32
        s2 = qk * qk_scale
        s2 = tl.where(k_mask[None, :], s2, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s2, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s2 - m_new[:, None])                  # (BQ, BK)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = acc / l_safe[:, None]

    o_ptrs = oh + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    tl.store(o_ptrs, o.to(tl.float16), mask=q_mask[:, None] & d_mask[None, :])

    L_i = m_i + tl.log2(l_safe)
    l_ptrs = L_ptr + pid_bh * stride_lh + offs_q * stride_lt
    tl.store(l_ptrs, L_i, mask=q_mask)


def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    """A2 -- block-sparse flash attention forward. See ALGORITHMS.md sec 1, 3.

    Inputs:
      Q, K, V        (B, H, T, d)         fp16
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query
      q_col_indices  (nnz,)               int32  block i, its live key blocks j
      sm_scale       float                       1/sqrt(d)
      BLOCK_Q, BLOCK_K  ints                     == block (the mask granularity)
    Returns:
      O              (B, H, T, d)         fp16
      L              (B, H, T)            fp32   log2 of the softmax denominator (sec 3)

    See ALGORITHMS.md sec 3 for O and L.
    """
    B, H, T, d = Q.shape
    BH = B * H
    Qf = Q.contiguous().view(BH, T, d)
    Kf = K.contiguous().view(BH, T, d)
    Vf = V.contiguous().view(BH, T, d)

    O = torch.empty((BH, T, d), device=Q.device, dtype=torch.float16)
    L = torch.empty((BH, T), device=Q.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(d)
    qk_scale = sm_scale * LOG2E
    n_q = T // BLOCK_Q
    grid = (n_q, BH)

    _fwd_kernel[grid](
        Qf, Kf, Vf, O, L,
        q_row_offsets.contiguous(), q_col_indices.contiguous(),
        qk_scale, T,
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        L.stride(0), L.stride(1),
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, D=d,
        num_warps=4,
    )
    return O.view(B, H, T, d), L.view(B, H, T)


# =============================================================================
# A3 -- sparse flash backward
# =============================================================================

@triton.jit
def _bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr, dK_ptr, dV_ptr,
    k_row_offsets, k_col_indices,
    qk_scale, sm_scale, T,
    stride_h, stride_t, stride_d,
    stride_lh, stride_lt,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    pid_k = tl.program_id(0)               # key block j
    pid_bh = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    k_mask = offs_k < T

    qh = Q_ptr + pid_bh * stride_h
    kh = K_ptr + pid_bh * stride_h
    vh = V_ptr + pid_bh * stride_h
    oh = O_ptr + pid_bh * stride_h
    doh = dO_ptr + pid_bh * stride_h

    k_ptrs = kh + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
    v_ptrs = vh + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
    Kj = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)   # (BK, BD)
    Vj = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)

    dk = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)
    dv = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)

    lo = tl.load(k_row_offsets + pid_k).to(tl.int32)
    hi = tl.load(k_row_offsets + pid_k + 1).to(tl.int32)

    for idx in range(lo, hi):
        iblk = tl.load(k_col_indices + idx).to(tl.int32)   # query block i
        offs_q = iblk * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = offs_q < T

        q_ptrs = qh + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
        do_ptrs = doh + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
        o_ptrs = oh + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
        Qi = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        dOi = tl.load(do_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        Oi = tl.load(o_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        Li = tl.load(L_ptr + pid_bh * stride_lh + offs_q * stride_lt,
                     mask=q_mask, other=0.0)

        qk = tl.dot(Qi, tl.trans(Kj))                      # (BQ, BK)
        s2 = qk * qk_scale
        s2 = tl.where(k_mask[None, :], s2, -float("inf"))
        p = tl.exp2(s2 - Li[:, None])                      # (BQ, BK) softmax probs
        p = tl.where(q_mask[:, None], p, 0.0)

        # dV_j += P^T @ dO_i
        dv += tl.dot(tl.trans(p.to(tl.float16)), dOi)

        # D_i = rowsum(dO_i * O_i)
        Di = tl.sum(dOi.to(tl.float32) * Oi.to(tl.float32), axis=1)   # (BQ,)

        dp = tl.dot(dOi, tl.trans(Vj))                     # (BQ, BK)
        ds = p * (dp - Di[:, None])                        # (BQ, BK)

        # dK_j += dS^T @ Q_i  (scaled by sm_scale at the end)
        dk += tl.dot(tl.trans(ds.to(tl.float16)), Qi)

    dk = dk * sm_scale

    dk_ptrs = dK_ptr + pid_bh * stride_h + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
    dv_ptrs = dV_ptr + pid_bh * stride_h + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
    store_mask = k_mask[:, None] & d_mask[None, :]
    tl.store(dk_ptrs, dk.to(tl.float16), mask=store_mask)
    tl.store(dv_ptrs, dv.to(tl.float16), mask=store_mask)


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr, dQ_ptr,
    q_row_offsets, q_col_indices,
    qk_scale, sm_scale, T,
    stride_h, stride_t, stride_d,
    stride_lh, stride_lt,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr, D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    q_mask = offs_q < T

    qh = Q_ptr + pid_bh * stride_h
    kh = K_ptr + pid_bh * stride_h
    vh = V_ptr + pid_bh * stride_h
    oh = O_ptr + pid_bh * stride_h
    doh = dO_ptr + pid_bh * stride_h

    q_ptrs = qh + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    do_ptrs = doh + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    o_ptrs = oh + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    Qi = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    dOi = tl.load(do_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    Oi = tl.load(o_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    Li = tl.load(L_ptr + pid_bh * stride_lh + offs_q * stride_lt,
                 mask=q_mask, other=0.0)

    Di = tl.sum(dOi.to(tl.float32) * Oi.to(tl.float32), axis=1)   # (BQ,)

    dq = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)

    lo = tl.load(q_row_offsets + pid_q).to(tl.int32)
    hi = tl.load(q_row_offsets + pid_q + 1).to(tl.int32)

    for idx in range(lo, hi):
        jblk = tl.load(q_col_indices + idx).to(tl.int32)   # key block j
        offs_k = jblk * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < T

        k_ptrs = kh + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
        v_ptrs = vh + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
        Kj = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
        Vj = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(Qi, tl.trans(Kj))                      # (BQ, BK)
        s2 = qk * qk_scale
        s2 = tl.where(k_mask[None, :], s2, -float("inf"))
        p = tl.exp2(s2 - Li[:, None])                      # (BQ, BK)

        dp = tl.dot(dOi, tl.trans(Vj))                     # (BQ, BK)
        ds = p * (dp - Di[:, None])                        # (BQ, BK)

        dq += tl.dot(ds.to(tl.float16), Kj)                # (BQ, BD)

    dq = dq * sm_scale

    dq_ptrs = dQ_ptr + pid_bh * stride_h + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    tl.store(dq_ptrs, dq.to(tl.float16), mask=q_mask[:, None] & d_mask[None, :])


def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,   # key-block view (sec 1)
                          q_row_offsets, q_col_indices,   # query-block view (sec 1)
                          sm_scale, BLOCK_Q, BLOCK_K):
    """A3 -- block-sparse flash attention backward. See ALGORITHMS.md sec 1, 4.

    Inputs:
      Q, K, V, O, dO (B, H, T, d)         fp16   O, dO are the forward output and its grad
      L              (B, H, T)            fp32   the forward residual
      k_row_offsets  (T//block + 1,)      int32  key-block view: for key block j,
      k_col_indices  (nnz,)               int32  the query blocks i that attend it
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query block i,
      q_col_indices  (nnz,)               int32  its key blocks j (same as forward)
      sm_scale       float
      BLOCK_Q, BLOCK_K  ints                     == block
    Returns:
      dQ, dK, dV     (B, H, T, d)         fp16

    See ALGORITHMS.md sec 4 for the five gradient equations.
    """
    B, H, T, d = Q.shape
    BH = B * H
    Qf = Q.contiguous().view(BH, T, d)
    Kf = K.contiguous().view(BH, T, d)
    Vf = V.contiguous().view(BH, T, d)
    Of = O.contiguous().view(BH, T, d)
    dOf = dO.contiguous().view(BH, T, d)
    Lf = L.contiguous().view(BH, T)

    dQ = torch.empty((BH, T, d), device=Q.device, dtype=torch.float16)
    dK = torch.empty((BH, T, d), device=Q.device, dtype=torch.float16)
    dV = torch.empty((BH, T, d), device=Q.device, dtype=torch.float16)

    BLOCK_D = triton.next_power_of_2(d)
    qk_scale = sm_scale * LOG2E
    n_q = T // BLOCK_Q
    n_k = T // BLOCK_K

    sh, st, sd = Qf.stride(0), Qf.stride(1), Qf.stride(2)
    slh, slt = Lf.stride(0), Lf.stride(1)

    _bwd_dkdv_kernel[(n_k, BH)](
        Qf, Kf, Vf, Of, Lf, dOf, dK, dV,
        k_row_offsets.contiguous(), k_col_indices.contiguous(),
        qk_scale, sm_scale, T,
        sh, st, sd, slh, slt,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, D=d,
        num_warps=4,
    )

    _bwd_dq_kernel[(n_q, BH)](
        Qf, Kf, Vf, Of, Lf, dOf, dQ,
        q_row_offsets.contiguous(), q_col_indices.contiguous(),
        qk_scale, sm_scale, T,
        sh, st, sd, slh, slt,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, D=d,
        num_warps=4,
    )

    return (dQ.view(B, H, T, d), dK.view(B, H, T, d), dV.view(B, H, T, d))
