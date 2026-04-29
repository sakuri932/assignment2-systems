from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class FlashAttentionPyTorch(torch.autograd.Function):
    """FlashAttention-2 implemented in pure PyTorch with tiled online softmax."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_q, d = Q.shape
        N_k = K.shape[1]
        scale = 1.0 / math.sqrt(d)

        Br = min(32, N_q)
        Bc = min(32, N_k)

        O = torch.zeros_like(Q)
        L = torch.full((B, N_q), float("-inf"), dtype=Q.dtype, device=Q.device)

        for i_start in range(0, N_q, Br):
            i_end = min(i_start + Br, N_q)
            Q_i = Q[:, i_start:i_end, :]
            O_i = torch.zeros(B, i_end - i_start, d, dtype=Q.dtype, device=Q.device)
            m_i = torch.full((B, i_end - i_start), float("-inf"), dtype=Q.dtype, device=Q.device)
            l_i = torch.zeros(B, i_end - i_start, dtype=Q.dtype, device=Q.device)

            for j_start in range(0, N_k, Bc):
                j_end = min(j_start + Bc, N_k)
                K_j = K[:, j_start:j_end, :]
                V_j = V[:, j_start:j_end, :]

                S_ij = torch.einsum("bid,bjd->bij", Q_i, K_j) * scale

                if is_causal:
                    q_idx = torch.arange(i_start, i_end, device=Q.device)
                    k_idx = torch.arange(j_start, j_end, device=Q.device)
                    S_ij = S_ij.masked_fill((q_idx[:, None] < k_idx[None, :])[None], float("-inf"))

                m_ij = S_ij.max(dim=-1).values
                P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))
                l_ij = P_ij.sum(dim=-1)

                m_i_new = torch.maximum(m_i, m_ij)
                alpha = torch.exp(m_i - m_i_new)
                beta = torch.exp(m_ij - m_i_new)

                O_i = alpha.unsqueeze(-1) * O_i + beta.unsqueeze(-1) * (P_ij @ V_j)
                l_i = alpha * l_i + beta * l_ij
                m_i = m_i_new

            O_i = O_i / l_i.unsqueeze(-1)
            O[:, i_start:i_end, :] = O_i
            L[:, i_start:i_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        B, N_q, d = Q.shape
        N_k = K.shape[1]
        scale = 1.0 / math.sqrt(d)
        is_causal = ctx.is_causal

        Br = min(32, N_q)
        Bc = min(32, N_k)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for i_start in range(0, N_q, Br):
            i_end = min(i_start + Br, N_q)
            Q_i = Q[:, i_start:i_end, :]
            O_i = O[:, i_start:i_end, :]
            dO_i = dO[:, i_start:i_end, :]
            L_i = L[:, i_start:i_end]
            D_i = (dO_i * O_i).sum(dim=-1)
            dQ_i = torch.zeros_like(Q_i)

            for j_start in range(0, N_k, Bc):
                j_end = min(j_start + Bc, N_k)
                K_j = K[:, j_start:j_end, :]
                V_j = V[:, j_start:j_end, :]

                S_ij = torch.einsum("bid,bjd->bij", Q_i, K_j) * scale
                if is_causal:
                    q_idx = torch.arange(i_start, i_end, device=Q.device)
                    k_idx = torch.arange(j_start, j_end, device=Q.device)
                    S_ij = S_ij.masked_fill((q_idx[:, None] < k_idx[None, :])[None], float("-inf"))

                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))
                dV[:, j_start:j_end, :] += torch.einsum("bij,bid->bjd", P_ij, dO_i)
                dP_ij = torch.einsum("bid,bjd->bij", dO_i, V_j)
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))
                dQ_i += torch.einsum("bij,bjd->bid", dS_ij, K_j) * scale
                dK[:, j_start:j_end, :] += torch.einsum("bij,bid->bjd", dS_ij, Q_i) * scale

            dQ[:, i_start:i_end, :] = dQ_i

        return dQ, dK, dV, None


# ---------------------------------------------------------------------------
# Triton kernel definitions (module-level so they're compiled once)
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_q, N_k, scale,
        IS_CAUSAL: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        q_block_idx = tl.program_id(1)

        q_start = q_block_idx * BLOCK_Q
        q_offs = q_start + tl.arange(0, BLOCK_Q)   # (BLOCK_Q,)
        d_offs = tl.arange(0, D)                    # (D,)

        # Load Q block: (BLOCK_Q, D)
        Q_block = tl.load(
            Q_ptr + batch_idx * stride_qb + q_offs[:, None] * stride_qq + d_offs[None, :] * stride_qd,
            mask=q_offs[:, None] < N_q,
            other=0.0,
        )

        m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        O_i = tl.zeros((BLOCK_Q, D), dtype=tl.float32)

        # For causal: only process key blocks where at least the last query has a valid key.
        # With BLOCK_Q == BLOCK_K the first query in the block is always valid in the last
        # processed block, so no all-masked rows arise.
        if IS_CAUSAL:
            k_loop_end = tl.minimum(q_start + BLOCK_Q, N_k)
        else:
            k_loop_end = N_k

        for j_start in range(0, k_loop_end, BLOCK_K):
            k_offs = j_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)

            # Load K^T: shape (D, BLOCK_K) = K[b, k_offs, :]^T
            # K[b, k, d]  →  K_T[d, k]  address: K_ptr + b*sb + d*sd + k*sk
            K_T = tl.load(
                K_ptr + batch_idx * stride_kb
                + d_offs[:, None] * stride_kd
                + k_offs[None, :] * stride_kk,
                mask=k_offs[None, :] < N_k,
                other=0.0,
            )  # (D, BLOCK_K)

            # Load V block: (BLOCK_K, D)
            V_block = tl.load(
                V_ptr + batch_idx * stride_vb
                + k_offs[:, None] * stride_vk
                + d_offs[None, :] * stride_vd,
                mask=k_offs[:, None] < N_k,
                other=0.0,
            )  # (BLOCK_K, D)

            # S = Q @ K^T : (BLOCK_Q, D) @ (D, BLOCK_K) = (BLOCK_Q, BLOCK_K)
            S = tl.dot(Q_block, K_T) * scale

            # Mask out-of-bounds keys
            S = tl.where(k_offs[None, :] < N_k, S, float("-inf"))

            # Causal mask: query i attends to key j only if j <= i (absolute indices)
            if IS_CAUSAL:
                S = tl.where(q_offs[:, None] >= k_offs[None, :], S, float("-inf"))

            # Online softmax update
            m_ij = tl.max(S, axis=1)              # (BLOCK_Q,)
            P_ij = tl.exp(S - m_ij[:, None])      # (BLOCK_Q, BLOCK_K)
            l_ij = tl.sum(P_ij, axis=1)            # (BLOCK_Q,)

            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta  = tl.exp(m_ij - m_i_new)

            O_i = alpha[:, None] * O_i + beta[:, None] * tl.dot(P_ij, V_block)
            l_i = alpha * l_i + beta * l_ij
            m_i = m_i_new

        # Normalize and compute logsumexp
        O_i = O_i / l_i[:, None]
        L_i = m_i + tl.log(l_i)

        # Store O: (BLOCK_Q, D)
        tl.store(
            O_ptr + batch_idx * stride_ob + q_offs[:, None] * stride_oq + d_offs[None, :] * stride_od,
            O_i,
            mask=q_offs[:, None] < N_q,
        )
        # Store L: (BLOCK_Q,)
        tl.store(
            L_ptr + batch_idx * stride_lb + q_offs * stride_lq,
            L_i,
            mask=q_offs < N_q,
        )

    _TRITON_AVAILABLE = True

except ImportError:
    _TRITON_AVAILABLE = False


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 using a Triton forward kernel; backward is PyTorch tiled."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if not _TRITON_AVAILABLE or not Q.is_cuda:
            # Triton requires both the library and a CUDA tensor; fall back to PyTorch tiled impl.
            return FlashAttentionPyTorch.forward(ctx, Q, K, V, is_causal)

        B, N_q, d = Q.shape
        N_k = K.shape[1]
        scale = 1.0 / math.sqrt(d)

        BLOCK_Q = 64
        BLOCK_K = 64
        # d must be a power of 2 and match the constexpr D in the kernel
        D = d

        O = torch.zeros_like(Q)
        L = torch.full((B, N_q), float("-inf"), dtype=Q.dtype, device=Q.device)

        grid = (B, triton.cdiv(N_q, BLOCK_Q))
        _flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k, scale,
            IS_CAUSAL=is_causal,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, D=D,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        B, N_q, d = Q.shape
        N_k = K.shape[1]
        scale = 1.0 / math.sqrt(d)
        is_causal = ctx.is_causal

        Br = min(64, N_q)
        Bc = min(64, N_k)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for i_start in range(0, N_q, Br):
            i_end = min(i_start + Br, N_q)
            Q_i  = Q[:, i_start:i_end, :]
            O_i  = O[:, i_start:i_end, :]
            dO_i = dO[:, i_start:i_end, :]
            L_i  = L[:, i_start:i_end]
            D_i  = (dO_i * O_i).sum(dim=-1)
            dQ_i = torch.zeros_like(Q_i)

            j_limit = (i_end if is_causal else N_k)
            for j_start in range(0, j_limit, Bc):
                j_end = min(j_start + Bc, N_k)
                K_j = K[:, j_start:j_end, :]
                V_j = V[:, j_start:j_end, :]

                S_ij = torch.einsum("bid,bjd->bij", Q_i, K_j) * scale
                if is_causal:
                    q_idx = torch.arange(i_start, i_end, device=Q.device)
                    k_idx = torch.arange(j_start, j_end, device=Q.device)
                    S_ij = S_ij.masked_fill((q_idx[:, None] < k_idx[None, :])[None], float("-inf"))

                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))
                dV[:, j_start:j_end, :] += torch.einsum("bij,bid->bjd", P_ij, dO_i)
                dP_ij = torch.einsum("bid,bjd->bij", dO_i, V_j)
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))
                dQ_i += torch.einsum("bij,bjd->bid", dS_ij, K_j) * scale
                dK[:, j_start:j_end, :] += torch.einsum("bij,bid->bjd", dS_ij, Q_i) * scale

            dQ[:, i_start:i_end, :] = dQ_i

        return dQ, dK, dV, None
