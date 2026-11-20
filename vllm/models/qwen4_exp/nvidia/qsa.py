# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

import os

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import ClassVar, cast

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.qwen3_vl import mrope_positions_factor
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    get_dtype_size,
    kv_cache_dtype_str_to_dtype,
    nvfp4_kv_cache_full_dim,
    nvfp4_split_data_scale,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImplBase,
    AttentionType,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)

from ..common.qsa import Qwen4ExpQSAMetadata, Qwen4ExpQSAMetadataBuilder
from ..common.qsa_indexer import QSAIndexerMetadata, qsa_kv_cache_dtype
from . import model
from .ops.qsa_indexer import _TOPK_WORKSPACE_BYTES
from .qsa_indexer import QSAIndexer


class Qwen4ExpQSAQKVIndexerLinear(MergedColumnParallelLinear):
    """Pack sharded Q/gate, K, V and replicated indexer Q/K in one GEMM."""

    def __init__(
        self,
        qkv_proj: QKVParallelLinear,
        index_size: int,
        quant_config: QuantizationConfig | None,
    ) -> None:
        self.num_kv_head_replicas = qkv_proj.num_kv_head_replicas
        super().__init__(
            input_size=qkv_proj.input_size,
            output_sizes=[*qkv_proj.output_sizes, index_size * qkv_proj.tp_size],
            bias=False,
            params_dtype=qkv_proj.params_dtype,
            quant_config=quant_config,
            prefix=qkv_proj.prefix,
        )

    def _load_shard(
        self,
        loader: Callable[..., None],
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == 3:
            shard_rank = 0
        elif loaded_shard_id in (1, 2):
            shard_rank = tp_rank // self.num_kv_head_replicas
        else:
            shard_rank = tp_rank
        self.tp_rank = shard_rank
        if param_tp_rank is not None:
            param.tp_rank = shard_rank
        try:
            loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(super().weight_loader, param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(
            super().weight_loader_v2, param, loaded_weight, loaded_shard_id
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        def remap_shards():
            for name, weight in weights:
                shard_id = getattr(weight, "shard_id", None)
                if isinstance(shard_id, str):
                    weight = weight.detach()
                    weight.shard_id = {"q": 0, "k": 1, "v": 2}[shard_id]
                yield name, weight

        return super().load_weights(remap_shards())

# KV cache dtypes accepted for the QSA main cache. fp8_e5m2 is left out on
# purpose (smaller mantissa, no benefit for K/V); nvfp4_4over6 changes the
# block-scale search in the writer and has not been evaluated on this path.
# The indexer caches are not affected: they are constructed in bf16 (the
# key-state cache packs int64 mRoPE positions into bf16 cells).
_QSA_KV_CACHE_DTYPES: tuple[str, ...] = ("auto", "bfloat16", "fp8", "fp8_e4m3", "nvfp4")

# Storage dtypes of the main cache. Quantized caches are allocated as uint8
# (STR_DTYPE_TO_TORCH_DTYPE); the bytes are reinterpreted right before the
# kernel.
_QSA_KV_STORAGE_DTYPES = (torch.bfloat16, torch.uint8, torch.float8_e4m3fn)

# Block-scale layout written by reshape_and_cache_flash for nvfp4: K block
# scales are stored linearly, V block scales are swizzled for the TRT-LLM
# decode kernels. The read kernel follows the same convention.
_NVFP4_V_SCALE_SWIZZLED = True
# A/B switch: QSA_CAKE=1 routes the NVFP4 main attention through FlashInfer's
# Cake kernels (`flashinfer.sparse.qsa_sparse_paged_attention`, sm_120a) and
# switches the cache writer to FlashInfer's
# `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping`. Cake reads *linear*
# V block scales, while `reshape_and_cache_flash` writes the TRT-LLM swizzle,
# so the two must move together — writer and reader are switched as a pair.
_QSA_CAKE = os.environ.get("QSA_CAKE") == "1"
# A/B switch: QSA_H1_DISABLE=1 skips the H1 dequant-avanse prefill path and
# runs the in-kernel NVFP4 read instead (smoke-test reference).
_QSA_H1_DISABLE = os.environ.get("QSA_H1_DISABLE") == "1"


def _nvfp4_attention_spec(spec: AttentionSpec) -> AttentionSpec:
    """Apply the nvfp4 page geometry to a QSA main-cache spec.

    K and V get separate head slots (all K heads first, then all V heads, the
    order ``kv_cache.split(num_kv_heads, dim=1)`` expects) and a cell shrinks
    from ``head_size + head_size_v`` bf16 values to
    ``head_size // 2 + head_size // 16`` bytes of fp4 data plus fp8 block
    scales. Same geometry as ``FlashInferBackend.customize_spec``. Idempotent,
    so it does not matter whether the model runner calls ``customize_spec``
    in addition to the owner applying it.
    """
    if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
        return spec
    hs_k = nvfp4_kv_cache_full_dim(spec.head_size)
    hs_v = nvfp4_kv_cache_full_dim(spec.head_size_v)
    assert hs_k == hs_v, "nvfp4 with asymmetric K/V head sizes not yet supported"
    return replace(
        spec,
        num_head_slots=2 * spec.num_kv_heads,
        state_content_bytes=hs_k * get_dtype_size(spec.dtype),
    )


def _nvfp4_cache_views(
    kv_cache: torch.Tensor, num_kv_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a packed nvfp4 cache ``[B, 2 * H, N, 144]`` into strided views.

    Returns ``(k_data, k_scales, v_data, v_scales)``, each ``[B, N, H, X]``
    with X = ``head_size // 2`` data bytes or ``head_size // 16`` block scales.
    The byte arithmetic is ``nvfp4_split_data_scale``, the same helper the
    writer side uses, so both sides see one layout by construction. Scales are
    returned as raw uint8 bits; the kernel decodes them.
    """
    k_side = kv_cache[:, :num_kv_heads].transpose(1, 2)
    v_side = kv_cache[:, num_kv_heads:].transpose(1, 2)
    k_data, k_scales = nvfp4_split_data_scale(k_side)
    v_data, v_scales = nvfp4_split_data_scale(v_side)
    return k_data, k_scales.view(torch.uint8), v_data, v_scales.view(torch.uint8)


class Qwen4ExpQSABackend(AttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # fp8/fp8_e4m3: e4m3 bytes in a uint8 cache, written by reshape_and_cache
    # with the layer's per-tensor scales and dequantized on load inside the QSA
    # Triton kernel.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "nvfp4",
    ]

    # The owner writes K/V before calling the sparse attention implementation.
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAImpl]:
        return Qwen4ExpQSAImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAImpl(AttentionImplBase[Qwen4ExpQSAMetadata]):
    """Run paged sparse GQA with the QSA Triton kernel."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        kv_cache_dtype: str,
    ) -> None:
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if kv_cache_dtype not in _QSA_KV_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP8-E4M3 or NVFP4 main KV cache"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        # nvfp4 is a separate branch (page geometry, writer, read kernel), not
        # fp8 with a different dtype.
        self.kv_cache_nvfp4 = kv_cache_dtype.startswith("nvfp4")
        self.kv_cache_fp8 = (
            is_quantized_kv_cache(kv_cache_dtype) and not self.kv_cache_nvfp4
        )
        # Strided views on the packed nvfp4 cache, built once per cache buffer.
        # as_strided is metadata only, but it would otherwise run in every
        # decode step, and the pointers stay stable across CUDA graph capture.
        self._nvfp4_views: tuple[int, tuple[torch.Tensor, ...]] | None = None
        # H1 dequant-avance scratch: one shared BF16 K/V paged-cache pair,
        # keyed on pool geometry (all layers share the pool, so one pair is
        # reused: each layer's dequant overwrites it right before that
        # layer's attention kernel runs).
        self._nvfp4_scratch: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def _nvfp4_views_for(self, kv_cache: torch.Tensor) -> tuple[torch.Tensor, ...]:
        key = kv_cache.data_ptr()
        cached = self._nvfp4_views
        if cached is None or cached[0] != key:
            cached = (key, _nvfp4_cache_views(kv_cache, self.num_kv_heads))
            self._nvfp4_views = cached
        return cached[1]

    def _nvfp4_scratch_for(
        self, k_data: torch.Tensor, v_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared BF16 scratch paged caches for the dequant-avance prefill
        path: same [B, PAGE_SIZE, H, ...] geometry as the packed views with a
        full-width last axis. Keyed on pool geometry (num_pages, page_size,
        heads, head_dim, device), not the layer tensor: every layer shares
        the one pool, so a single pair is overwritten per layer in stream
        order — no cross-layer overlap.
        """
        key = (
            k_data.shape[0],
            k_data.shape[1],
            k_data.shape[2],
            2 * k_data.shape[3],
            v_data.shape[3] * 2,
            k_data.device.index,
        )
        scratch = self._nvfp4_scratch.get(key)
        if scratch is None:
            shape_k = (*k_data.shape[:3], 2 * k_data.shape[3])
            shape_v = (*v_data.shape[:3], 2 * v_data.shape[3])
            scratch = (
                torch.empty(shape_k, dtype=torch.bfloat16, device=k_data.device),
                torch.empty(shape_v, dtype=torch.bfloat16, device=v_data.device),
            )
            self._nvfp4_scratch.clear()
            self._nvfp4_scratch[key] = scratch
        return scratch

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Scatter K/V using the same layout and scales as the fused writer."""
        if not self.kv_cache_nvfp4:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
            return
        # nvfp4 stores K and V in separate head slots of
        # head_size // 2 + head_size // 16 bytes, so the cache is cut by head
        # slot here: (B, 2n, N, 144) -> two views (B, N, n, 144), the 4D shape
        # reshape_and_cache_flash expects for nvfp4.
        n = self.num_kv_heads
        if _QSA_CAKE:
            # Cake reads linear V block scales; reshape_and_cache_flash writes
            # the TRT-LLM swizzle. FlashInfer's writer stores both sides
            # linearly, which is exactly the format Cake was validated against.
            from flashinfer import (
                nvfp4_quantize_append_paged_kv_cache_with_slot_mapping as append,
            )

            k_view, k_sf, v_view, v_sf = self._nvfp4_views_for(kv_cache)
            append(
                key,
                value,
                slot_mapping,
                (k_view, v_view),
                (k_sf.view(torch.float8_e4m3fn), v_sf.view(torch.float8_e4m3fn)),
                layer._k_scale,
                layer._v_scale,
                kv_layout="NHD",
            )
            return
        ops.reshape_and_cache_flash(
            key,
            value,
            kv_cache[:, :n].transpose(1, 2),
            kv_cache[:, n:].transpose(1, 2),
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Qwen4ExpQSAMetadata,
        output: torch.Tensor,
        *,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Run sparse attention over the owner's indexer-selected K/V rows."""
        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        k_scale = v_scale = None
        k_scale_cache: torch.Tensor | None = None
        v_scale_cache: torch.Tensor | None = None
        if self.kv_cache_nvfp4:
            # Packed rows carry data and block scales for K and V in separate
            # head slots; there is nothing to split at head_size.
            key_cache, k_scale_cache, value_cache, v_scale_cache = (
                self._nvfp4_views_for(kv_cache)
            )
            # Per-layer dequant scales as 1-element device buffers
            # (set_default_quant_scales, register_buffer=True): the kernel's
            # quantized branches read them device-side, so CUDA graph
            # capture stays copy-free.
            k_scale = getattr(layer, "_k_scale", None)
            v_scale = getattr(layer, "_v_scale", None)
        else:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
            key_cache = canonicalize_singleton_dim_strides(key_cache)
            value_cache = canonicalize_singleton_dim_strides(value_cache)
            if self.kv_cache_fp8:
                # The cache is allocated as uint8; on sm_89+ reinterpret the
                # e4m3 bytes (same itemsize, so shape and strides preserved).
                # Below sm_89 the kernel byte-decodes the raw uint8 storage.
                if current_platform.has_device_capability(89):
                    key_cache = key_cache.view(torch.float8_e4m3fn)
                    value_cache = value_cache.view(torch.float8_e4m3fn)
                # Host-side per-tensor dequant scales (Python floats), folded
                # into the kernel's scales.
                k_scale = layer._k_scale_float
                v_scale = layer._v_scale_float
        if query.dtype != torch.bfloat16 or key_cache.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.uint8,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 Q and BF16, FP8-e4m3 or NVFP4 K/V"
            )

        if _QSA_CAKE and self.kv_cache_nvfp4:
            # Cake path: FlashInfer's native sm_120a NVFP4 sparse attention reads
            # the packed cache directly (0.41-0.60x the FP8 Triton time in the
            # isolated bench), so neither the BF16 scratch of H1 nor the
            # in-kernel Triton E2M1 unpack is needed. `_nvfp4_views_for` returns
            # uint8 scale views; Cake wants them as float8_e4m3fn.
            from flashinfer.sparse import qsa_sparse_paged_attention as cake_qsa

            # The unfused gate projection hands back a flat [tokens, H * D]
            # tensor (and the qsa_prepare path a 3D one); Cake validates the
            # shape against q, so fold it the same way the Triton path does.
            gate_view = (
                output_gate[:num_tokens]
                .reshape(query[:num_tokens].shape)
                .contiguous()
            )
            cake_qsa(
                query[:num_tokens],
                key_cache,
                value_cache,
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                use_prefill_config,
                output[:num_tokens],
                float(layer._k_scale_float),
                float(layer._v_scale_float),
                output_gate=gate_view,
                k_sf=k_scale_cache.view(torch.float8_e4m3fn),
                v_sf=v_scale_cache.view(torch.float8_e4m3fn),
                backend="cake",
            )
            return output

        if self.kv_cache_nvfp4 and use_prefill_config and not _QSA_H1_DISABLE:
            # H1 dequant-avanse: the in-kernel E2M1 unpack is the read-side
            # bottleneck (pins num_stages=1 and kills split-kv on sm_120).
            # Decode the live pages ahead into shared BF16 scratch and run
            # the fast BF16 branch (num_stages=2, tuned sm_120 config).
            # Decode/MTP-verify keeps the in-kernel path: tiny sparse tiles,
            # no scratch traffic, zero regression surface.
            from .ops.qsa import qsa_nvfp4_dequant_pages

            scratch_k, scratch_v = self._nvfp4_scratch_for(key_cache, value_cache)
            qsa_nvfp4_dequant_pages(
                key_cache,
                value_cache,
                k_scale_cache,
                v_scale_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                k_scale,
                v_scale,
                scratch_k,
                scratch_v,
                max_seq_len=attn_metadata.max_seq_len,
                v_scale_swizzled=_NVFP4_V_SCALE_SWIZZLED,
            )
            key_cache, value_cache = scratch_k, scratch_v
            k_scale_cache = v_scale_cache = None
            k_scale = v_scale = None

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            use_prefill_config,
            output[:num_tokens],
            k_scale=k_scale,
            v_scale=v_scale,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            v_scale_swizzled=_NVFP4_V_SCALE_SWIZZLED,
            output_gate=output_gate[:num_tokens],
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA indexer branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        topk_indices_buffer: torch.Tensor,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in _QSA_KV_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP8-E4M3 or NVFP4 main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        # Decode/verify batches have at most 1 + num_spec query tokens per
        # request; use_prefill_config (max_query_len > this) steers the
        # config table. Shorter batches take the decode profile — harmless,
        # the difference is tile-shape tuning, not correctness.
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            mrope_positions_factor=mrope_positions_factor(vllm_config),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = qsa_kv_cache_dtype(cache_config, prefix)
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16, FP8-e4m3 (uint8) or NVFP4 (uint8) cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSABackend
        self.impl: Qwen4ExpQSAImpl = Qwen4ExpQSAImpl(  # type: ignore[assignment]
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            self.kv_cache_dtype,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        # One launch does the indexer prepare, the main QK-norm/RoPE/gate and
        # the main K/V cache write (see QSAIndexer.forward); otherwise all of
        # them take the separate kernels. The fused writer receives an fp8
        # pointer, which Triton cannot lower below sm_89.
        # H1 overlay (flashnext smoke 10-31, backup qsa.py.bak-fused): the
        # fused qsa_prepare derives the main head geometry from the packed
        # NVFP4 cache cell (main_head_dim = shape[-1] // 2,
        # num_main_kv_heads = shape[2]), which only round-trips for the fp8
        # K|V-stacked layout. With NVFP4 head_dim 256 (cell 144) the qkv
        # width is not a multiple of 2 * cell, so its shape assert fires on
        # the first prefill (qsa_prepare.py:572) and the whole PP pipeline
        # deadlocks. The fused kernel itself targets the fp8 cache layout,
        # so a host-side derive fix could still mis-index the NVFP4 cell
        # silently. Force the separate-kernel reference path: its
        # do_kv_cache_update is NVFP4-aware and layout-correct, and both
        # smoke cases (a/b) share it, so the a/b comparison stays valid.
        self.use_fused_qsa_prepare = False
        # Without a quant config here (BF16, or ModelOpt FP4, which keeps
        # attention in BF16) the fused layer is unquantized. Otherwise the
        # indexer shares the q/k/v GEMM only if it uses the same quant method.
        fused_quant_config = model.without_modelopt_fp4(quant_config)
        projection_method = type(self.qkv_proj.quant_method)
        self.fuse_indexer_projection = vllm_config.lora_config is None and (
            fused_quant_config is None
            or projection_method is type(self.indexer.index_qk_proj.quant_method)
        )
        if self.fuse_indexer_projection:
            self.index_qk_size = self.indexer.index_qk_proj.output_size
            self.qkv_proj = Qwen4ExpQSAQKVIndexerLinear(
                self.qkv_proj,
                self.index_qk_size,
                fused_quant_config,
            )
            del self.indexer.index_qk_proj

        # MTP skip_topk steps reuse rows frozen from step 0; the valid count
        # is a row column, so compaction/reuse keep it paired with the content.
        self.topk_indices_buffer = topk_indices_buffer

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def process_weights_after_loading(self) -> None:
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = None
        if self.fuse_indexer_projection:
            mapper = WeightsMapper(
                orig_to_new_stacked={
                    "indexer.index_qk_proj.": ("qkv_proj.", 3),
                }
            )
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # The owner builds its own spec, so the nvfp4 geometry is applied here
        # as well as in customize_spec; both are idempotent.
        return _nvfp4_attention_spec(
            FullAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                head_size_v=self.head_dim,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            )
        )

    @eager_break_during_capture
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        output_gate: torch.Tensor | None,
        qkv: torch.Tensor,
    ) -> None:
        # query/key/value/output_gate are None when the fused prepare runs
        # inside the indexer launch.
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            # Profiling/dummy run: the indexer allocates its selection scratch
            # on demand, so touch its peak here for the memory profiler (as
            # sparse_attn_indexer does): the prefill logits budget or the
            # worst-case decode logits [decode tokens, compressed columns],
            # plus the top-k workspace.
            logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            cfg = get_current_vllm_config_or_none()
            if cfg is not None:
                spec = cfg.speculative_config
                num_spec = spec.num_speculative_tokens if spec is not None else 0
                decode_tokens = min(
                    cfg.scheduler_config.max_num_seqs * (num_spec + 1),
                    cfg.scheduler_config.max_num_batched_tokens,
                )
                block_size = cfg.cache_config.block_size
                columns = cdiv(cfg.model_config.max_model_len, block_size) * cdiv(
                    block_size, self.indexer.compress_ratio
                )
                logits_bytes = max(logits_bytes, decode_tokens * columns * 4)
            _ = torch.empty(
                logits_bytes + _TOPK_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=output.device,
            )
            output.zero_()
            return
        main_metadata = cast(Qwen4ExpQSAMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        indexer_metadata = cast(
            QSAIndexerMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        selected, main_outputs = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
            attn=self,
            qkv=qkv,
            slot_mapping=main_metadata.slot_mapping,
        )
        if selected.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        if main_outputs is None:
            assert key is not None and value is not None
            self.impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        else:
            query, output_gate = main_outputs
        assert query is not None and output_gate is not None
        self.impl.forward(
            self,
            query,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=indexer_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
            output_gate=output_gate,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.fuse_indexer_projection:
            qkv, projected_qk = qkv.split(
                [2 * self.q_size + 2 * self.kv_size, self.index_qk_size], dim=-1
            )
        else:
            projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        num_tokens = hidden_states.shape[0]
        if not self.use_fused_qsa_prepare:
            q, k, v, gate = self._project_qkv_gate(qkv, positions)
            assert gate is not None
            query = q.view(num_tokens, self.num_heads, self.head_dim)
            key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            # Norm/RoPE/gate and the K/V cache write happen inside _run_qsa.
            query = key = value = gate = None
        attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
        self._run_qsa(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
            gate,
            qkv,
        )
        flat_output = attn_output.view(num_tokens, -1)
        output, _ = self.o_proj(flat_output)
        return output


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSABackend",
    "Qwen4ExpQSAImpl",
]
