#!/usr/bin/env bash
# Подъём Qwen4Exp QSA с NVFP4 KV-кешем (8×RTX 5060 Ti, TP=8).
#
# Набор флагов и переменных окружения снят с боевой командной строки прода
# (20.11) и сверян с ней построчно. Это пример, а не копия боевого launchera:
# пути и имя чекпоинта подставь свои, а сверяться надо с launcher'ем своей
# установки.
#
# Почему `python -m vllm.entrypoints.openai.api_server`, а не `vllm serve`.
# В 0.34.0 субкоманда `vllm serve` не принимает --no-enable-extensible-kv-cache
# (парсер отвечает «unrecognized arguments»), а без этого флага 0.34.0 не
# стартует: появившийся в 0.34 extensible-кеш ломает явный размер KV-пула.
# Модуль api_server принимает весь набор — на нём и стоит прод. Проверено так:
#   vllm serve <model> ... --no-enable-extensible-kv-cache  → error
#   python -m vllm.entrypoints.openai.api_server ... → стартует
set -euo pipefail

MODEL=${MODEL:-<ckpt>/Qwen4Exp-QSA-NVFP4}
SERVED=${SERVED:-qwen4exp-qsa-nvfp4}
PORT=${PORT:-8600}
VENV=${VENV:-$HOME/venv/vllm-cake-0340}

# Пул KV задаётся явно: 2.60 GiB = 530,606 токенов (блок внимания 1472 токена,
# num_gpu_blocks 360). Автосайзинг съедает память до прогрева ядер, а при явном
# бюджёте ёмкость определяется только байтами пула и dtype.
KV_BYTES=${KV_BYTES:-2791728742}

# Окружение прода (в бою уходит в systemd-run --setenv):
#   QSA_CAKE=1                       — A/B-переключатель cake-пути в NVFP4-attention;
#                                      замеры этой ветки сняты на =1
#   VLLM_KV_CACHE_LAYOUT=BLHNC       — раскладка, под которой написаны ядра
#   VLLM_USE_BREAKABLE_CUDAGRAPH=1   — при =0 декод уезжает в eager (~8.8 t/s)
#   VLLM_USE_FLASHINFER_SAMPLER=0    — семплер FlashInfer на этой сборке медленнее
#   VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 — all-reduce через PCIe P2P, без NVLink
#   PYTORCH_CUDA_ALLOC_CONF          — expandable_segments; в паре с cumem-аллокатором
export QSA_CAKE=1
export VLLM_KV_CACHE_LAYOUT=${VLLM_KV_CACHE_LAYOUT:-BLHNC}
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=${VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED" \
  --port "$PORT" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --kv-cache-dtype nvfp4 \
  --max-model-len 210192 \
  --gpu-memory-utilization 0.97 \
  --gdn-prefill-backend triton \
  --moe-backend marlin \
  --max-num-seqs 8 \
  --no-enable-extensible-kv-cache \
  --enable-cumem-allocator \
  --kv-cache-memory-bytes "$KV_BYTES" \
  --kv-offloading-size 48 \
  --kv-offloading-backend native \
  --enable-prefix-caching \
  --spec-method mtp --spec-tokens 3 \
  --attention-backend FLASH_ATTN \
  --engram-config '{"cpu_offload": true}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","prefill_cudagraph_capture_token_sizes":[4,8,16,32],"max_prefill_cudagraph_capture_tokens":32,"decode_cudagraph_capture_request_sizes":[1,2,4,8],"max_decode_cudagraph_capture_requests":8}'

# Про --engram-config '{"cpu_offload": true}'. Таблицы PLE уходят в RAM. Раньше
# в этом файле он значился «не воспроизведённым» — воспроизведён, это ровно та
# строка, что стоит в бою.
#
# Про --attention-backend FLASH_ATTN. Флаг декоративный: в flash_attn.py список
# supported_kv_cache_dtypes не содержит nvfp4, поэтому KV Serving отдаёт
# FlashInfer. Держим, чтобы не расходиться с продом; на скорость не влияет.
#
# Про --max-num-seqs 8. Меньше — теряются сессии, больше — 37 draft-графов и OOM
# на захвате.
#
# Старт идёт на PCIe Gen3 и сам переключается на Gen4 по готовности /health
# (в бою — pcie-staging.sh). Оставлять сервис на Gen3 после старта Нельзя: декод
# падает примерно в 5 раз.
