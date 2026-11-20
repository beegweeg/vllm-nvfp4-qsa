#!/usr/bin/env bash
# Подъём Qwen4Exp QSA с NVFP4 KV-кешем.
#
# Набор флагов снят с рабочей командной строки прода 20.11 и дополнен тем, без
# czego 0.34.0 не стартует (см. CAKE-NVFP4.md, раздел « Запуск»). Это пример, а
# не копия боевого launchera: пути и имя чекпоинта подставь свои, а сверяться
# надо с launcher'ем своей установки.
set -euo pipefail

MODEL=${MODEL:-<ckpt>/Qwen4Exp-QSA-NVFP4}
SERVED=${SERVED:-qwen4exp-qsa-nvfp4}
PORT=${PORT:-8600}
VENV=${VENV:-$HOME/venv/vllm-cake-0340}

# Пул KV задаётся явно: 2.60 GiB = 530,606 токенов (блок внимания 1472,
# num_gpu_blocks 414). Автосайзинг съедает память до прогрева ядер.
KV_BYTES=${KV_BYTES:-2791728742}

exec "$VENV/bin/vllm" serve "$MODEL" \
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
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","prefill_cudagraph_capture_token_sizes":[4,8,16,32],"max_prefill_cudagraph_capture_tokens":32,"decode_cudagraph_capture_request_sizes":[1,2,4,8],"max_decode_cudagraph_capture_requests":8}'

# Про PLE. В проде таблицы PLE уходят в RAM (`{"cpu_offload": true}`), но
# переключатель это или правка config.json — здесь не воспроизведён: сверься со
# стартером своей установки, прежде чем поднимать без него.
#
# Про --attention-backend FLASH_ATTN. Флаг декоративный: в flash_attn.py список
# supported_kv_cache_dtypes не содержит nvfp4, поэтому KV Serving отдаёт
# FlashInfer. Держим, чтобы не расходиться с продом; на скорость не влияет.
