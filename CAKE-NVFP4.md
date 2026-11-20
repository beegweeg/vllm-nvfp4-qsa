# CAKE / NVFP4 KV — форк vLLM

Инструкции по этой ветке: что изменено против апстрима, как собрать окружение
и какими флагами запускать. Замеры ёмкости KV, декода и префилла (NVFP4 против
FP8) — в [CAKE-NVFP4-BENCHMARKS.md](CAKE-NVFP4-BENCHMARKS.md). Железо и
эксплуатация — в вики `main-tools/wiki`, `projects/sticky-proxy/deploy.md`. Там же лежат настоящие
пути (venv, каталог чекпоинта, каталог launcher'а) и имя сервера: в этом репозитории они заменены
на плейсхолдеры, ветка публичная.

## Что это

NVFP4 KV-кеш для **Qwen4Exp QSA**. В апстриме его нет:
`Qwen4ExpQSABackend.supported_kv_cache_dtypes` не содержит `nvfp4`, а
`Qwen4ExpQSAImpl.__init__` бросает `NotImplementedError`. Без этой ветки
`--kv-cache-dtype nvfp4` для этой геометрии не запускается.

Назначение — держать KV в 4-бит при NVFP4-весах. Замерено: в пул 2.60 GiB на
8×RTX 5060 Ti влезает **530,606 токенов** против 343,671 у FP8 при том же
бюджете. Это два режима: либо 2 генерирующих сессии по 205k на GPU, либо больше
сессий, где генерирует только часть, а остальные лежат в хост-тире.

## Ветки

| ветка | база | назначение |
|---|---|---|
| `cake-nvfp4-0.34.0` | тег `v0.34.0` (`62d4bc3c7e`) | **прод**, `~/venv/vllm-cake-0340` |
| `cake-nvfp4-0.33` | `fa3b9d1377` (29.10.2026) | откат, `~/venv/vllm-cake-033` |
| `lab/experiments-0.33` | `cake-nvfp4-0.33` | Flint и Qwen3.5, см. соответствующий раздел |
| `main` | апстрим `main` | зеркало, для diff/rebase |

На GitHub из этого опубликована только `cake-nvfp4-0.34.0` — форк держится одна
продакшн-ветка. Остальные три живут здесь, в Gitea и на проде; `main` нужен для
`git diff` и `git log` против апстрима, на GitHub его нет.

Проверено 18.11 клонированием с Gitea: `cake-nvfp4-0.34.0` побайтово воспроизводит `vllm/` из `~/venv/vllm-cake-0340`, `lab/experiments-0.33` - из `~/venv/vllm-cake-033`; расхождений 0 в обоих. Не путать: рабочий 0.33-venv = `cake-nvfp4-0.33` **плюс** `lab/experiments-0.33` - без lab-ветки расходятся `models/flint/model.py`, `models/flint/ops/norm.py` и `models/qwen3_5/nvidia/model.py`. В venv сверх этого лежат сборочные артефакты (`.so`, `third_party/*`, `vllm_flash_attn/*`) и три `*.orig`, оставшихся от наложения патчей, - в репозитории их нет.

## Изменения против апстрима

Ветка `cake-nvfp4-0.34.0` = тег `v0.34.0` + 6 коммитов, 6 файлов:

| файл | что даёт |
|---|---|
| `models/qwen4_exp/nvidia/ops/qsa.py` | Triton-ядра деквантизации NVFP4 (E2M1 + block scales), разворот packed-страниц |
| `models/qwen4_exp/nvidia/qsa.py` | NVFP4-путь attention: spec геометрии, strided-вью над packed-кешем, cake-путь, H1-префилл |
| `platforms/interface.py` | NVFP4-геометрия страницы для бэкендов без `customize_spec` |
| `v1/attention/backends/flashinfer.py` | attention sink + NVFP4 KV на SM12x через LSE-эпилог |
| `v1/worker/gpu/spec_decode/speculator_config.py` | перенос `enable_cumem_allocator` в конфиг MTP-драфта |
| `model_executor/warmup/qwen4_exp_qsa_warmup.py` | `kv_cache_dtype` в прогрев QSA-ядер |

Ветка `cake-nvfp4-0.33` — те же 5 смысловых правок, но лежат в других файлах
(`config/speculative.py` вместо `speculator_config.py`) плюс guard прогрева
по `QSA_CAKE=1` вместо передачи `kv_cache_dtype`.

Коммиты независимы и разложены по смыслу — первые два (`[Bugfix][Platforms]`,
`[Bugfix][Spec Decode]`) самодостаточны и кандидаты в апстрим отдельно от
остального. Пути в таблице — относительно каталога `vllm/`.

## Сборка окружения

```bash
# внутренний Gitea — основной источник правды; публичное зеркало — форк на GitHub
git clone <gitea>/self-hosted/vllm-cake.git
# git clone https://github.com/beegweeg/vllm-nvfp4-qsa.git
cd vllm-cake
git checkout cake-nvfp4-0.34.0
python3 -m venv ~/venv/vllm-cake-0340
~/venv/vllm-cake-0340/bin/pip install -e . --no-build-isolation
```

Ставить из этого же checkout, а не из wheel'а: правки живут в исходниках.
После установки проверить:

```bash
SP=~/venv/vllm-cake-0340/lib/python3.12/site-packages/vllm
grep -c nvfp4 $SP/models/qwen4_exp/nvidia/qsa.py   # > 0
grep -n 'supported_kv_cache_dtypes' -A3 $SP/models/qwen4_exp/nvidia/qsa.py | head
```

## Запуск

Готовый набор флагов — [`launch.example.sh`](launch.example.sh) в корне ветки: пути и имя
чекоинта вынесены в переменные окружения (`MODEL`, `SERVED`, `PORT`, `VENV`,
`KV_BYTES`). Боевой launcher лежит не в репозитории, а на сервере фермы
(`~/flashnext-cake/start-8600.sh`), он же кнопка на дашборде — сверяться надо с
ним. Без части флагов 0.34.0 не стартует:

- `--no-enable-extensible-kv-cache`
- `--max-num-seqs 8` — иначе 37 draft-графов и OOM на захвате
- `--kv-cache-memory-bytes 2791728742` — явный размер пула (2.60 GiB =
  530,606 токенов), автосайзинг съедает память до прогрева ядер
- `--kv-offloading-size 48 --kv-offloading-backend native`
- `--enable-cumem-allocator` — без него MTP + оффлоад не сходятся (см. коммит
  `[Bugfix][Spec Decode]`)
- MRV2-ключи capture-размеров в `--compilation-config`

Старт идёт на PCIe Gen3 и сам переключается на Gen4 по готовности `/health`
(`pcie-staging.sh`, `PCIE_STAGING=1`). Оставлять сервис на Gen3 после старта
нельзя: декод падает примерно в 5 раз. Причина, почему прогрев PCIe-IPC не проходит на Gen4,
ещё не закрыта; разбор и замеры — в вики.

## Ветка `lab/experiments-0.33`

Не относится к NVFP4. Три правки, найденные в общем 0.33-venv от более ранней
работы и сохранённые здесь, чтобы не потерять: Flint с квантованным KV, block
dim во Flint `silu_and_mul` при TP, квантованный embedding у Qwen3.5.
В NVFP4-серию не входят и в апстрим не предлагаются.

## Плоские патчи

Те же правки в виде `.patch` для наложения на чистый wheel —
`~/flashnext-cake/nvfp4-kv-patches/` на сервере фермы.
Актуальный источник — эта ветка; патчи могут отставать.
