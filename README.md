# llama-cpp-mera-quant-bench

Измерение **деградации метрик бенчмарка MERA в зависимости от квантования** GGUF-модели.
Все кванты одной модели из её репозитория HuggingFace прогоняются через одинаковый набор
**локально оцениваемых** задач MERA; метрики сравниваются с эталоном (BF16). Инференс —
**предсобранный llama.cpp** (v0.4.1 + PR #27537, echo+logprobs) как OpenAI-совместимый
`llama-server`, к которому обращается форк lm-evaluation-harness от MERA (бэкенд
`local-completions`).

## Как использовать

1. Google Colab → GPU-рантайм **T4 / L4 / A100** (сборка `native` требует AVX-512 — на
   CPU-рантайме EPYC не запустится).
2. Секреты Colab → добавить `HF_TOKEN`.
3. Открыть `mera_quant_bench.ipynb` → в **Блоке 3** задать модель (`PUBLISHER` /
   `BASE_MODEL_NAME`) и пресет → **Runtime → Run all**.
4. Сначала прогоните пресет `smoke` — он за ~15 мин на квант проверяет обе механики
   (echo+logprobs и генерацию).
5. Полный прогон — несколько сессий: **чекпоинт на Google Drive** возобновляет
   автоматически (готовые пары «квант × задача» пропускаются).

Архив llama.cpp берётся с Google Drive:
`/MyDrive/llama.cpp_v0.4.1-pr27537-version/native/gpu_all.tar.gz`, целостность
проверяется SHA-256 по `manifest.json` (сборка — проектом
[llama-cpp-colab-builder](https://github.com/GGUF-LLM-Test/llama-cpp-colab-builder)).

## Пресеты задач (Блок 3)

| Ключ | Название | Задачи | LIMIT |
|---|---|---|---|
| `smoke` | Быстрая проверка пайплайна | bps, simplear | 20 |
| `choice` | Выбор ответа — знания и этика (loglikelihood) | bps, rummlu, ruethics, ruhatespeech, ruhhh | 100 |
| `generation` | Генерация ответов (арифметика) | simplear | 100 |
| `all` | Все локально оцениваемые | все 6 | 100 |

## Почему только эти 6 задач

MERA держит ответы тестов закрытыми для большинства задач (оценка — через лидерборд).
Проверено по датасетам `MERA-evaluation/MERA` (поле `outputs` в test-сплите):

- **Включены** (ответы публичны, метрики считаются локально): `bps`, `rummlu`,
  `ruethics`, `ruhatespeech`, `ruhhh` (multiple_choice → loglikelihood, требует
  echo+logprobs) и `simplear` (генерация, exact_match).
- **Исключены**: 15 задач с закрытым тестом; `ruhumaneval` (pass@k исполняет
  сгенерированный код); `rudetox` (метрика J требует нейро-скорер). Пути ручного
  добавления: дополните `TASK_INFO` в Блоке 3 — для закрытых задач скоринг будет
  `bypass/999` без `--predict_only`-режима.

## Методика

- Скоринг **локальный**: `lm_eval` без `--predict_only`; few-shot — из YAML задач;
  `--seed 1234`; генеративные задачи — greedy (`do_sample=False`).
- **Эталон**: BF16 → Q8_0 → крупнейший влезающий в VRAM.
- **Деградация**: Δ в п.п. и % к эталону по каждой задаче + средний балл; вердикты:
  ≤1% «≈ без потерь», ≤3% «рекомендуется», ≤7% «приемлемо», >7% «заметная деградация»;
  Pareto-фронт «размер vs качество».
- **Честность**: задача исключается из средних, если у эталона метрика 0/нет данных
  (`unreliable`); версии transformers/lm-eval/llama.cpp фиксируются в чекпоинте и отчёте.
- **Гейт echo/logprobs**: перед тратой GPU-времени эталонная модель проверяет, что
  сервер отдаёт logprobs на echo-запрос — иначе прогон останавливается с диагностикой.

## Структура вывода на Google Drive

`/MyDrive/{PUBLISHER}_{MODEL}_MERA_Quant_Results/` — `checkpoint.json`,
`raw_logs/<квант>/<задача>/` (results/samples/task_meta), `mera_results.csv`,
`degradation.csv`, `final_ranking.csv`, `degradation_plots.png`,
`degradation_heatmap.png`, `final_analysis.png`, `mera_quant_report.md`,
`reparsed_from_raw.csv`.

## Совместимость с transformers

С 11.09.2026 сабмодуль MERA — `artemorloff/lm-evaluation-harness @ feat/text_benches`
(lm-eval 0.4.13.dev0, [коммит 8698451](https://github.com/MERA-Evaluation/MERA/commit/8698451dd7c281462c684190c9e38d2fae0f6d7a)):
промпты всех задач сверены побайтово со старым пином. Форк совместим с
предустановленным в Colab transformers 5.x из коробки (`AutoModelForVision2Seq`
предоставляется через `lm_eval.models.transformers_compat`), баг логирования
`api_models.py` (UnboundLocalError, маскировавший реальные сбои) исправлен upstream.
Пины, патчи и sys.path-обходы не нужны: `pip install -e ".[api]"` — и всё работает.

## Ограничения

- Сборка `native` (CUDA 75;80;89): только GPU-рантаймы на Xeon (T4/L4/A100).
- echo+logprobs (PR #27537) — ключевая зависимость loglikelihood-задач; гейт на ЭТАПЕ 0.
- `--use_cache` lm-eval не используется: ключ кэша не различает модели.
- Скорости (tok/s) не мерим — фиксируется только wall-time задач; глубокие метрики
  квантования (PPL/KLD) — отдельная серия Unsloth Quant Test.

## Происхождение

Механика — `RU_LLM_Benchmarks V6/V6.1` (MERA + llama-server + local-completions);
логика раннера — `Unsloth Qwen 3.5 0.8B Quant Test v5` (кванты, эталон, чекпоинт,
отчёты); сборка llama.cpp — `llama-cpp-colab-builder`. Ноутбук генерируется из
`tools/build_notebook.py` (единственный источник правды): правки — там, затем
`python tools/build_notebook.py`.

Ветка `GGUF-LLM-Test/llama.cpp@v0.4.1-pr27537-version` дополнительно несёт фикс
[454e945](https://github.com/GGUF-LLM-Test/llama.cpp/commit/454e945): ленивое
повышение `n_outputs_max` до `n_batch` при первом echo+logprobs-запросе (без
ключей и без стартовой VRAM). Без него параллельные echo-задачи в одном
decode-батче переполняли буфер выходов и убивали сервер
(`GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)`); с ним —
echo-промпты идут полными батчами и краш невозможен при любой конкуренции.
