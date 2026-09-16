# Дизайн: llama-cpp-mera-quant-bench

- **Дата:** 2026-09-16
- **Статус:** одобрен пользователем (brainstorming завершён)
- **Продукт:** Colab-notebook + README в репозитории `GGUF-LLM-Test/llama-cpp-mera-quant-bench`

## 1. Цель и контекст

Измерить **деградацию метрик бенчмарка MERA в зависимости от уровня квантования** одной
GGUF-модели. Все доступные кванты модели прогоняются через одинаковый набор локально
оцениваемых задач MERA; метрики сравниваются с эталоном (BF16). Инференс — через
**предсобранный llama.cpp** (v0.4.1 + PR #27537, echo+logprobs) из архива на Google Drive,
запущенный как OpenAI-совместимый `llama-server`, к которому обращается форк
lm-evaluation-harness от MERA через бэкенд `local-completions`.

Источники проекта:

| Источник | Что берём |
|---|---|
| `RU_LLM_Benchmarks V6/V6.1` (Sorted Drafts/Drafts) | механика: MERA-форк lm-eval + llama-server + `local-completions`; установка MERA; запуск/остановка сервера; токенизаторы; graceful-падение по задачам |
| `Unsloth Qwen 3.5 0.8B Quant Test v5` (Sorted Drafts) | логика раннера: автообнаружение квантов через HfApi, эталон BF16→Q8_0, атомарный чекпоинт, raw_logs на Диске с пересборкой без GPU, финальные таблицы/графики/вердикты/Pareto |
| `llama_cpp_colab_builder` | оформление репо (README-структура), работа с архивом сборки: SHA-256 по `manifest.json`, smoke-тесты бинарников |

Принципы: **один прогон — одна модель** (конфиг-ячейка); все кванты из её GGUF-репозитория
(автообнаружение, исключая `mmproj`); результаты — в отдельную папку Google Drive
`{PUBLISHER}_{MODEL}_MERA_Quant_Results`.

## 2. Зафиксированные решения

1. **Формат артефакта** — один Colab-notebook + README/docs (как `llama_cpp_colab_builder`), без Python-пакета.
2. **Сервер** — развёртывание из архива
   `Google Drive:/llama.cpp_v0.4.1-pr27537-version/native/gpu_all.tar.gz`
   (вариант `native`, CUDA 75;80;89, разделяемые `.so`). Без компиляции и без фолбэков на
   другие варианты сборки. Проверка целостности: SHA-256 против `manifest.json` того же
   релиза; smoke-тесты `--version` + `ldd`.
3. **Драйвер lm-eval** — CLI-подпроцесс **на каждую задачу каждого кванта** (подход 1 из
   трёх рассмотренных): проверено прогоном V6.1, падение одной задачи не рушит остальные,
   чекпоинт естественной гранулярности «квант × задача».
4. **Только локально оцениваемые задачи** (см. §3): `bps`, `rummlu`, `ruethics`,
   `ruhatespeech`, `ruhhh` (multiple_choice / loglikelihood) + `simplear` (generate_until).
   Исключены: 15 задач с закрытым тестом, `ruhumaneval` (pass@k требует исполнения кода),
   `rudetox` (метрика J требует нейро-скорер).
5. **Пресеты задач** — 4 группы с понятными русскими названиями, подсказками и описаниями
   (§4).
6. **Режим скоринга** — БЕЗ `--predict_only`: у отобранных задач ответы в публичном
   test-сплите, метрики считаются локально. few-shot — из YAML задач (не из CLI, по
   рекомендации форка). `--seed 1234` (официальный сид MERA), `--log_samples`,
   `--gen_kwargs do_sample=False` — только для генеративных задач (воспроизводимость).
7. **Эталон** — BF16 → Q8_0 → крупнейший влезающий в VRAM (логика v5).
8. **Публикация** — `github.com/GGUF-LLM-Test/llama-cpp-mera-quant-bench`, публичный, тот же
   аккаунт, что у `llama-cpp-colab-builder`.

## 3. Обоснование выбора задач (результат исследования)

Исследование: HF datasets-server (`MERA-evaluation/MERA`, test-сплит, поле `outputs` по
всем 23 конфигам), карточки датасетов, YAML задач (`benchmark_tasks/<task>/<task>.yaml`,
base-шаблоны `custom_loglikelihood_task.yaml` / `custom_generate_task.yaml`), документация
`MODEL_SCORING*.md`, эмпирика V6.1/Untitled4.

| Задача | output_type | outputs в test | Метрики | Вердикт |
|---|---|---|---|---|
| bps | multiple_choice | **есть** | acc | ✅ включена |
| rummlu | multiple_choice | **есть** | acc + 56 доменных | ✅ включена (открытый диагностический) |
| ruethics | multiple_choice | **есть** | 15 × mcc_* | ✅ включена |
| ruhatespeech | multiple_choice | **есть** | acc + 6 групповых | ✅ включена |
| ruhhh | multiple_choice | **есть** | acc + 3 групповых | ✅ включена |
| simplear | generate_until | **есть** | exact_match | ✅ включена (2-shot) |
| ruhumaneval | generate_until | есть | pass@1/5/10 | ❌ исключена: скоринг исполняет сгенерированный код |
| rudetox | generate_until | есть | j, sta, sim, fl | ❌ исключена: нейро-скорер, тяжёлый |
| mathlogicqa, multiq, parus, rcb, rumodar, rumultiar, ruopenbookqa, rutie, ruworldtree, rwsd, chegeka, lcs, rucodeeval, mamuramu, use | — | **пусто** | — | ❌ исключены: закрытый тест, оценка только на лидерборде |

Примечания:
- Пустота `outputs` подтверждена на нескольких смещениях (0/100/500/1000/3000).
- Аномалия V6.1 (ненулевые `em,scoring` у rumodar/rumultiar) объясняется служебным
  `public_test`-сплитом rumodar (с ответами) и `train` у rumultiar; официальные тесты этих
  задач закрыты — в проекте не используются.
- `modules/scoring/README.md`: «there are no correct answers to all tasks in the public
  data» — выборка только открытых задач корректна по методологии MERA.
- Для исследования деградации относительное сравнение квантов на открытых задачах
  валидно: одинаковые промпты, эталон и механика для всех квантов.

## 4. Пресеты задач

```python
PRESETS = {
  "smoke":      "Быстрая проверка пайплайна",
  "choice":     "Выбор ответа — знания и этика (loglikelihood)",
  "generation": "Генерация ответов (арифметика)",
  "all":        "Все локально оцениваемые задачи",
}
```

| Ключ | Название | Задачи | LIMIT по умолчанию | Подсказка |
|---|---|---|---|---|
| `smoke` | Быстрая проверка пайплайна | bps, simplear | 20 | ~15 мин/квант; проверяет ОБЕ механики: echo+logprobs (bps) и генерацию (simplear) |
| `choice` | Выбор ответа — знания и этика | bps, rummlu, ruethics, ruhatespeech, ruhhh | 100 | чистый loglikelihood-путь; главный тест echo+logprobs (PR #27537) |
| `generation` | Генерация ответов | simplear | 100 | свободная генерация, exact_match |
| `all` | Все локально оцениваемые | все 6 | 100 | полный прогон; несколько сессий, чекпоинт возобновляет |

При выполнении конфиг-ячейки печатается таблица пресетов с описаниями; выбранному —
список задач, число примеров, оценка времени. LIMIT задаваемый, переопределяет дефолт
пресета.

## 5. Архитектура notebook (блоки 0–10)

Скелет v5 (нумерованные `# @title`-блоки, Run all), механика V6.1.

- **Блок 0. Диагностика окружения** — GPU (имя, CC, VRAM), CPU, RAM, диск. CPU-часть
  сборки `native` требует AVX-512: на EPYC-рантайме без GPU — предупреждение и остановка.
- **Блок 1. HF-токен** — `userdata.get("HF_TOKEN")`, проверка `whoami`, экспорт в `os.environ`.
- **Блок 2. Google Drive + место** — mount, проверка свободного места (архив ~124 МБ +
  модели до ~20 ГБ + сырые логи).
- **Блок 3. КОНФИГУРАЦИЯ (менять здесь)**:
  - `PUBLISHER`, `BASE_MODEL_NAME` → `REPO_ID` (дефолт `unsloth/Qwen3.5-9B-GGUF`);
  - `PRESET` + `LIMIT`; `SEED = 1234`; `SERVER_PORT = 8000`;
  - `MAX_VRAM_GB` — фильтр квантов (размер + запас ~4 ГБ на контекст/KV);
  - `FORCE_RERUN = False`;
  - `LLAMA_CPP_ARCHIVE` — путь к архиву на Диске; `LLAMA_CPP_MANIFEST` — к `manifest.json`;
  - `SAVE_DIR = /content/drive/MyDrive/{PUBLISHER}_{MODEL}_MERA_Quant_Results`,
    `RAW_LOGS_DIR = SAVE_DIR/raw_logs`, `CHECKPOINT_PATH = SAVE_DIR/checkpoint.json`;
  - печать выбранного пресета: задачи, few-shot из YAML, примеры, оценка времени.
- **Блок 4. Установка MERA** — перенос Блока 1.1 из V6.1: pin `transformers>=4.44,<5.00`,
  `git clone --recurse-submodules https://github.com/MERA-Evaluation/MERA.git`,
  `pip install -e ./MERA/lm-evaluation-harness`, патч логирования `api_models.py`,
  минимальные зависимости (`openai`, `datasets`, `pandas`, …). Проверка `lm_eval --help`.
- **Блок 5. Развёртывание llama.cpp из архива**:
  1. SHA-256 архива против `manifest.json` (несовпадение — остановка с диагностикой);
  2. `tar -xzf` → `/content/llama_cpp_bin/`, `chmod +x llama-server`;
  3. smoke: `./llama-server --version`, `ldd` (критические `.so` из архива);
  4. печать версии/коммита сборки.
- **Блок 6. Вспомогательные функции**:
  - `get_available_quants(repo_id)` — HfApi, `.gguf` без `mmproj`, размеры, сортировка по размеру;
  - `select_reference_model(quants, max_vram)` — BF16 → Q8_0 → крупнейший влезающий;
  - `short_name(filename)`;
  - чекпоинт: `load_checkpoint` / `save_checkpoint` (атомарно через `.tmp` + `os.replace`),
    в чекпоинте `config_meta = {repo_id, preset, limit, seed, tasks}`; при несовпадении с
    текущим конфигом — предупреждение и требование `FORCE_RERUN=True` (или удалить папку);
  - `get_tokenizer_path(repo_id)` — кэш на Диске → локально → скачивание
    (`snapshot_download` только файлы токенизатора), патч `config.json` int→float
    (`routed_scaling_factor` и др., из V6.1 — актуально для MoE-моделей);
  - `kill_existing_server(port)`, `start_llama_server(model_path, port, alias)`
    (из V6.1: `start_new_session=True`, health-поллинг `/health`, логи в файл,
    `LD_LIBRARY_PATH` → `/content/llama_cpp_bin`, `-ngl -1`, `-c 8192`, `--seed`),
    `stop_llama_server(...)` (SIGTERM→SIGKILL группе процессов);
  - `smoke_test_echo_logprobs(port)` — POST `/v1/completions`
    `{"prompt": "1+1=", "max_tokens": 1, "echo": true, "logprobs": 5}`; валидация: ответ
    200, есть `choices[0].logprobs.token_logprobs` длиной > 1; **критический гейт**;
  - `run_task(quant, task, model_path, tokenizer_path)` — CLI-подпроцесс:
    ```
    lm_eval --model local-completions
      --model_args model=<alias>,base_url=http://127.0.0.1:{port}/v1/completions,
                   num_concurrent=8,tokenizer_backend=huggingface,
                   tokenizer=<tokenizer_path>,tokenized_requests=False,timeout=<...>
      --tasks <task> --include_path ./MERA/benchmark_tasks
      --output_path ./results/<quant>/<task> --log_samples
      --seed {SEED} --batch_size 1 [--limit N] [--gen_kwargs do_sample=False] --verbosity ERROR
    ```
    (без `--predict_only`, без `--num_fewshot` — из YAML; `--gen_kwargs` только для
    generate_until-задач; env: `PYTHONPATH=./MERA`, `TOKENIZERS_PARALLELISM=false`,
    `HF_DATASETS_IN_MEMORY_MAX_SIZE`);
    копирование `results_*.json` + `samples_*.jsonl` в `RAW_LOGS_DIR/<quant>/<task>/`;
    парсинг метрик: первичная метрика задачи (§6) + все сырые;
    замер wall-time; таймаут подзадачи 2 ч;
  - `cleanup(model_path)` — из v5: удаление модели, HF-кэша, `/tmp/*`; НЕ трогает
    raw_logs и чекпоинт;
  - самотест: все требуемые функции/переменные определены (как Блок 5.9 в v5).
- **Блок 7. Основной цикл** (резюмируемый):
  - **ЭТАП 0 — эталон**: выбрать эталон, скачать, запустить сервер, **гейт
    smoke_test_echo_logprobs** (провал — остановка с внятной диагностикой «сборка не
    поддерживает echo+logprobs, loglikelihood-задачи не пройдут»); прогнать все задачи
    пресета; чекпоинт после каждой;
  - **ЭТАП 1 — кванты** по возрастанию размера: фильтр `MAX_VRAM_GB`; skip по чекпоинту;
    сервер → все задачи пресета → чекпоинт после каждой → `cleanup`;
  - обработка `KeyboardInterrupt` через `try/finally` — сервер всегда останавливается
    (механика V6.1), выполненные задачи сохранены.
- **Блок 8. Агрегация и визуализация**:
  - восстановление `df`: CSV с Диска + чекпоинт (dedupe по Quant, `keep=last`);
  - таблица: квант × задача (первичные метрики) + Δ к эталону (п.п. и %) + средние;
  - **правило ненадёжности**: задача помечается `unreliable`, если у эталона метрика 0.0
    или NaN — исключается из средних и деградации (честность анализа);
  - графики (matplotlib): деградация каждой метрики vs `Size_GB` (аннотации квантов),
    общий средний балл vs `Size_GB`, heatmap «квант × задача» (Δ п.п.);
  - артефакты: `mera_results.csv`, `degradation.csv`, PNG, MD-отчёт (легенда метрик,
    hardware-тег, секция по каждому кванту).
- **Блок 9. Финальный анализ** — вердикты по порогам относительной деградации среднего
  балла: ≤1% «≈ без потерь», ≤3% «рекомендуется», ≤7% «приемлемо», >7% «заметная
  деградация»; Pareto-фронт (Size_GB vs средний балл); `final_ranking.csv`,
  `final_analysis.png`.
- **Блок 10. Пересборка из raw_logs без GPU** — повторный парсинг
  `RAW_LOGS_DIR/<quant>/<task>/results_*.json` в df (аналог Блока 9 v5).

## 6. Первичные метрики и деградация

| Задача | Первичная метрика | Источник в results_*.json |
|---|---|---|
| bps | acc | `results["bps"]["acc,none"]` |
| rummlu | acc | `results["rummlu"]["acc,none"]` |
| ruethics | mcc_mean | среднее по 15 `mcc_*,none` |
| ruhatespeech | acc | `results["ruhatespeech"]["acc,none"]` |
| ruhhh | acc | `results["ruhhh"]["acc,none"]` |
| simplear | exact_match | `results["simplear"]["exact_match,none"]` |

Деградация кванта Q относительно эталона B: `Δ_task = m(Q,task) − m(B,task)` (п.п.) и
`δ_task = (m(Q,task) − m(B,task)) / m(B,task)` (%; при m(B)=0 задача уже помечена
unreliable). Средний балл кванта = среднее первичных метрик надёжных задач пресета.

## 7. Структура вывода на Google Drive

```
{PUBLISHER}_{MODEL}_MERA_Quant_Results/
├── checkpoint.json               # атомарный, config_meta внутри
├── raw_logs/
│   └── <quant>/                  # BF16, Q8_0, Q6_K, ...
│       └── <task>/
│           ├── results_*.json    # копия вывода lm_eval
│           ├── samples_*.jsonl
│           └── server_stderr.log (при падении задачи)
├── mera_results.csv              # квант × задача × метрики × wall_time
├── degradation.csv               # Δ п.п., δ %, средний балл, вердикт
├── final_ranking.csv
├── degradation_plots.png
├── final_analysis.png
└── mera_quant_report.md
```

## 8. Обработка ошибок

| Ситуация | Поведение |
|---|---|
| SHA-256 архива ≠ manifest | остановка до любых действий |
| Нет GPU / CPU без AVX-512 | остановка (native-сборка) |
| smoke echo/logprobs провален | остановка с диагностикой (loglikelihood невозможен) |
| Ошибка одной задачи | `continue` к следующей (лог в raw_logs, не в чекпоинт — повторится при следующем запуске) |
| Ошибка кванта (скачивание/сервер) | квант помечен Error, цикл продолжается |
| Квант не влезает в VRAM | skip с пометкой |
| KeyboardInterrupt / Stop Colab | `finally`: остановка сервера; готовое сохранено в чекпоинт |
| Чекпоинт с другим config_meta | предупреждение, требование FORCE_RERUN или удаления папки |
| Повторный запуск | skip готовых «квант × задача», докачка недостающих |

## 9. Верификация (без Colab)

1. `python -m json.tool` / `nbformat.validate` — валидность ipynb.
2. `ast.parse` каждой code-ячейки — синтаксис.
3. Согласованность имён между блоками (самотест-ячейка в notebook дублирует список).
4. README соответствует фактическому поведению (пресеты, пути, артефакты).
5. Публикация: `gh repo create GGUF-LLM-Test/llama-cpp-mera-quant-bench --public`,
   push, проверка URL.

Реальный прогон — в Colab пользователем (T4/L4/A100, секрет `HF_TOKEN`, Run all);
сначала пресет `smoke` на одной модели.

## 10. Ограничения и риски

- **Сборка `native`**: CPU-часть оптимизирована под машину сборки (AVX-512) → только
  GPU-рантаймы Colab на Xeon (T4/L4/A100); на no-GPU EPYC упадёт.
- **echo+logprobs (PR #27537) эмпирически не доказан** (отладка Untitled10 не была
  пройдена) → гейт ЭТАПА 0; при провале проект честно сообщает об этом.
- **Скоринг `,scoring`-метрик не используется** — работаем со стандартными метриками
  форка (`acc,none`, `exact_match,none`, `mcc_*`) на открытых задачах.
- Полный `all`-прогон ~20+ квантов — несколько Colab-сессий; чекпоинт возобновляет
  (disk space guard перед каждым квантом).
- Форк lm-eval — база LM-Harness v0.4.8; pin `transformers<5.00` обязателен
  (в 5.x нет `AutoModelForVision2Seq`).
- `--use_cache` lm-eval НЕ используется: ключ кэша не различает модели (риск подмены
  ответов между квантами).

## 11. Вне объёма (осознанно)

- Упаковка `submission.zip` и лидерборд MERA.
- 15 закрытых задач, `rudetox`, `ruhumaneval` (пути ручного добавления — в README).
- Замер скорости (tok/s) — фиксируется только wall-time задач; глубокие метрики
  квантования (PPL/KLD) — отдельная серия Unsloth.
- SSH/CloudPub-шапка из V6.
- Компиляция llama.cpp и альтернативные варианты сборки (`universal`, `native-static`).
