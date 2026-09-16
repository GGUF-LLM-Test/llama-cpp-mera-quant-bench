# MERA Quant Bench — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Собрать Colab-notebook `mera_quant_bench.ipynb` + README, измеряющие деградацию метрик MERA по квантам одной GGUF-модели через предсобранный llama.cpp (v0.4.1+PR#27537) из архива Google Drive, и опубликовать репозиторий на GitHub.

**Architecture:** Notebook генерируется детерминированно из `tools/build_notebook.py` (единственный источник правды: список ячеек в питоне → JSON ipynb → автопроверка ast/структуры). Пайплайн: конфиг с пресетами задач → установка MERA-форка lm-eval (pin transformers<5.00) → распаковка llama-server из архива с проверкой SHA-256 по manifest.json → цикл «квант × задача» через `lm_eval --model local-completions` с локальным скорингом → чекпоинт на Drive → таблицы деградации/графики/вердикты. Механика портирована из `RU_LLM_Benchmarks_V6.1.ipynb` и `Unsloth Qwen 3.5 0.8B Quant Test v5.ipynb`.

**Tech Stack:** Python 3.13 (Colab), lm-evaluation-harness (форк MERA, база v0.4.8), llama.cpp v0.4.1+PR#27537 (native/gpu_all, CUDA 75;80;89), HfApi, pandas, matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-16-mera-quant-bench-design.md` (в этом же репо; исполнитель читает спеку и план вместе).

## Global Constraints

- Рабочая директория: `C:\Users\Oleg\Documents\OpenCode Projects\TestLLM\llama-cpp-mera-quant-bench` (git-репо, ветка `main`).
- Notebook: имя файла `mera_quant_bench.ipynb`; генерируется ТОЛЬКО через `python tools/build_notebook.py`; руками ipynb не правим.
- Документация и комментарии в notebook/README — на русском.
- Pin: `pip install "transformers>=4.44,<5.00"` (спека §10.1). Версии `transformers`/`lm_eval` фиксируются в чекпоинте.
- Задачи — ТОЛЬКО локально оцениваемые: `bps, rummlu, ruethics, ruhatespeech, ruhhh, simplear` (спека §3). Пресеты: `smoke/choice/generation/all`.
- `lm_eval`: БЕЗ `--predict_only`; БЕЗ `--num_fewshot` (берётся из YAML задач); `--seed 1234`; `--log_samples`; `--gen_kwargs do_sample=False` только для `type=="gen"`.
- Сервер: архив `LLAMA_CPP_ARCHIVE` (native/gpu_all), SHA-256 против `manifest.json` (ключи русские: `артефакты` → `native/gpu_all` → `sha256_tar`, `коммит`, `ветка`, `cuda_архитектуры`), tar-распаковка, smoke `--version` + `ldd`.
- Гейт `smoke_test_echo_logprobs` — критический на эталоне (спека §8).
- Чекпоинт: `config_meta = {repo_id, preset, limit, seed, tasks, transformers_version, lm_eval_version, llama_cpp_sha256}`; несовпадение `repo_id/preset/limit/seed/tasks` → RuntimeError; дрейф версий → предупреждение.
- `--use_cache` lm-eval НЕ использовать (спека §10.2).
- Коммиты — короткие, английский, императив (стиль: `llama-cpp-colab-builder`).
- Публикация: `gh repo create GGUF-LLM-Test/llama-cpp-mera-quant-bench --public --source . --push` (gh авторизован как asbelin, scope repo).
- Каждый шаг «сборка+проверка» — команда `python tools/build_notebook.py`; ожидаемый вывод: `OK: N ячеек (M code), все code-ячейки парсятся, порядок блоков верный`.

## Источники портирования (абсолютные пути, только для чтения)

- V6.1: `C:\Users\Oleg\Documents\OpenCode Projects\TestLLM\Sorted Drafts\RU_LLM_Benchmarks\RU_LLM_Benchmarks_V6.1.ipynb` (механика MERA/сервера/токенизатора; конспект ячеек: `C:\Users\Oleg\AppData\Local\Temp\opencode\v61_all_cells.txt`)
- v5: `C:\Users\Oleg\Documents\OpenCode Projects\TestLLM\Sorted Drafts\Unsloth_Quant\Unsloth Qwen 3.5 0.8B Quant Test v5.ipynb` (логика раннера; конспект: `C:\Users\Oleg\AppData\Local\Temp\opencode\v5_sources.txt`)
- manifest.json (эталон SHA-256): `C:\Users\Oleg\Documents\OpenCode Projects\TestLLM\google-drive\llama.cpp_v0.4.1-pr27537-version\manifest.json`

В плане весь финальный код уже включён — портирование не требуется, код ниже переносится дословно.

---

### Task 1: Каркас — build-инструмент, .gitignore, титульная markdown-ячейка

**Files:**
- Create: `tools/build_notebook.py`
- Create: `.gitignore`
- Generated: `mera_quant_bench.ipynb` (коммитится как артефакт)

**Interfaces:**
- Produces: `tools/build_notebook.py` со структурой: `CELLS = [md_cell(src), code_cell(src), ...]`; функции `md_cell/code_cell/build/verify`; `EXPECTED_BLOCKS` — упорядоченный список маркеров первых строк code-ячейки (`БЛОК 0`, `БЛОК 1`, … `БЛОК 10`). Каждый следующий таск добавляет ячейки в `CELLS` и (при новом блоке) маркер в `EXPECTED_BLOCKS`.

- [ ] **Step 1: Создать `.gitignore`**

```gitignore
__pycache__/
.ipynb_checkpoints/
*.pyc
```

- [ ] **Step 2: Создать `tools/build_notebook.py` (каркас + титульная ячейка)**

```python
"""Сборка и проверка mera_quant_bench.ipynb.

Единственный источник правды для notebook. Правки делаются здесь,
затем: python tools/build_notebook.py
"""
import ast
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "mera_quant_bench.ipynb"

MD_INTRO = """\
# MERA Quant Bench — деградация метрик MERA по квантам GGUF

Оценивает **все кванты одной модели** (репозиторий `{PUBLISHER}/{MODEL}-GGUF`) на **локально
проверяемых** задачах MERA через предсобранный `llama-server` (llama.cpp v0.4.1 + PR #27537,
echo+logprobs) и форк lm-evaluation-harness от MERA (бэкенд `local-completions`).

**Как использовать:** GPU-рантайм (T4/L4/A100) → секрет `HF_TOKEN` → Runtime → Run all.
Сначала прогоните пресет `smoke` (Блок 3). Полный прогон — несколько сессий: чекпоинт на
Google Drive возобновляет автоматом.

Схема блоков: 0 диагностика · 1 токен · 2 Drive · 3 конфигурация · 4 MERA · 5 llama.cpp ·
6 функции · 7 цикл · 8 агрегация · 9 финал · 10 пересборка из логов.
"""

EXPECTED_BLOCKS = []  # заполняется по мере добавления ячеек


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "id": "%08x" % random.getrandbits(32),
            "metadata": {}, "source": src.splitlines(keepends=True)}


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "id": "%08x" % random.getrandbits(32),
            "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


CELLS = [
    md_cell(MD_INTRO),
]


def build() -> None:
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": CELLS,
    }
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def verify() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, c in enumerate(code_cells):
        src = "".join(c["source"])
        ast.parse(src)  # SyntaxError → падение с номером ячейки
    titles = ["".join(c["source"]).splitlines()[1] for c in code_cells]
    for marker in EXPECTED_BLOCKS:
        assert any(marker in t for t in titles), f"Не найден блок: {marker}"
    pos = [-1]
    for marker in EXPECTED_BLOCKS:
        idx = next(i for i, t in enumerate(titles) if marker in t)
        assert idx > pos[-1], f"Нарушен порядок блоков: {marker}"
        pos.append(idx)
    print(f"OK: {len(nb['cells'])} ячеек ({len(code_cells)} code), "
          f"все code-ячейки парсятся, порядок блоков верный")


if __name__ == "__main__":
    random.seed(20260916)
    build()
    verify()
```

- [ ] **Step 3: Собрать и проверить**

Run: `python tools/build_notebook.py`
Expected: `OK: 1 ячеек (0 code), все code-ячейки парсятся, порядок блоков верный`

- [ ] **Step 4: Commit**

```bash
git add .gitignore tools/build_notebook.py mera_quant_bench.ipynb
git commit -m "Add notebook build tool with title cell"
```

---

### Task 2: Блоки 0–2 — диагностика, HF-токен, Google Drive

**Files:**
- Modify: `tools/build_notebook.py` (добавить 3 ячейки в `CELLS`, 3 маркера в `EXPECTED_BLOCKS`)

**Interfaces:**
- Produces (глобальные переменные notebook): `torch` (import), `shutil`, `datetime`; `os.environ["HF_TOKEN"]`; `DRIVE_ROOT = "/content/drive"`.

- [ ] **Step 1: Добавить в `CELLS` (после md-ячейки) три code-ячейки и маркеры**

`EXPECTED_BLOCKS = ["БЛОК 0", "БЛОК 1", "БЛОК 2"]`

```python
BLOCK_00 = """\
# @title БЛОК 0: ДИАГНОСТИКА ОКРУЖЕНИЯ
import os, shutil, subprocess, time, json, gc
from pathlib import Path
from datetime import datetime
import torch

print(f"🖥️ Диагностика: {datetime.now():%Y-%m-%d %H:%M:%S}")
if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU не найден. Среда выполнения → Сменить среду выполнения → T4/L4/A100.")

props = torch.cuda.get_device_properties(0)
print(f"  GPU: {props.name} ({props.total_memory / 1e9:.1f} ГБ VRAM, CC {props.major}.{props.minor})")

isa = subprocess.run("grep -o -m1 avx512f /proc/cpuinfo", shell=True,
                     capture_output=True, text=True).stdout.strip()
print(f"  CPU avx512f: {'✅' if isa else '❌ нет'} (сборка native требует AVX-512)")
if not isa:
    raise RuntimeError("❌ CPU без AVX-512: native-сборка llama.cpp не запустится. Нужен GPU-рантайм на Xeon (T4/L4/A100).")

ram = {}
with open("/proc/meminfo") as f:
    for line in f:
        if ":" in line:
            k, v = line.split(":", 1)
            ram[k] = int(v.split()[0])
print(f"  RAM: {ram.get('MemTotal', 0) / 1e6:.1f} ГБ (доступно {ram.get('MemAvailable', 0) / 1e6:.1f} ГБ)")
total, used, free = shutil.disk_usage("/")
print(f"  Диск: свободно {free / 1e9:.1f} ГБ")
print("✅ Блок 0 завершён.")
"""

BLOCK_01 = """\
# @title БЛОК 1: ТОКЕН HUGGING FACE
from google.colab import userdata
from huggingface_hub import HfApi

hf_token = userdata.get("HF_TOKEN")
if not hf_token or not str(hf_token).strip():
    print("⚠️ Токен HF_TOKEN не найден в секретах Colab. Модели из приватных репо будут недоступны.")
    os.environ["HF_TOKEN"] = ""
else:
    try:
        user = HfApi().whoami(token=hf_token)
        print(f"✅ Токен проверен. Добро пожаловать, {user.get('name', 'пользователь')}!")
        os.environ["HF_TOKEN"] = str(hf_token).strip()
    except Exception as e:
        print(f"❌ Токен недействителен: {e}")
        os.environ["HF_TOKEN"] = ""
"""

BLOCK_02 = """\
# @title БЛОК 2: GOOGLE DRIVE И СВОБОДНОЕ МЕСТО
from google.colab import drive

print("💾 Монтирование Google Drive...")
drive.mount("/content/drive", force_remount=False)
DRIVE_ROOT = "/content/drive"

total, used, free = shutil.disk_usage(DRIVE_ROOT)
print(f"💾 Свободно на Google Диске: {free / 1e9:.2f} ГБ")
if free / 1e9 < 2.0:
    print("⚠️ ВНИМАНИЕ: на Диске меньше 2 ГБ — чекпоинт и сырые логи могут не сохраниться!")
"""
```

- [ ] **Step 2: Собрать и проверить**

Run: `python tools/build_notebook.py`
Expected: `OK: 4 ячеек (3 code), ...`

- [ ] **Step 3: Commit**

```bash
git add tools/build_notebook.py mera_quant_bench.ipynb
git commit -m "Add blocks 0-2: env diagnostics, HF token, Drive mount"
```

---

### Task 3: Блок 3 — конфигурация и пресеты задач

**Files:**
- Modify: `tools/build_notebook.py` (+1 ячейка, +1 маркер)

**Interfaces:**
- Produces (глобальные, используются всеми следующими блоками):
  - `PUBLISHER: str`, `BASE_MODEL_NAME: str`, `REPO_ID: str`
  - `PRESET: str`, `PRESETS: dict`, `TASKS: list[str]`, `LIMIT: int`, `TASK_INFO: dict`
  - `SEED=1234`, `SERVER_PORT=8000`, `MAX_VRAM_GB: float`, `FORCE_RERUN: bool`
  - `LLAMA_CPP_ARCHIVE: str`, `LLAMA_CPP_MANIFEST: str`
  - `SAVE_DIR: Path`, `RAW_LOGS_DIR: Path`, `CHECKPOINT_PATH: str`
  - `MERA_REPO_DIR: Path`, `LM_EVAL_PATH: Path`, `MERA_TASKS_PATH: Path`
  - `LOCAL_MODEL_DIR, LOCAL_BIN_DIR, LOG_DIR, LOCAL_RESULTS_DIR: Path` (созданы mkdir)
  - `get_time() -> str`

- [ ] **Step 1: Добавить ячейку `BLOCK_03` и маркер `"БЛОК 3"`**

```python
BLOCK_03 = """\
# @title БЛОК 3: КОНФИГУРАЦИЯ (менять параметры здесь)
import torch

# ---------- Модель (один прогон = одна модель) ----------
PUBLISHER = "unsloth"
BASE_MODEL_NAME = "Qwen3.5-9B"
REPO_ID = f"{PUBLISHER}/{BASE_MODEL_NAME}-GGUF"

# ---------- Пресет задач ----------
PRESET = "all"   # "smoke" | "choice" | "generation" | "all"

TASK_INFO = {
    "bps":          {"type": "mc",  "primary_key": "acc,none",         "fewshot": 1, "desc": "Сбалансированные скобочные последовательности (код, математика)"},
    "rummlu":       {"type": "mc",  "primary_key": "acc,none",         "fewshot": 1, "desc": "Русский MMLU — знания по 57 доменам (открытый диагностический сет)"},
    "ruethics":     {"type": "mc",  "primary_key": "mcc_mean",         "fewshot": 0, "desc": "Этика: 5 измерений, метрика MCC"},
    "ruhatespeech": {"type": "mc",  "primary_key": "acc,none",         "fewshot": 1, "desc": "Разметка токсичных комментариев"},
    "ruhhh":        {"type": "mc",  "primary_key": "acc,none",         "fewshot": 0, "desc": "Helpful / Honest / Harmless"},
    "simplear":     {"type": "gen", "primary_key": "exact_match,none", "fewshot": 2, "desc": "Арифметика: точный ответ (exact match)"},
}

PRESETS = {
    "smoke": {"title": "Быстрая проверка пайплайна",
               "tasks": ["bps", "simplear"], "limit": 20,
               "hint": "~15 мин/квант. Проверяет ОБЕ механики: echo+logprobs (bps) и генерацию (simplear)."},
    "choice": {"title": "Выбор ответа — знания и этика (loglikelihood)",
               "tasks": ["bps", "rummlu", "ruethics", "ruhatespeech", "ruhhh"], "limit": 100,
               "hint": "Чистый loglikelihood-путь: главный тест echo+logprobs (PR #27537)."},
    "generation": {"title": "Генерация ответов (арифметика)",
               "tasks": ["simplear"], "limit": 100,
               "hint": "Свободная генерация ответов, метрика exact_match."},
    "all": {"title": "Все локально оцениваемые задачи",
               "tasks": list(TASK_INFO), "limit": 100,
               "hint": "Полный прогон: несколько сессий, чекпоинт возобновит автоматом."},
}

assert PRESET in PRESETS, f"Неизвестный пресет {PRESET}: {list(PRESETS)}"
TASKS = PRESETS[PRESET]["tasks"]
LIMIT = PRESETS[PRESET]["limit"]   # None → дефолт пресета; можно задать своё число здесь

# ---------- Параметры прогона ----------
SEED = 1234            # официальный сид MERA
SERVER_PORT = 8000
MAX_VRAM_GB = None     # None → авто (VRAM GPU − 2 ГБ); можно задать вручную
FORCE_RERUN = False    # True = игнорировать чекпоинт и пересчитать всё

if MAX_VRAM_GB is None:
    MAX_VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / 1e9 - 2.0, 1)

# ---------- Сборка llama.cpp (архив на Drive) ----------
_DRIVE = Path(DRIVE_ROOT) / "MyDrive"
LLAMA_CPP_ARCHIVE = str(_DRIVE / "llama.cpp_v0.4.1-pr27537-version" / "native" / "gpu_all.tar.gz")
LLAMA_CPP_MANIFEST = str(_DRIVE / "llama.cpp_v0.4.1-pr27537-version" / "manifest.json")

# ---------- Пути вывода ----------
SAVE_DIR = _DRIVE / f"{PUBLISHER}_{BASE_MODEL_NAME}_MERA_Quant_Results"
RAW_LOGS_DIR = SAVE_DIR / "raw_logs"
CHECKPOINT_PATH = str(SAVE_DIR / "checkpoint.json")

MERA_REPO_DIR = Path("./MERA").resolve()
LM_EVAL_PATH = MERA_REPO_DIR / "lm-evaluation-harness"
MERA_TASKS_PATH = MERA_REPO_DIR / "benchmark_tasks"

LOCAL_MODEL_DIR = Path("./models").resolve()
LOCAL_BIN_DIR = Path("./llama_cpp_bin").resolve()
LOG_DIR = Path("./logs").resolve()
LOCAL_RESULTS_DIR = Path("./results").resolve()

for d in (SAVE_DIR, RAW_LOGS_DIR, MERA_REPO_DIR.parent, LOCAL_MODEL_DIR, LOCAL_BIN_DIR, LOG_DIR, LOCAL_RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def get_time():
    return datetime.now().strftime("%H:%M:%S")

print("=" * 70)
print(f"🎯 Модель: {REPO_ID}")
print(f"📋 Пресет «{PRESETS[PRESET]['title']}» — задач: {len(TASKS)}, LIMIT={LIMIT}")
print(f"   {PRESETS[PRESET]['hint']}")
for t in TASKS:
    info = TASK_INFO[t]
    kind = "loglikelihood" if info["type"] == "mc" else "генерация"
    print(f"   • {t:14s} [{kind}, {info['fewshot']}-shot] — {info['desc']}")
print(f"🎲 seed={SEED}, порт={SERVER_PORT}, VRAM-лимит={MAX_VRAM_GB} ГБ, FORCE_RERUN={FORCE_RERUN}")
print(f"📦 llama.cpp: {LLAMA_CPP_ARCHIVE}")
print(f"📂 Результаты: {SAVE_DIR}")
print("=" * 70)
"""
```

- [ ] **Step 2: Собрать и проверить**

Run: `python tools/build_notebook.py`
Expected: `OK: 5 ячеек (4 code), ...`

- [ ] **Step 3: Commit**

```bash
git add tools/build_notebook.py mera_quant_bench.ipynb
git commit -m "Add block 3: config, task presets, paths"
```

---

### Task 4: Блок 4 — установка MERA (форк lm-eval)

**Files:**
- Modify: `tools/build_notebook.py` (+1 ячейка, +1 маркер)

**Interfaces:**
- Produces: `TRANSFORMERS_VERSION: str`, `LM_EVAL_VERSION: str`; установленный CLI `lm_eval`; `sys.path` с `LM_EVAL_PATH`; патч `api_models.py` применён.
- Consumes: `MERA_REPO_DIR`, `LM_EVAL_PATH` (Task 3).

- [ ] **Step 1: Добавить ячейку `BLOCK_04` и маркер `"БЛОК 4"`** (порт Блока 1.1 из V6.1; отличия: без создания папок и сида — они в Блоке 3; патч `api_models.py` применяется безусловно — в V6.1 он ошибочно был внутри ветки `else`)

```python
BLOCK_04 = """\
# @title БЛОК 4: УСТАНОВКА MERA (форк lm-evaluation-harness)
import sys, importlib, importlib.metadata, subprocess

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_DATASETS_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

# 1. Pin transformers<5.00: в 5.x нет AutoModelForVision2Seq, ломается import lm_eval
print("🔧 Фиксация transformers>=4.44,<5.00 (совместимость с форком MERA)...")
r = subprocess.run('pip install "transformers>=4.44,<5.00" accelerate -q',
                   shell=True, capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError(f"❌ Не удалось установить transformers 4.x: {r.stderr[-800:]}")

# 2. Клонирование MERA с подмодулями
if not MERA_REPO_DIR.exists():
    print("📥 Клонирование MERA (с подмодулями)...")
    r = subprocess.run("git clone --recurse-submodules https://github.com/MERA-Evaluation/MERA.git",
                       shell=True, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"❌ Ошибка клонирования MERA: {r.stderr[-800:]}")
else:
    print("ℹ️ MERA уже склонирован. Обновление подмодулей...")
    subprocess.run("cd MERA && git pull --all --rebase --recurse-submodules",
                   shell=True, capture_output=True, text=True)
if not LM_EVAL_PATH.exists():
    raise RuntimeError(f"❌ {LM_EVAL_PATH} не существует — проверьте подмодули MERA.")

# 3. Патч бага логирования в lm-eval (маскирует реальные ошибки) — безусловно
api_models = LM_EVAL_PATH / "lm_eval" / "models" / "api_models.py"
content = api_models.read_text(encoding="utf-8")
patched = content.replace(
    'eval_logger.error(f"Exception:{repr(e)}, {outputs}, retrying.")',
    'eval_logger.error(f"Exception:{repr(e)}, retrying.")')
if patched != content:
    api_models.write_text(patched, encoding="utf-8")
    print("✅ Патч api_models.py применён (UnboundLocalError больше не маскирует сбои).")

# 4. Установка форка без перезаписи зависимостей
print("📦 Установка lm-evaluation-harness из форка MERA...")
r = subprocess.run(f"pip install -e {LM_EVAL_PATH} -q", shell=True,
                   capture_output=True, text=True)
if r.returncode != 0:
    print(f"⚠️ pip install -e вернул код {r.returncode}: {r.stderr[-500:]}")

# 5. Минимальные зависимости
MINIMAL_DEPS = ["huggingface_hub", "datasets", "openai", "tiktoken", "pandas",
                "tqdm", "nest_asyncio", "sqlitedict", "dill", "sacrebleu", "rouge_score"]
subprocess.run(f"pip install {' '.join(MINIMAL_DEPS)} -q", shell=True, capture_output=True)

# 6. Импорт из форка + проверка критичного класса
sys.path.insert(0, str(LM_EVAL_PATH))
for mod in [m for m in list(sys.modules) if m.startswith("lm_eval")]:
    del sys.modules[mod]
import lm_eval
import transformers
from transformers import AutoModelForVision2Seq  # критичная проверка пина
r = subprocess.run("lm_eval --help", shell=True, capture_output=True, text=True, timeout=30)
assert r.returncode == 0, "❌ CLI lm_eval не отвечает"

TRANSFORMERS_VERSION = importlib.metadata.version("transformers")
LM_EVAL_VERSION = importlib.metadata.version("lm_eval")
print(f"✅ Блок 4 завершён: lm_eval {LM_EVAL_VERSION} из {lm_eval.__file__}")
print(f"   transformers {TRANSFORMERS_VERSION}, AutoModelForVision2Seq доступен.")
"""
```

- [ ] **Step 2: Собрать и проверить**

Run: `python tools/build_notebook.py`
Expected: `OK: 6 ячеек (5 code), ...`

- [ ] **Step 3: Commit**

```bash
git add tools/build_notebook.py mera_quant_bench.ipynb
git commit -m "Add block 4: MERA fork install with transformers pin"
```

---

### Task 5: Блок 5 — развёртывание llama.cpp из архива

**Files:**
- Modify: `tools/build_notebook.py` (+1 ячейка, +1 маркер)

**Interfaces:**
- Produces: `LLAMA_SERVER_BIN: str` (путь к бинарнику), `LLAMA_BUILD_INFO: dict` = `{"sha256_tar": str, "commit": str, "branch": str, "cuda": str}`.
- Consumes: `LLAMA_CPP_ARCHIVE`, `LLAMA_CPP_MANIFEST`, `LOCAL_BIN_DIR` (Task 3).

- [ ] **Step 1: Добавить ячейку `BLOCK_05` и маркер `"БЛОК 5"`**

```python
BLOCK_05 = """\
# @title БЛОК 5: РАЗВЁРТЫВАНИЕ LLAMA.CPP ИЗ АРХИВА (SHA-256 по manifest.json)
import hashlib

def sha256_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

archive = Path(LLAMA_CPP_ARCHIVE)
manifest_path = Path(LLAMA_CPP_MANIFEST)
assert archive.exists(), f"❌ Архив не найден: {archive}"
assert manifest_path.exists(), f"❌ manifest.json не найден: {manifest_path}"

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
art = manifest["артефакты"]["native/gpu_all"]
print(f"📦 Архив: {archive} ({archive.stat().st_size / 1e6:.0f} МБ)")
print(f"🔖 Сборка: {manifest['ветка']}, коммит {manifest['коммит']}, CUDA {manifest['cuda_архитектуры']['gpu_all']}")

actual_sha = sha256_file(archive)
if actual_sha.lower() != art["sha256_tar"].lower():
    raise RuntimeError(f"❌ SHA-256 архива не совпал с manifest.json!\\n  ожидание: {art['sha256_tar']}\\n  факт:     {actual_sha}")
print("✅ SHA-256 подтверждён.")

server_bin = next(LOCAL_BIN_DIR.rglob("llama-server"), None)
if server_bin is None:
    print("📥 Распаковка архива...")
    r = subprocess.run(f"tar -xzf '{archive}' -C '{LOCAL_BIN_DIR}'",
                       shell=True, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"❌ Ошибка распаковки: {r.stderr[-800:]}")
    server_bin = next(LOCAL_BIN_DIR.rglob("llama-server"), None)
assert server_bin is not None, "❌ llama-server не найден в архиве."
server_bin.chmod(0o755)

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = f"{server_bin.parent}:{env.get('LD_LIBRARY_PATH', '')}"
r = subprocess.run([str(server_bin), "--version"], capture_output=True, text=True,
                   env=env, timeout=30)
print(f"✅ llama-server: {(r.stdout or r.stderr).strip().splitlines()[0]}")

ldd = subprocess.run(["ldd", str(server_bin)], capture_output=True, text=True,
                     env=env).stdout
missing = [l.strip() for l in ldd.splitlines() if "not found" in l]
if missing:
    raise RuntimeError(f"❌ Недостающие библиотеки: {missing}")

LLAMA_SERVER_BIN = str(server_bin)
LLAMA_BUILD_INFO = {"sha256_tar": actual_sha, "commit": manifest["коммит"],
                    "branch": manifest["ветка"], "cuda": manifest["cuda_архитектуры"]["gpu_all"]}
print(f"✅ Блок 5 завершён: {LLAMA_SERVER_BIN}")
"""
```

- [ ] **Step 2: Собрать и проверить**

Run: `python tools/build_notebook.py`
Expected: `OK: 7 ячеек (6 code), ...`

- [ ] **Step 3: Commit**

```bash
git add tools/build_notebook.py mera_quant_bench.ipynb
git commit -m "Add block 5: llama.cpp deploy from archive with manifest check"
```

---

### Task 6: Блок 6 — вспомогательные функции (самый большой блок)

**Files:**
- Modify: `tools/build_notebook.py` (+1 ячейка, +1 маркер)

**Interfaces:**
- Consumes: всё из Tasks 3–5 (`REPO_ID, TASKS, TASK_INFO, LIMIT, SEED, SERVER_PORT, MAX_VRAM_GB, FORCE_RERUN, CHECKPOINT_PATH, RAW_LOGS_DIR, LOCAL_MODEL_DIR, LOG_DIR, MERA_REPO_DIR, MERA_TASKS_PATH, LOCAL_RESULTS_DIR, LLAMA_SERVER_BIN, LLAMA_BUILD_INFO, TRANSFORMERS_VERSION, LM_EVAL_VERSION, get_time()`).
- Produces (используются Блоками 7–10):
  - `get_available_quants(repo_id: str) -> list[dict]` (`{"filename": str, "size_gb": float}`, сортировка по размеру)
  - `select_reference_model(quants: list[dict], max_vram_gb: float) -> dict | None`
  - `short_name(filename: str) -> str`
  - `load_checkpoint() -> dict` (`{"config_meta": dict, "done": {quant: {"size_gb", "is_reference", <task>: {"primary", "metrics", "wall_s"}}}, "reference": str | None}`)
  - `save_checkpoint(state: dict) -> None`
  - `get_tokenizer_path(repo_id: str) -> str`
  - `kill_existing_server(port: int)`, `start_llama_server(model_path, port, alias) -> (process, stdout_file, stderr_file)`, `stop_llama_server(process, stdout_file, stderr_file)`
  - `smoke_test_echo_logprobs(port: int) -> (bool, str)`
  - `run_task(quant_name: str, task: str, base_repo: str, tokenizer_path: str) -> dict | None` (`{"primary": float | None, "metrics": dict, "wall_s": float}`)
  - `parse_task_metrics(quant_name: str, task: str) -> dict | None` (та же структура, из raw_logs)
  - `download_model(filename: str) -> (Path, float)`, `check_disk_space(required_gb: float)`, `cleanup(model_path)`

- [ ] **Step 1: Добавить ячейку `BLOCK_06` и маркер `"БЛОК 6"`**

```python
BLOCK_06 = '''\
# @title БЛОК 6: ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
import requests
from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata, hf_hub_download, snapshot_download

# ---------- 6.1 Каталог квантов и эталон (логика v5) ----------
def get_available_quants(repo_id):
    print(f"🔍 Кванты в {repo_id}...")
    quants = []
    for f in HfApi().list_repo_files(repo_id, repo_type="model"):
        if f.endswith(".gguf") and "mmproj" not in f.lower():
            meta = get_hf_file_metadata(hf_hub_url(repo_id, f, repo_type="model"))
            quants.append({"filename": f, "size_gb": meta.size / (1024 ** 3)})
    quants.sort(key=lambda x: x["size_gb"])
    print(f"✅ Найдено квантов: {len(quants)}")
    return quants

def select_reference_model(quants, max_vram_gb):
    for tag in ("BF16", "Q8_0"):
        cand = [q for q in quants if tag in q["filename"].upper()]
        if cand and cand[0]["size_gb"] + 4.0 <= max_vram_gb:
            return cand[0]
    fitting = [q for q in quants if q["size_gb"] + 4.0 <= max_vram_gb]
    return max(fitting, key=lambda x: x["size_gb"]) if fitting else None

def short_name(filename):
    return filename.replace(f"{BASE_MODEL_NAME}-", "").replace(".gguf", "")

# ---------- 6.2 Чекпоинт (атомарный, с config_meta) ----------
CONFIG_KEYS = ["repo_id", "preset", "limit", "seed", "tasks"]
VERSION_KEYS = ["transformers_version", "lm_eval_version", "llama_cpp_sha256"]

def build_config_meta():
    return {"repo_id": REPO_ID, "preset": PRESET, "limit": LIMIT, "seed": SEED,
            "tasks": sorted(TASKS),
            "transformers_version": TRANSFORMERS_VERSION,
            "lm_eval_version": LM_EVAL_VERSION,
            "llama_cpp_sha256": LLAMA_BUILD_INFO["sha256_tar"][:16]}

def fresh_state():
    return {"config_meta": build_config_meta(), "done": {}, "reference": None}

def load_checkpoint():
    if FORCE_RERUN:
        print("⚠️ FORCE_RERUN=True: чекпоинт игнорируется, всё пересчитывается.")
        return fresh_state()
    p = Path(CHECKPOINT_PATH)
    if p.exists():
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ Чекпоинт повреждён ({e}) — начинаем с чистого листа.")
            return fresh_state()
        meta = build_config_meta()
        old = state.get("config_meta", {})
        diff = [k for k in CONFIG_KEYS if old.get(k) != meta[k]]
        if diff:
            raise RuntimeError(
                f"❌ Чекпоинт {p} собран с другим конфигом (отличается: {diff}).\\n"
                f"   Установите FORCE_RERUN=True или удалите {SAVE_DIR}.")
        for k in VERSION_KEYS:
            if old.get(k) != meta[k]:
                print(f"⚠️ Дрейф {k}: чекпоинт={old.get(k)} → сейчас={meta[k]} (прогон продолжается, помечено в отчёте).")
        print(f"💾 Чекпоинт: готово квантов — {len(state['done'])}.")
        return state
    return fresh_state()

def save_checkpoint(state):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, CHECKPOINT_PATH)

# ---------- 6.3 Токенизатор (кэш на Drive + патч config.json; порт V6.1) ----------
def get_tokenizer_path(repo_id):
    base_repo = repo_id.replace("-GGUF", "")
    safe = base_repo.replace("/", "_")
    local_dir = LOCAL_MODEL_DIR / f"tokenizer_{safe}"
    drive_dir = SAVE_DIR / "tokenizers" / f"tokenizer_{safe}"
    if drive_dir.exists() and any(drive_dir.iterdir()):
        if not local_dir.exists():
            shutil.copytree(drive_dir, local_dir, dirs_exist_ok=True)
        return str(local_dir)
    if not (local_dir.exists() and any(local_dir.iterdir())):
        print(f"  ⬇️ Токенизатор {base_repo}...")
        snapshot_download(repo_id=base_repo, local_dir=str(local_dir),
                          allow_patterns=["tokenizer*", "special_tokens_map.json",
                                          "tokenizer_config.json", "vocab.json",
                                          "merges.txt", "*.model", "config.json"],
                          token=os.environ.get("HF_TOKEN"))
    cfg = local_dir / "config.json"
    if cfg.exists():
        try:
            config = json.loads(cfg.read_text(encoding="utf-8"))
            changed = False
            for key in ("routed_scaling_factor", "rope_theta", "sliding_window"):
                if key in config and isinstance(config[key], int):
                    config[key] = float(config[key])
                    changed = True
            if changed:
                cfg.write_text(json.dump(config, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"  ⚠️ Патч config.json не удался: {e}")
    drive_dir.mkdir(parents=True, exist_ok=True)
    for item in local_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, drive_dir / item.name)
    return str(local_dir)
'''
```

Внимание: в функции выше опечатка быть не может — `cfg.write_text(json.dumps(config, indent=2), encoding="utf-8")` (не `json.dump`). Ниже — продолжение той же ячейки (в `tools/build_notebook.py` это одна строковая константа `BLOCK_06`, части 6.4–6.6 дописываются в неё):

```python
BLOCK_06 += '''\

# ---------- 6.4 Сервер llama.cpp (порт V6.1: изоляция группы процессов) ----------
def kill_existing_server(port):
    r = subprocess.run(f"lsof -t -i:{port}", shell=True, capture_output=True, text=True)
    for pid in (r.stdout.strip().split("\\n") if r.stdout.strip() else []):
        if pid:
            subprocess.run(f"kill -9 {pid}", shell=True, capture_output=True)
            time.sleep(1)
    subprocess.run("pkill -9 -f llama-server", shell=True, capture_output=True)

def start_llama_server(model_path, port, alias):
    stdout_f = open(LOG_DIR / "server_stdout.log", "w", encoding="utf-8")
    stderr_f = open(LOG_DIR / "server_stderr.log", "w", encoding="utf-8")
    cmd = [LLAMA_SERVER_BIN, "-m", str(model_path),
           "--host", "127.0.0.1", "--port", str(port),
           "-ngl", "-1", "-c", "8192", "-b", "2048", "-t", "8",
           "--seed", str(SEED), "--alias", alias]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{Path(LLAMA_SERVER_BIN).parent}:{env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env,
                            start_new_session=True)  # изоляция от SIGINT Colab
    for _ in range(60):
        if proc.poll() is not None:
            stop_llama_server(proc, stdout_f, stderr_f)
            err = (LOG_DIR / "server_stderr.log").read_text(encoding="utf-8")[-2000:]
            raise RuntimeError(f"❌ Сервер упал при старте (code {proc.returncode}).\\n{err}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return proc, stdout_f, stderr_f
        except Exception:
            pass
        time.sleep(2)
    stop_llama_server(proc, stdout_f, stderr_f)
    raise RuntimeError("❌ Сервер не ответил за 120 сек (таймаут /health).")

def stop_llama_server(proc, stdout_f, stderr_f):
    if proc:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:
                proc.kill()
    for f in (stdout_f, stderr_f):
        if f:
            try:
                f.close()
            except Exception:
                pass

def smoke_test_echo_logprobs(port):
    """Критический гейт: echo+logprobs в /v1/completions (PR #27537)."""
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/v1/completions",
            json={"prompt": "Столица России — Москва. Столица Франции — ",
                  "max_tokens": 1, "echo": True, "logprobs": 5},
            timeout=180)
        r.raise_for_status()
        lp = (r.json()["choices"][0].get("logprobs") or {}).get("token_logprobs")
        n = len(lp) if lp else 0
        if n < 5:
            return False, f"получено {n} logprob-ов (нужно ≥ 5) — патч echo/logprobs не работает"
        return True, f"OK: {n} logprob-ов на echo-запросе"
    except Exception as e:
        return False, f"запрос не прошёл: {e!r}"

# ---------- 6.5 Запуск одной задачи (lm_eval CLI, локальный скоринг) ----------
def run_task(quant_name, task, base_repo, tokenizer_path):
    out_dir = LOCAL_RESULTS_DIR / quant_name / task
    out_dir.mkdir(parents=True, exist_ok=True)
    model_args = (f"model={base_repo},"
                  f"base_url=http://127.0.0.1:{SERVER_PORT}/v1/completions,"
                  f"num_concurrent=8,"
                  f"tokenizer_backend=huggingface,"
                  f"tokenizer={tokenizer_path},"
                  f"tokenized_requests=False,"
                  f"timeout=1000000")
    cmd = ["lm_eval", "--model", "local-completions", "--model_args", model_args,
           "--tasks", task, "--include_path", str(MERA_TASKS_PATH),
           "--output_path", str(out_dir), "--log_samples",
           "--seed", str(SEED), "--batch_size", "1", "--verbosity", "ERROR"]
    if LIMIT:
        cmd += ["--limit", str(LIMIT)]
    if TASK_INFO[task]["type"] == "gen":
        cmd += ["--gen_kwargs", "do_sample=False"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(MERA_REPO_DIR)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "23400000"
    env["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        print(f"  ❌ [{get_time()}] {task}: таймаут CLI > 2 ч")
        return None
    wall_s = round(time.time() - t0, 1)

    task_log_dir = RAW_LOGS_DIR / quant_name / task
    task_log_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.rglob("results_*.json"):
        shutil.copy2(f, task_log_dir / f.name)
    for f in out_dir.rglob(f"samples_{task}_*.jsonl"):
        shutil.copy2(f, task_log_dir / f.name)
    (task_log_dir / "task_meta.json").write_text(
        json.dumps({"quant": quant_name, "task": task, "wall_s": wall_s,
                    "exit_code": result.returncode, "cmd": cmd,
                    "stderr_tail": result.stderr[-2000:]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    if result.returncode != 0:
        print(f"  ❌ [{get_time()}] {task}: exit {result.returncode}: {result.stderr[-400:]}")
        return None
    parsed = parse_task_metrics(quant_name, task)
    if parsed is not None:
        parsed["wall_s"] = wall_s
    return parsed

def parse_task_metrics(quant_name, task):
    files = sorted((RAW_LOGS_DIR / quant_name / task).glob("results_*.json"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    raw = json.loads(files[-1].read_text(encoding="utf-8"))
    res = (raw.get("results") or {}).get(task)
    if not isinstance(res, dict):
        return None
    metrics = {k: v for k, v in res.items() if isinstance(v, (int, float))}
    primary_key = TASK_INFO[task]["primary_key"]
    if primary_key == "mcc_mean":
        mcc = [v for k, v in res.items() if k.startswith("mcc_") and isinstance(v, (int, float))]
        primary = round(sum(mcc) / len(mcc), 4) if mcc else None
    else:
        primary = res.get(primary_key)
        primary = round(primary, 4) if isinstance(primary, (int, float)) else None
    return {"primary": primary, "metrics": metrics}

# ---------- 6.6 Скачивание, место, очистка (логика v5) ----------
def check_disk_space(required_gb):
    free_gb = shutil.disk_usage("/").free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(f"⚠️ Мало места: свободно {free_gb:.1f} ГБ, требуется {required_gb:.1f} ГБ.")

def download_model(filename):
    print(f"  ⬇️ Скачивание {filename}...")
    t0 = time.time()
    path = hf_hub_download(repo_id=REPO_ID, filename=filename,
                           local_dir=str(LOCAL_MODEL_DIR),
                           token=os.environ.get("HF_TOKEN"))
    size_gb = round(os.path.getsize(path) / (1024 ** 3), 2)
    print(f"  ✅ {size_gb} ГБ за {time.time() - t0:.0f} сек")
    return Path(path), size_gb

def cleanup(model_path):
    gc.collect()
    time.sleep(1)
    if model_path and os.path.exists(model_path):
        os.remove(model_path)
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)
    print("  🧹 Очистка выполнена (модель и HF-кэш удалены; raw_logs и чекпоинт не тронуты).")

_required = ["get_available_quants", "select_reference_model", "short_name",
             "load_checkpoint", "save_checkpoint", "get_tokenizer_path",
             "kill_existing_server", "start_llama_server", "stop_llama_server",
             "smoke_test_echo_logprobs", "run_task", "parse_task_metrics",
             "check_disk_space", "download_model", "cleanup"]
_missing = [n for n in _required if n not in globals()]
assert not _missing, f"❌ Не определены функции: {_missing}"
print(f"✅ Блок 6 завершён: {len(_required)} функций определены.")
'''
```

**Важная деталь транскрипции:** `BLOCK_06` — ОДНА строковая константа (两部分 выше объединяются `BLOCK_06 += ...`). Внутри неё строки с `{`/`}` не требуют экранирования (используется `'''...'''` без f-строк), но `\n` внутри одинарных-тройных кавычек должен остаться как escape `\\n` в исходниках build-скрипта там, где нужен перевод строки внутри строки notebook (как в примерах выше). Также исправить в части 6.3: использовать `json.dumps` (см. примечание выше).

- [ ] **Step 2: Собрать и проверить**

Run: `python tools/build_notebook.py`
Expected: `OK: 8 ячеек (7 code), ...` — ast.parse гарантирует синтаксис ячейки Блок 6 целиком.

- [ ] **Step 3: Commit**

```bash
git add tools/build_notebook.py mera_quant_bench.ipynb
git commit -m "Add block 6: quants, checkpoint, server, echo-gate, run_task"
```

---

### Task 7: Блок 7 — основной цикл (эталон + кванты)

**Files:**
- Modify: `tools/build_notebook.py` (+1 ячейка, +1 маркер)

**Interfaces:**
- Consumes: все функции Блока 6, `TASKS, MAX_VRAM_GB, REPO_ID, SERVER_PORT, BASE_MODEL_NAME` (Task 3).
- Produces: заполненный `state`/чекпоинт на Drive; глобальные `REF_NAME: str`, `all_quants: list[dict]`; функция `evaluate_quant(q_info: dict, is_reference: bool = False) -> None`.

- [ ] **Step 1: Добавить ячейку `BLOCK_07` и маркер `"БЛОК 7"`**

```python
BLOCK_07 = """\
# @title БЛОК 7: ОСНОВНОЙ ЦИКЛ (ЭТАП 0 — эталон, ЭТАП 1 — кванты)
state = load_checkpoint()
done = state["done"]

all_quants = get_available_quants(REPO_ID)
ref_q = select_reference_model(all_quants, MAX_VRAM_GB)
if ref_q is None:
    raise RuntimeError(f"❌ Ни один квант не влезает в {MAX_VRAM_GB} ГБ VRAM.")
REF_NAME = short_name(ref_q["filename"])
print(f"🏆 Эталон: {ref_q['filename']} ({ref_q['size_gb']:.1f} ГБ)")

def evaluate_quant(q_info, is_reference=False):
    name = short_name(q_info["filename"])
    pending = [t for t in TASKS if not (name in done and t in done.get(name, {}) and t != "size_gb")]
    pending = [t for t in pending if t in TASK_INFO]
    if name in done and not pending:
        print(f"⏭️ [{get_time()}] {name}: все задачи готовы (чекпоинт).")
        return
    print(f"\\n{'=' * 70}\\n▶ [{get_time()}] {name} ({q_info['size_gb']:.1f} ГБ), задач: {len(pending)}\\n{'=' * 70}")
    check_disk_space(q_info["size_gb"] * 2.2 + 5.0)
    model_path = None
    server = None
    try:
        kill_existing_server(SERVER_PORT)
        model_path, _size = download_model(q_info["filename"])
        tokenizer_path = get_tokenizer_path(REPO_ID)
        print(f"  🚀 [{get_time()}] Запуск llama-server (seed={SEED})...")
        server = start_llama_server(model_path, SERVER_PORT, alias=REPO_ID.replace("-GGUF", ""))
        if is_reference:
            ok, msg = smoke_test_echo_logprobs(SERVER_PORT)
            print(f"  🧪 Гейт echo/logprobs: {msg}")
            if not ok:
                raise RuntimeError(
                    "❌ Сборка llama-server не отдаёт echo+logprobs — loglikelihood-задачи "
                    "не пройдут. Проверьте архив (PR #27537) и Блок 5. Прогон остановлен "
                    "ДО траты GPU-времени.")
        row = done.setdefault(name, {})
        row["size_gb"] = q_info["size_gb"]
        if is_reference:
            row["is_reference"] = True
            state["reference"] = name
        for task in pending:
            print(f"  🧪 [{get_time()}] {task} ...", end=" ", flush=True)
            res = run_task(name, task, REPO_ID.replace("-GGUF", ""), tokenizer_path)
            if res is None:
                print("❌ (не в чекпоинт — повторится при следующем запуске)")
                continue
            row[task] = res
            save_checkpoint(state)
            print(f"✅ primary={res['primary']} ({res['wall_s']} c)")
        save_checkpoint(state)
    except KeyboardInterrupt:
        print(f"\\n⏹️ [{get_time()}] {name}: прервано. Выполненное — в чекпоинте.")
    finally:
        if server:
            stop_llama_server(*server)
        kill_existing_server(SERVER_PORT)
        cleanup(model_path)

# ---------- ЭТАП 0: эталон (BF16 → Q8_0) ----------
if REF_NAME in done and any(t in done[REF_NAME] for t in TASKS):
    print(f"\\n⏭️ ЭТАП 0: эталон {REF_NAME} уже посчитан (чекпоинт).")
else:
    print(f"\\n{'=' * 70}\\n🏁 ЭТАП 0: ЭТАЛОН {ref_q['filename']}\\n{'=' * 70}")
    evaluate_quant(ref_q, is_reference=True)

# ---------- ЭТАП 1: остальные кванты ----------
print(f"\\n{'=' * 70}\\n🔄 ЭТАП 1: ОСТАЛЬНЫЕ КВАНТЫ\\n{'=' * 70}")
for q in all_quants:
    name = short_name(q["filename"])
    if name == REF_NAME:
        continue
    if q["size_gb"] + 4.0 > MAX_VRAM_GB:
        print(f"⏭️ {name}: не влезает в VRAM ({q['size_gb']:.1f} ГБ + 4 ГБ > {MAX_VRAM_GB} ГБ)")
        continue
    evaluate_quant(q)

print(f"\\n🏁 [{get_time()}] ЦИКЛ ЗАВЕРШЁН. Переходите к Блоку 8.")
"""
```

- [ ] **Step 2: Собрать и проверить**

Run: `python tools/build_notebook.py`
Expected: `OK: 9 ячеек (8 code), ...`

- [ ] **Step 3: Commit**

```bash
git add tools/build_notebook.py mera_quant_bench.ipynb
git commit -m "Add block 7: resumable main loop with reference and echo gate"
```

---

### Task 8: Блок 8 — агрегация, деградация, графики, отчёт

**Files:**
- Modify: `tools/build_notebook.py` (+1 ячейка, +1 маркер)

**Interfaces:**
- Consumes: `CHECKPOINT_PATH, TASKS, TASK_INFO, SAVE_DIR, REF_NAME` + pandas/matplotlib.
- Produces: на Drive — `mera_results.csv`, `degradation.csv`, `degradation_plots.png`, `mera_quant_report.md`; глобальные `df` (DataFrame), `unreliable: list[str]`, `reliable: list[str]`.

- [ ] **Step 1: Добавить ячейку `BLOCK_08` и маркер `"БЛОК 8"`**

```python
BLOCK_08 = '''\
# @title БЛОК 8: АГРЕГАЦИЯ, ДЕГРАДАЦИЯ, ГРАФИКИ, ОТЧЁТ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

state = json.loads(Path(CHECKPOINT_PATH).read_text(encoding="utf-8"))
done = state["done"]
meta = state.get("config_meta", {})

rows = []
for name, cell in done.items():
    row = {"Quant": name, "Size_GB": cell.get("size_gb"),
           "Is_Ref": bool(cell.get("is_reference"))}
    for t in TASKS:
        row[t] = (cell.get(t) or {}).get("primary")
    rows.append(row)
df = pd.DataFrame(rows).sort_values("Size_GB").reset_index(drop=True)

ref_rows = df[df["Is_Ref"]]
ref_row = ref_rows.iloc[0] if len(ref_rows) else None

# Правило ненадёжности: у эталона метрика None/NaN/0 → задача исключается из средних
unreliable = []
for t in TASKS:
    v = ref_row.get(t) if ref_row is not None else None
    if v is None or (isinstance(v, float) and (np.isnan(v) or v == 0)):
        unreliable.append(t)
reliable = [t for t in TASKS if t not in unreliable]
if unreliable:
    print(f"⚠️ Ненадёжные задачи (у эталона 0/нет данных), исключены из средних: {unreliable}")

df["AvgScore"] = df[reliable].mean(axis=1) if reliable else np.nan
ref_avg = float(ref_row["AvgScore"]) if ref_row is not None and not pd.isna(ref_row.get("AvgScore", np.nan)) else np.nan
df["Delta_pp"] = (df["AvgScore"] - ref_avg) * 100
df["Delta_pct"] = (df["AvgScore"] / ref_avg - 1) * 100 if ref_avg else np.nan
for t in reliable:
    df[f"{t}_dpp"] = (df[t] - float(ref_row[t])) * 100

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
cols = ["Quant", "Size_GB", "Is_Ref"] + reliable + ["AvgScore", "Delta_pp", "Delta_pct"]
display(df[[c for c in cols if c in df.columns]].round(4))

df.to_csv(SAVE_DIR / "mera_results.csv", index=False, encoding="utf-8")
df[["Quant", "Size_GB", "Is_Ref", "AvgScore", "Delta_pp", "Delta_pct"]].round(4) \\
    .to_csv(SAVE_DIR / "degradation.csv", index=False, encoding="utf-8")

# ---------- Графики ----------
sns_style = plt.style.context("default")
fig, axes = plt.subplots(1 + (len(reliable) + 3) // 4, 4, figsize=(20, 4.5 * (1 + (len(reliable) + 3) // 4)))
axes = np.atleast_2d(axes).ravel()
panels = reliable + ["AvgScore"]
for ax, t in zip(axes, panels):
    sub = df.dropna(subset=[t]) if t in df.columns else df.dropna(subset=["AvgScore"])
    ax.plot(sub["Size_GB"], sub[t], "o-")
    for _, r in sub.iterrows():
        ax.annotate(r["Quant"], (r["Size_GB"], r[t]), fontsize=7,
                    xytext=(0, 5), textcoords="offset points", ha="center")
    ax.set_xlabel("Размер, ГБ")
    ax.set_title(t + (" (среднее)" if t == "AvgScore" else ""), fontweight="bold")
    if t != "AvgScore" and ref_row is not None and not pd.isna(ref_row.get(t, np.nan)):
        ax.axhline(float(ref_row[t]), color="gray", ls=":", lw=1)
for ax in axes[len(panels):]:
    ax.axis("off")
fig.suptitle(f"Деградация метрик MERA: {REPO_ID} (пресет {PRESET}, LIMIT={LIMIT})", fontweight="bold")
fig.tight_layout()
fig.savefig(SAVE_DIR / "degradation_plots.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------- Heatmap Δ п.п. ----------
if reliable:
    hm = df.set_index("Quant")[[f"{t}_dpp" for t in reliable]].round(2)
    fig2, ax2 = plt.subplots(figsize=(1.6 * len(reliable) + 4, 0.5 * len(hm) + 2))
    im = ax2.imshow(hm.values, cmap="RdYlGn", aspect="auto")
    ax2.set_xticks(range(len(hm.columns)), hm.columns, rotation=30, ha="right")
    ax2.set_yticks(range(len(hm.index)), hm.index)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8)
    fig2.colorbar(im, label="Δ к эталону, п.п.")
    ax2.set_title("Деградация по задачам (п.п., меньше — хуже)", fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(SAVE_DIR / "degradation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.show()

# ---------- MD-отчёт ----------
def _md_table(dframe, floatfmt=4):
    d = dframe.round(floatfmt)
    head = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = "\\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in d.astype(object).values)
    return "\\n".join([head, sep, body])

md = []
md.append(f"# Деградация MERA по квантам: {REPO_ID}")
md.append(f"**Пресет:** {PRESET} ({PRESETS[PRESET]['title']}), LIMIT={LIMIT}, seed={SEED}")
md.append(f"**Эталон:** {REF_NAME}")
md.append(f"**Среда:** transformers {meta.get('transformers_version')} · lm-eval {meta.get('lm_eval_version')} · "
          f"llama.cpp {LLAMA_BUILD_INFO['branch']}@{LLAMA_BUILD_INFO['commit']} (CUDA {LLAMA_BUILD_INFO['cuda']})")
md.append(f"**GPU:** {torch.cuda.get_device_properties(0).name}")
if unreliable:
    md.append(f"\\n⚠️ **Ненадёжные задачи** (у эталона 0/нет данных — исключены из средних): {', '.join(unreliable)}")
md.append("\\n## Сводная таблица\\n")
md.append(_md_table(df[[c for c in cols if c in df.columns]]))
md.append("\\n## Деградация по задачам (Δ п.п.)\\n")
md.append(_md_table(df[["Quant", "Size_GB"] + [f"{t}_dpp" for t in reliable]]) if reliable else "Нет надёжных задач.")
(SAVE_DIR / "mera_quant_report.md").write_text("\\n".join(md), encoding="utf-8")
print(f"\\n💾 Сохранено в {SAVE_DIR}: mera_results.csv, degradation.csv, degradation_plots.png, "
      f"degradation_heatmap.png, mera_quant_report.md")
'''
```

- [ ] **Step 2: Собрать и проверить**

Run: `python tools/build_notebook.py`
Expected: `OK: 10 ячеек (9 code), ...`

- [ ] **Step 3: Commit**

```bash
git add tools/build_notebook.py mera_quant_bench.ipynb
git commit -m "Add block 8: aggregation, degradation, plots, markdown report"
```

---

### Task 9: Блоки 9–10 — финальный рейтинг и пересборка из логов

**Files:**
- Modify: `tools/build_notebook.py` (+2 ячейки, +2 маркера `"БЛОК 9"`, `"БЛОК 10"`)

**Interfaces:**
- Consumes: `df, unreliable, reliable, ref_avg, SAVE_DIR, RAW_LOGS_DIR, TASKS, parse_task_metrics` .
- Produces: `final_ranking.csv`, `final_analysis.png` (Drive); `reparsed_from_raw.csv` (Drive).

- [ ] **Step 1: Добавить ячейки `BLOCK_09`, `BLOCK_10` и маркеры**

```python
BLOCK_09 = """\
# @title БЛОК 9: ФИНАЛЬНЫЙ РЕЙТИНГ (вердикты + Pareto)
def verdict(delta_pct):
    if delta_pct is None or (isinstance(delta_pct, float) and np.isnan(delta_pct)):
        return "нет данных"
    deg = -delta_pct
    if deg <= 1:  return "≈ без потерь"
    if deg <= 3:  return "рекомендуется"
    if deg <= 7:  return "приемлемо"
    return "⚠️ заметная деградация"

df["Verdict"] = df["Delta_pct"].apply(verdict)
df.loc[df["Is_Ref"], "Verdict"] = "🏆 ЭТАЛОН"
rank_cols = ["Quant", "Size_GB", "AvgScore", "Delta_pp", "Delta_pct", "Verdict"]
display(df[rank_cols].round(4))
df[rank_cols].round(4).to_csv(SAVE_DIR / "final_ranking.csv", index=False, encoding="utf-8")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sub = df.dropna(subset=["AvgScore"])
ax1.plot(sub["Size_GB"], sub["AvgScore"], "o-")
for _, r in sub.iterrows():
    ax1.annotate(r["Quant"], (r["Size_GB"], r["AvgScore"]), fontsize=8,
                 xytext=(0, 6), textcoords="offset points", ha="center")
ax1.set_xlabel("Размер, ГБ"); ax1.set_ylabel("Средний балл MERA")
ax1.set_title(f"Средний балл vs размер · {BASE_MODEL_NAME}", fontweight="bold")

pts = sub[~sub["Is_Ref"]].sort_values("AvgScore", ascending=False)
front, best = [], -np.inf
for _, r in pts.iterrows():
    if r["Size_GB"] < best:
        front.append(r["Quant"]); best = r["Size_GB"]
ax2.scatter(pts["Size_GB"], pts["AvgScore"], s=40, color="lightgray", label="Все кванты")
fp = pts[pts["Quant"].isin(front)]
ax2.scatter(fp["Size_GB"], fp["AvgScore"], s=80, color="tab:blue", label="Pareto-фронт")
for _, r in fp.iterrows():
    ax2.annotate(r["Quant"], (r["Size_GB"], r["AvgScore"]), fontsize=8,
                 xytext=(4, 4), textcoords="offset points")
ax2.set_xlabel("Размер, ГБ"); ax2.set_ylabel("Средний балл")
ax2.set_title("Pareto: меньше ГБ при том же качестве", fontweight="bold")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(SAVE_DIR / "final_analysis.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"💾 Итоговый рейтинг: {SAVE_DIR / 'final_ranking.csv'}")
"""

BLOCK_10 = """\
# @title БЛОК 10: ПЕРЕСБОРКА ИЗ СЫРЫХ ЛОГОВ (без GPU и без моделей)
rows = []
for quant_dir in sorted(p for p in RAW_LOGS_DIR.iterdir() if p.is_dir()):
    row = {"Quant": quant_dir.name}
    for task_dir in sorted(p for p in quant_dir.iterdir() if p.is_dir()):
        if task_dir.name not in TASK_INFO:
            continue
        parsed = parse_task_metrics(quant_dir.name, task_dir.name)
        if parsed:
            row[task_dir.name] = parsed["primary"]
            row[f"{task_dir.name}_wall_s"] = parsed.get("wall_s")
    rows.append(row)
df_re = pd.DataFrame(rows)
display(df_re)
df_re.to_csv(SAVE_DIR / "reparsed_from_raw.csv", index=False, encoding="utf-8")
print(f"💾 {SAVE_DIR / 'reparsed_from_raw.csv'}")
"""
```

- [ ] **Step 2: Собрать и проверить**

Run: `python tools/build_notebook.py`
Expected: `OK: 12 ячеек (11 code), ...`

- [ ] **Step 3: Commit**

```bash
git add tools/build_notebook.py mera_quant_bench.ipynb
git commit -m "Add blocks 9-10: verdicts, Pareto ranking, raw-log rebuild"
```

---

### Task 10: README.md

**Files:**
- Create: `README.md` (в корне репо)

**Interfaces:**
- Consumes: факты спеки (§3 таблица задач, §4 пресеты, §10.1 подход transformers, §10.2 ограничения).

- [ ] **Step 1: Написать `README.md`** — полный текст:

```markdown
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

Форк MERA (база LM-Harness v0.4.8) импортирует `AutoModelForVision2Seq`, которого нет в
transformers 5.x. Решение: **pin `transformers>=4.44,<5.00`** — проверено прогоном
RU_LLM_Benchmarks V6.1 ровно на этом стеке (local-completions + llama-server). Если
будущий образ Colab не позволит ставить 4.x — фолбэк (патч двух файлов форка, подход из
серии MERA vLLM v2):

```python
TARGET_FILES = ["lm_eval/models/huggingface.py", "lm_eval/models/hf_vlms.py"]
for fp in [f"{LM_EVAL_PATH}/{t}" for t in TARGET_FILES]:
    content = open(fp, encoding="utf-8").read()
    if "AutoModelForVision2Seq" not in content:
        continue
    content = content.replace(
        "from transformers import AutoModelForVision2Seq",
        "from transformers import AutoModelForCausalLM as AutoModelForVision2Seq")
    content = content.replace(
        "transformers.AutoModelForVision2Seq",
        "getattr(transformers, 'AutoModelForImageTextToText', "
        "getattr(transformers, 'AutoModelForVision2Seq', type(None)))")
    open(fp, "w", encoding="utf-8").write(content)
```

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
```

- [ ] **Step 2: Проверить согласованность README с notebook**

Run: `git grep -n "smoke\|choice\|generation\|all" -- mera_quant_bench.ipynb | Select-String "PRESET"` (PowerShell) — убедиться, что пресеты в README совпадают с `PRESETS` в Блоке 3 (smoke: bps+simplear/20; choice: 5 mc/100; generation: simplear/100; all: 6/100). Вручную сверить пути артефактов README ↔ Блок 8/9.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Add README: usage, presets, methodology, limitations"
```

---

### Task 11: Финальная верификация и публикация на GitHub

**Files:**
- Modify: ничего нового; проверка целостности репо.

**Interfaces:**
- Consumes: всё.

- [ ] **Step 1: Финальная сборка + проверка**

Run: `python tools/build_notebook.py`
Expected: `OK: 12 ячеек (11 code), все code-ячейки парсятся, порядок блоков верный`

- [ ] **Step 2: Проверить структуру репо и статус git**

Run (PowerShell): `git status --short; Get-ChildItem -Recurse -File | Select-Object FullName`
Expected: `README.md, .gitignore, docs/superpowers/specs/..., docs/superpowers/plans/..., tools/build_notebook.py, mera_quant_bench.ipynb`; `git status` чистый (всё закоммичено).

- [ ] **Step 3: Сверить ipynb с Colab-требованиями**

Run: `python -c "import json; nb=json.load(open('mera_quant_bench.ipynb',encoding='utf-8')); assert nb['nbformat']==4; assert nb['metadata']['colab']; print('notebook OK')"`
Expected: `notebook OK`

- [ ] **Step 4: Создать публичный репозиторий и запушить**

```bash
gh repo create GGUF-LLM-Test/llama-cpp-mera-quant-bench --public --description "MERA benchmark degradation across GGUF quantization levels, served by prebuilt llama.cpp (v0.4.1+PR#27537) via lm-eval local-completions" --source . --push
```
Expected: URL `https://github.com/GGUF-LLM-Test/llama-cpp-mera-quant-bench`. Если репо уже существует — `git remote add origin https://github.com/GGUF-LLM-Test/llama-cpp-mera-quant-bench.git; git push -u origin main` и сообщить.

- [ ] **Step 5: Commit (если что-то осталось) и финальный отчёт**

```bash
git status --short
git log --oneline
```
Expected: чистое дерево, ~12 коммитов. Сообщить пользователю URL репозитория и инструкцию первого запуска (smoke-пресет в Colab).

---

## Self-Review (выполнено при написании плана)

- **Покрытие спеки:** Блоки 0–10 (спека §5) → Tasks 2–9; пресеты (§4) → Task 3; выбор задач (§3) → Task 3 `TASK_INFO` + README (Task 10); echo-гейт (§8) → Task 6 (`smoke_test_echo_logprobs`) + Task 7 (ЭТАП 0); чекпоинт/config_meta (§5, §8) → Task 6; правило ненадёжности (§6) → Task 8; вердикты/Pareto (§5 Блок 9) → Task 9; пересборка (§5 Блок 10) → Task 9; transformers-подход §10.1 → Task 4 + README; публикация (§2.8) → Task 11. Пропусков нет.
- **Плейсхолдеры:** отсутствуют — весь код финальный.
- **Консистентность имён:** `TASK_INFO[t]["primary_key"]`/`["type"]`/`["fewshot"]` (Task 3) ↔ использование в Task 6 (`run_task`, `parse_task_metrics`) — совпадает; `LLAMA_BUILD_INFO["sha256_tar"]` (Task 5) ↔ `build_config_meta` (Task 6) — совпадает; `state["reference"]` (Task 6/7) ↔ Task 8 — используется `Is_Ref`, оба поля пишутся.
