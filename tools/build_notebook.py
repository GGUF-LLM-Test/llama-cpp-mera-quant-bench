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

EXPECTED_BLOCKS = ["БЛОК 0", "БЛОК 1", "БЛОК 2", "БЛОК 3"]


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


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "id": "%08x" % random.getrandbits(32),
            "metadata": {}, "source": src.splitlines(keepends=True)}


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "id": "%08x" % random.getrandbits(32),
            "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


CELLS = [
    md_cell(MD_INTRO),
    code_cell(BLOCK_00),
    code_cell(BLOCK_01),
    code_cell(BLOCK_02),
    code_cell(BLOCK_03),
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
    titles = ["".join(c["source"]).splitlines()[0] for c in code_cells]
    for marker in EXPECTED_BLOCKS:
        assert any(marker + ":" in t for t in titles), f"Не найден блок: {marker}"
    pos = [-1]
    for marker in EXPECTED_BLOCKS:
        idx = next(i for i, t in enumerate(titles) if marker + ":" in t)
        assert idx > pos[-1], f"Нарушен порядок блоков: {marker}"
        pos.append(idx)
    print(f"OK: {len(nb['cells'])} ячеек ({len(code_cells)} code), "
          f"все code-ячейки парсятся, порядок блоков верный")


if __name__ == "__main__":
    random.seed(20260916)
    build()
    verify()
