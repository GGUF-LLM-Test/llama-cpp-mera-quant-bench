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

EXPECTED_BLOCKS = ["БЛОК 0", "БЛОК 1", "БЛОК 2", "БЛОК 3", "БЛОК 4", "БЛОК 5"]


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
if r.returncode != 0:
    out = (r.stdout or "")[-800:] + (r.stderr or "")[-800:]
    raise RuntimeError(f"❌ llama-server --version завершился с кодом {r.returncode}: {out}")
ver_line = ((r.stdout or r.stderr or "(пусто)").strip().splitlines() or ["(пусто)"])[0]
print(f"✅ llama-server: {ver_line}")

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
    code_cell(BLOCK_04),
    code_cell(BLOCK_05),
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
