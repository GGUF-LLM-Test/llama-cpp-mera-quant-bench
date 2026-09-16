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

EXPECTED_BLOCKS = ["БЛОК 0", "БЛОК 1", "БЛОК 2"]


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
