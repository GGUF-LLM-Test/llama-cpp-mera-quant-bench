"""Сборка и проверка mera_quant_bench.ipynb.

Единственный источник правды для notebook. Правки делаются здесь,
затем: python tools/build_notebook.py
"""
import ast
import json
import random
from pathlib import Path

random.seed(20260916)

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

EXPECTED_BLOCKS = ["БЛОК 0", "БЛОК 1", "БЛОК 2", "БЛОК 3", "БЛОК 4", "БЛОК 5", "БЛОК 6", "БЛОК 7"]


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
                cfg.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"  ⚠️ Патч config.json не удался: {e}")
    drive_dir.mkdir(parents=True, exist_ok=True)
    for item in local_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, drive_dir / item.name)
    return str(local_dir)
'''

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
    code_cell(BLOCK_06),
    code_cell(BLOCK_07),
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
    build()
    verify()
