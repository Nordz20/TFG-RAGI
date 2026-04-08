import json
import time
from pathlib import Path
import requests
import re

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "data"
URL = "https://wiig.dia.fi.upm.es/ollama/v1/chat/completions"
MODEL = "llama3"

FIELD_OUT = "description2"
MAX_WORDS = 28
FORCE_REGEN = True  # siempre regenerar

# ========= Networking (más robusto) =========
session = requests.Session()
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# ========= Helpers =========
SPANISH_WORDS_RE = re.compile(
    r"\b(el|la|los|las|un|una|unos|unas|de|del|que|y|en|por|para|con|sin|como|"
    r"figura|figuras|muestra|muestran|curvas|puntos|comparaci[oó]n|representa|"
    r"se|es|son|entre|sobre|seg[uú]n|pero)\b",
    flags=re.I,
)
SPANISH_ACCENTS_RE = re.compile(r"[ñáéíóúü]", flags=re.I)

def looks_spanish(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return bool(SPANISH_WORDS_RE.search(t) or SPANISH_ACCENTS_RE.search(t))

def normalize_slashes(p: str) -> str:
    return (p or "").replace("\\", "/")

FIG_REF_RE = re.compile(r"\b(fig\.?|figure)\s*\d+\b", flags=re.I)
def mentions_other_figures(text: str) -> bool:
    return bool(FIG_REF_RE.search((text or "")))

# Evitar frases plantilla
GENERIC_RE = re.compile(
    r"(is presented|is shown|is depicted|shown in this figure|presented in this figure|"
    r"the figure shows|this figure shows|the figure depicts|this figure depicts|"
    r"the figure illustrates|this figure illustrates|an illustration depicts|"
    r"an illustration illustrates|here is the revised output|here is the rewritten output)",
    flags=re.I,
)

def looks_generic(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if GENERIC_RE.search(t):
        return True
    if len(t.split()) < 7:
        return True
    return False

def is_one_sentence(text: str) -> bool:
    t = (text or "").strip()
    parts = [p for p in re.split(r"[.!?]+", t) if p.strip()]
    return len(parts) == 1

def word_count(text: str) -> int:
    return len(re.findall(r"\S+", (text or "").strip()))

# Limpieza de “meta texto” que mete el modelo
META_PREFIX_RE = re.compile(
    r"^(here is (the )?(revised|rewritten) output:\s*|revised output:\s*|rewritten output:\s*|output:\s*)",
    flags=re.I,
)

def clean_llm_output(txt: str) -> str:
    t = (txt or "").strip()
    t = " ".join(t.replace("\n", " ").split())
    t = re.sub(r'^(["“”])(.+)(["“”])$', r"\2", t).strip()  # comillas
    t = META_PREFIX_RE.sub("", t).strip()                  # prefijos basura
    return t

# Evitar “pipeline” si no está en la caption (alucinación típica)
def pipeline_not_in_caption(out: str, caption: str) -> bool:
    o = (out or "").lower()
    c = (caption or "").lower()
    if "pipeline" in o:
        return "pipeline" not in c
    return False

def infer_hint_from_caption(caption: str) -> str:
    c = (caption or "").lower()
    if any(k in c for k in ["taxonomy", "categorize", "categorise"]):
        return "Hint: taxonomy diagram."
    if any(k in c for k in ["chronological", "timeline", "evolution", "from 201", "to 202"]):
        return "Hint: timeline."
    if any(k in c for k in ["pipeline", "framework", "workflow", "stages"]):
        return "Hint: pipeline/workflow diagram."
    if any(k in c for k in ["error bars", "confidence interval", "95%"]):
        return "Hint: plot with error bars."
    if any(k in c for k in ["by ", "over ", "versus", "vs", "projected", "incidence", "degradation", "interactions"]):
        return "Hint: line chart or comparative plot."
    return "Hint: scientific figure."

def build_prompt(caption: str) -> str:
    hint = infer_hint_from_caption(caption)
    return f"""
Write the description in ENGLISH ONLY. Do not use Spanish.
Use ONLY the explicit information in the caption. No external knowledge.
If information is missing, say so neutrally without inventing.

Goal: Describe the FIGURE (what it is / what is plotted or structured), not just the conclusion.
Do NOT mention any other figure numbers or cross-references (e.g., "Figure 3", "as in Figure 3").
Do NOT include meta text like "Here is the revised output".
Avoid generic filler like "is presented" or "the figure shows/depicts/illustrates".

Task:
Write ONE sentence (max {MAX_WORDS} words), concrete, not starting with "This figure shows".

Return ONLY the sentence (no JSON, no extra text).

{hint}

Caption:
<<<{caption}>>>
""".strip()

def build_fix_prompt(caption: str, reason: str, prev: str) -> str:
    hint = infer_hint_from_caption(caption)
    return f"""
ENGLISH ONLY.
Fix the previous output because: {reason}

Rules:
- ONE sentence, max {MAX_WORDS} words.
- Use ONLY the caption; no external knowledge.
- Do NOT mention any figure numbers/cross-references.
- Do NOT include meta text like "Here is the revised output".
- Be specific (what is plotted/diagrammed), avoid generic phrases.

Previous output:
<<<{prev}>>>

{hint}

Caption:
<<<{caption}>>>
""".strip()

def ollama_text(prompt: str, timeout: int = 180, retries: int = 6) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 140,
    }

    last_text = ""
    for i in range(retries):
        r = session.post(URL, json=payload, headers=HEADERS, timeout=timeout)
        last_text = (r.text or "")[:300]

        # backoff en bloqueos / rate limit / errores temporales
        if r.status_code in (403, 429) or 500 <= r.status_code <= 599:
            wait = 2.5 * (i + 1)
            print(f"[WARN] HTTP {r.status_code} -> retry in {wait:.1f}s")
            time.sleep(wait)
            continue

        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return clean_llm_output(txt)

    raise RuntimeError(f"Request failed after retries. Last status={r.status_code}, body={last_text}")

def generate_good_desc(caption: str, max_tries: int = 3) -> str:
    prompt = build_prompt(caption)
    prev = ""
    for _ in range(max_tries):
        out = ollama_text(prompt)

        if not out.strip():
            prev, prompt = out, build_fix_prompt(caption, "Empty output.", out)
            time.sleep(0.4); continue

        if looks_spanish(out):
            prev, prompt = out, build_fix_prompt(caption, "Output contains Spanish.", out)
            time.sleep(0.4); continue

        if mentions_other_figures(out):
            prev, prompt = out, build_fix_prompt(caption, "Output mentions figure numbers/cross-references.", out)
            time.sleep(0.4); continue

        if pipeline_not_in_caption(out, caption):
            prev, prompt = out, build_fix_prompt(caption, "Output mentions 'pipeline' but caption does not.", out)
            time.sleep(0.4); continue

        if not is_one_sentence(out):
            prev, prompt = out, build_fix_prompt(caption, "Output is not a single sentence.", out)
            time.sleep(0.4); continue

        if word_count(out) > MAX_WORDS:
            prev, prompt = out, build_fix_prompt(caption, f"Output has more than {MAX_WORDS} words.", out)
            time.sleep(0.4); continue

        if looks_generic(out):
            prev, prompt = out, build_fix_prompt(caption, "Output is too generic or contains meta text.", out)
            time.sleep(0.4); continue

        return out

    return prev or ""

def main():
    manifests = sorted(BASE_DIR.glob("*/manifest.json"))
    if not manifests:
        print("[WARN] No se encontraron manifests en", BASE_DIR)
        return

    total = 0

    for mpath in manifests:
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ERR] No pude leer {mpath}: {e}")
            continue

        changed = False

        for item in data:
            # normaliza rutas
            if "source_pdf" in item:
                item["source_pdf"] = normalize_slashes(item["source_pdf"])
            if "path_rel" in item:
                item["path_rel"] = normalize_slashes(item["path_rel"])

            caption = (item.get("caption") or "").strip()
            if not caption:
                item[FIELD_OUT] = ""
                changed = True
                continue

            try:
                desc = generate_good_desc(caption, max_tries=3)
                item[FIELD_OUT] = desc  # sobrescribe siempre
                changed = True
                total += 1
                print(f"[REGEN] {item.get('doc_id')} p{item.get('page')} id{item.get('id')}")
                time.sleep(0.35)  # un poco más lento = menos bloqueos
            except Exception as e:
                print(f"[ERR] {mpath.name} id{item.get('id')}: {e}")

        if changed:
            mpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[SAVE] {mpath}")

    print(f"\n[FIN] Regeneradas total: {total}")

if __name__ == "__main__":
    main()
