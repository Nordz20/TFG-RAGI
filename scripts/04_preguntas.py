# scripts/gen_questions.py
# - Lee cada manifest.json en ./extract_figures_only_caption/*
# - Para cada item, usa crop (imagen) + caption y genera EXACTAMENTE 2 preguntas (questions)
# - Guarda de vuelta el manifest.json (sobrescribe questions siempre)
#
# Mejoras:
# - Preguntas estilo "buscador": Q1=concepto/relación, Q2=detalle discriminativo.
# - Evita preguntas ambiguas/no contestables (meet/intersect/around YEAR/etc).
# - Evita mencionar IDs de modelos si no están en caption.
# - Detecta nº de paneles (1-4) sin OCR y lo usa como restricción.
# - Preflight (1 vez) para saber si gemma3 acepta imágenes -> menos reintentos.
# - Cache base64 y resize opcional para acelerar.

import base64
import json
import re
import time
from pathlib import Path
from difflib import SequenceMatcher

import requests

# ================== CONFIG ==================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR_CANDIDATES = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "extract_figures_only_caption",
    SCRIPT_DIR / "data",
    SCRIPT_DIR / "extract_figures_only_caption",
]
URL = "https://wiig.dia.fi.upm.es/ollama/v1/chat/completions"

MODEL_VISION_PRIMARY = "gemma3:12b"
MODEL_VISION_SECONDARY = "gemma3:4b"
MODEL_TEXT_FALLBACK = "qwen3:8b"  # caption-only si falla imagen

FIELD_OUT = "questions"  # list[str] tamaño 2

SLEEP_BETWEEN = 0.20
FORCE_REGEN = True
TRIES = 3

MAX_WORDS_Q = 18  # pregunta corta (query-like)

# Acelerar payload: resize opcional (si Pillow está disponible)
RESIZE_MAX_W = 1024  # pon None para desactivar

def resolve_base_dir() -> Path | None:
    for candidate in BASE_DIR_CANDIDATES:
        if candidate.exists() and any(candidate.glob("*/manifest.json")):
            return candidate
    return None

# ================== HTTP ==================
session = requests.Session()
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# ================== REGEX / VALIDATORS ==================
FIG_REF_RE = re.compile(r"\b(fig\.?|figure)\s*\d+\b", flags=re.I)

GENERIC_Q_RE = re.compile(
    r"(what does (this|the) figure show|what is shown|what is depicted|"
    r"describe (this|the) figure|summarize (this|the) figure|"
    r"what can be inferred in general|give an overview|provide an overview)",
    flags=re.I,
)

# Preguntas típicamente malas/ambiguas para retrieval
BANNED_PHRASES_RE = re.compile(
    r"(meet|intersect|cross|where do .* meet|where do .* intersect|"
    r"around\s+\d{4}|approximately\s+\d{4}|at\s+around\s+\d{4}|"
    r"faster initial increase|faster increase|steeper at the start|"
    r"what date is indicated|exact value|exactly how many|read the value)",
    flags=re.I,
)

# Evitar mencionar IDs concretos si el caption no los trae (reduce alucinación)
MODEL_TOKEN_RE = re.compile(
    r"\b(gpt[\-_]?\d(\.\d+)?|davinci[\-_]?\d*|claude[\-_]?\d|grok[\-_]?\d|o\d)\b",
    flags=re.I,
)

BAD_MULTI_RE = re.compile(r"\b(multi-?panel|two-?panel|three-?panel|four-?panel)\b", flags=re.I)

# Stopwords simples para anclaje semántico
STOP = set("""
a an the and or of to in on for with by from into over under between across as at is are was were be being been
this that these those it its their his her our your
show shows shown depict depicts depicted illustrate illustrates illustrated present presents presented
figure panel panels plot chart graph
""".split())

def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = " ".join(s.replace("\n", " ").split())
    s = re.sub(r'^(["“”])(.+)(["“”])$', r"\2", s).strip()
    return s

def word_count(s: str) -> int:
    return len(re.findall(r"\S+", (s or "").strip()))

def too_similar(a: str, b: str) -> bool:
    a2, b2 = (a or "").lower(), (b or "").lower()
    return SequenceMatcher(None, a2, b2).ratio() >= 0.82

def extract_anchor_terms(caption: str, k: int = 10) -> set[str]:
    # saca palabras útiles del caption (muy simple) para que las preguntas queden "anchored"
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", (caption or "").lower())
    cand = [w for w in words if w not in STOP]
    freq = {}
    for w in cand:
        freq[w] = freq.get(w, 0) + 1
    # prioriza por frecuencia y longitud
    ranked = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    return set([w for w, _ in ranked[:k]])

def has_anchor(q: str, anchors: set[str]) -> bool:
    ql = (q or "").lower()
    return any(a in ql for a in anchors) or ("curve" in ql) or ("line" in ql) or ("threshold" in ql) or ("inflection" in ql)

def normalize_question(q: str) -> str:
    q = clean_text(q)
    if not q:
        return ""
    q = q[:-1].strip() if q.endswith("?") else q
    q = q[0].upper() + q[1:] if len(q) > 1 else q.upper()
    return q + "?"

def fallback_general_questions(caption: str, fig_type: str | None) -> list[str]:
    cap = (caption or "").strip().lower()
    ftype = (fig_type or "").strip().lower()

    if any(k in cap for k in ["timeline", "over time", "evolution", "trend"]):
        q1 = "What main topic or process is represented over time in the figure?"
        q2 = "What overall trend or progression is shown across the figure?"
    elif any(k in cap or k in ftype for k in ["taxonomy", "category", "categories"]):
        q1 = "What main concept is organized in the figure?"
        q2 = "How are the main categories structured in the figure?"
    elif any(k in cap or k in ftype for k in ["pipeline", "workflow", "framework", "architecture"]):
        q1 = "What main objective or task does the figure describe?"
        q2 = "How is the overall pipeline or architecture organized in the figure?"
    else:
        q1 = "What main idea or comparison does the figure present?"
        q2 = "What overall structure or pattern is shown in the figure?"

    q1 = normalize_question(q1)
    q2 = normalize_question(q2)
    if too_similar(q1, q2):
        q2 = "What broad structure is used to organize the information in the figure?"
        q2 = normalize_question(q2)
    return [q1, q2]

def bad_question(q: str, caption: str, panels: int) -> bool:
    t = (q or "").strip()
    if not t:
        return True
    if not t.endswith("?"):
        return True
    if word_count(t) < 6:
        return True
    if word_count(t) > MAX_WORDS_Q:
        return True
    if FIG_REF_RE.search(t):
        return True
    if GENERIC_Q_RE.search(t.lower()):
        return True
    if BANNED_PHRASES_RE.search(t.lower()):
        return True

    # Si el caption NO menciona modelos concretos, no permitas tokens tipo gpt_4, davinci, etc.
    # (Así evitamos alucinaciones del tipo "gpt_4 and davinci_002 curves meet")
    if MODEL_TOKEN_RE.search(t.lower()):
        cap = (caption or "").lower()
        if not MODEL_TOKEN_RE.search(cap):
            return True

    # Si el detector dice 1 panel, no aceptes preguntas que hablen de multi-panel explícitamente
    if panels == 1 and BAD_MULTI_RE.search(t.lower()):
        return True

    return False

# ================== IMAGE PATH ==================
B64_CACHE: dict[str, str] = {}

def _read_and_resize_bytes(img_path: Path) -> bytes:
    b = img_path.read_bytes()
    if RESIZE_MAX_W is None:
        return b
    try:
        from PIL import Image
        import io
        im = Image.open(img_path)
        if im.width <= RESIZE_MAX_W:
            return b
        new_h = int(im.height * (RESIZE_MAX_W / im.width))
        im = im.convert("RGB").resize((RESIZE_MAX_W, max(1, new_h)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()
    except Exception:
        # si Pillow no está o falla, usa bytes originales
        return b

def to_data_url_cached(img_path: Path) -> str:
    key = str(img_path.resolve())
    if key in B64_CACHE:
        return B64_CACHE[key]
    b = _read_and_resize_bytes(img_path)
    # si hemos re-encodeado a JPG en resize, mime jpeg; si no, usa extensión
    ext = img_path.suffix.lower().lstrip(".")
    mime = "jpeg" if (RESIZE_MAX_W is not None and len(b) != img_path.stat().st_size) else ("jpeg" if ext == "jpg" else ext)
    if mime not in ("jpeg", "png", "webp"):
        mime = "png"
    data_url = f"data:image/{mime};base64,{base64.b64encode(b).decode('utf-8')}"
    B64_CACHE[key] = data_url
    return data_url

def resolve_image_path(item: dict, manifest_path: Path) -> Path | None:
    image_path = item.get("image_path")
    if image_path:
        p = Path(image_path)
        if p.exists():
            return p
        p2 = (manifest_path.parent / image_path).resolve()
        if p2.exists():
            return p2
        p3 = (PROJECT_ROOT / image_path).resolve()
        if p3.exists():
            return p3

    pr = item.get("path_rel")
    if pr:
        p = Path(pr)
        if p.exists():
            return p
        p2 = Path(".") / p
        if p2.exists():
            return p2

    doc_id = item.get("doc_id")
    fname = item.get("file")
    if doc_id and fname:
        p_crops = manifest_path.parent / "crops" / fname
        if p_crops.exists():
            return p_crops
        p_images = manifest_path.parent / "images" / fname
        if p_images.exists():
            return p_images

    if fname:
        p = manifest_path.parent / "crops" / fname
        if p.exists():
            return p

    return None

# ================== PANEL DETECTION (sin OCR) ==================
def estimate_panels(img_path: Path | None) -> int:
    if img_path is None:
        return 1
    try:
        from PIL import Image
        im = Image.open(img_path).convert("L")
        max_w = 800
        if im.width > max_w:
            new_h = int(im.height * (max_w / im.width))
            im = im.resize((max_w, max(1, new_h)))
        w, h = im.size
        pix = list(im.getdata())

        def col_white_ratio(x: int) -> float:
            white = 0
            base = x
            for y in range(h):
                if pix[base + y * w] >= 245:
                    white += 1
            return white / h

        def row_white_ratio(y: int) -> float:
            white = 0
            base = y * w
            for x in range(w):
                if pix[base + x] >= 245:
                    white += 1
            return white / w

        step = 2 if w > 500 else 1
        col = [col_white_ratio(x) for x in range(0, w, step)]
        row = [row_white_ratio(y) for y in range(0, h, step)]

        def count_gaps(profile):
            thr = 0.995
            min_run = max(6, int(len(profile) * 0.02))
            left = int(len(profile) * 0.03)
            right = int(len(profile) * 0.97)

            gaps = 0
            run = 0
            in_gap = False
            for i, v in enumerate(profile):
                if i < left or i > right:
                    continue
                if v >= thr:
                    run += 1
                    if (not in_gap) and run >= min_run:
                        gaps += 1
                        in_gap = True
                else:
                    run = 0
                    in_gap = False
            return gaps

        v_gaps = count_gaps(col)
        h_gaps = count_gaps(row)
        panels_h = min(4, v_gaps + 1) if v_gaps > 0 else 1
        panels_v = min(4, h_gaps + 1) if h_gaps > 0 else 1
        return max(1, min(4, max(panels_h, panels_v)))
    except Exception:
        return 1

# ================== PROMPTS ==================
def build_prompt(caption: str, fig_type: str | None, panels: int, anchors: set[str]) -> str:
    type_hint = f"Figure type hint: {fig_type}.\n" if fig_type else ""
    panel_hint = f"The figure appears to have {panels} panel(s).\n"
    anchor_hint = ""
    if anchors:
        anchor_hint = "Anchor terms from caption: " + ", ".join(sorted(list(anchors))[:8]) + ".\n"

    return f"""
You are given a scientific figure (image) and its caption.
Create EXACTLY TWO user-like search questions that this figure can answer.

Output MUST be valid JSON: {{"q1":"...?", "q2":"...?"}} and nothing else.

Question roles:
- q1 (General scope): ask a broad question about the main topic or objective.
- q2 (General structure): ask a broad question about the overall structure, organization, or trend.

Constraints:
- ENGLISH ONLY.
- Each question must be answerable from the figure + caption without reading tiny text.
- Keep questions as general and high-level as possible.
- Avoid exact-value reading (no "what exact date/value is indicated").
- Avoid vague time references like "around 2023".
- Do NOT ask about curve intersections/meeting/crossing.
- Do NOT mention figure numbers (no "Figure 2"/"Fig.").
- If the caption does NOT mention specific model names (e.g., GPT-4), do not mention them in the questions.
- Keep each question <= {MAX_WORDS_Q} words.
- Do NOT say "multi-panel/two-panel/three-panel" unless the figure clearly has >1 panel (here: {panels}).

{type_hint}{panel_hint}{anchor_hint}
Caption:
<<<{caption}>>>
""".strip()

def build_fix_prompt(caption: str, fig_type: str | None, panels: int, anchors: set[str], reason: str, prev: str) -> str:
    type_hint = f"Figure type hint: {fig_type}.\n" if fig_type else ""
    anchor_hint = ""
    if anchors:
        anchor_hint = "Anchor terms from caption: " + ", ".join(sorted(list(anchors))[:8]) + ".\n"

    return f"""
Rewrite because: {reason}

Return ONLY valid JSON: {{"q1":"...?", "q2":"...?"}}
- EXACTLY two questions.
- ENGLISH ONLY.
- q1 = General scope, q2 = General structure.
- Keep both questions high-level and general.
- Answerable from figure+caption without reading tiny text.
- No exact-value/date reading; no vague "around YEAR"; no intersections/meeting.
- No figure-number references.
- Keep each <= {MAX_WORDS_Q} words.
- Do NOT say multi-panel/two-panel/etc if panels=1 (here panels={panels}).

{type_hint}{anchor_hint}
Previous output:
<<<{prev}>>>

Caption:
<<<{caption}>>>
""".strip()

# ================== OLLAMA CALL ==================
def _post(payload: dict, timeout: int):
    r = session.post(URL, json=payload, headers=HEADERS, timeout=timeout)
    if not r.ok:
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text[:500]}
        raise requests.HTTPError(f"HTTP {r.status_code}: {err}", response=r)
    return r

def call_ollama(prompt: str, image_path: Path | None, model: str, timeout: int = 120) -> str:
    if image_path is not None:
        data_url = to_data_url_cached(image_path)

        payload_a = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 220,
        }
        try:
            r = _post(payload_a, timeout)
            return clean_text(r.json()["choices"][0]["message"]["content"])
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code in (400, 422):
                payload_b = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": data_url},
                        ],
                    }],
                    "stream": False,
                    "temperature": 0.2,
                    "max_tokens": 220,
                }
                r = _post(payload_b, timeout)
                return clean_text(r.json()["choices"][0]["message"]["content"])
            raise

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 220,
    }
    r = _post(payload, timeout)
    return clean_text(r.json()["choices"][0]["message"]["content"])

# ================== PARSE + VALIDATE ==================
def parse_questions_json(txt: str) -> tuple[str, str] | None:
    try:
        obj = json.loads(txt)
        q1 = clean_text(obj.get("q1", ""))
        q2 = clean_text(obj.get("q2", ""))
        return q1, q2
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            q1 = clean_text(obj.get("q1", ""))
            q2 = clean_text(obj.get("q2", ""))
            return q1, q2
        except Exception:
            return None

def finalize_questions(q1: str, q2: str) -> tuple[str, str]:
    q1 = q1 if q1.endswith("?") else (q1 + "?")
    q2 = q2 if q2.endswith("?") else (q2 + "?")
    return q1, q2

def supports_images_once(model: str, test_img: Path) -> bool:
    # preflight rápido para no intentar imagen+modelo por cada item si el servidor no lo soporta
    prompt = 'Return ONLY JSON: {"q1":"What is compared?","q2":"What trend is visible?"}'
    try:
        _ = call_ollama(prompt, test_img, model, timeout=25)
        return True
    except Exception:
        return False

def generate_questions(caption: str, img_path: Path | None, fig_type: str | None,
                       img_ok_primary: bool, img_ok_secondary: bool,
                       tries: int = TRIES):
    panels = estimate_panels(img_path)
    anchors = extract_anchor_terms(caption, k=10)

    prompt = build_prompt(caption, fig_type, panels, anchors)
    used_model = ""
    used_image = False

    for _ in range(tries):
        out = ""

        # Vision primary si soporta imágenes
        if img_path is not None and img_ok_primary:
            try:
                out = call_ollama(prompt, img_path, MODEL_VISION_PRIMARY)
                used_model = MODEL_VISION_PRIMARY
                used_image = True
            except Exception:
                out = ""

        # Vision secondary si soporta imágenes
        if (not out) and img_path is not None and img_ok_secondary:
            try:
                out = call_ollama(prompt, img_path, MODEL_VISION_SECONDARY)
                used_model = MODEL_VISION_SECONDARY
                used_image = True
            except Exception:
                out = ""

        # Text fallback
        if not out:
            out = call_ollama(prompt, None, MODEL_TEXT_FALLBACK)
            used_model = MODEL_TEXT_FALLBACK
            used_image = False

        parsed = parse_questions_json(out)
        if not parsed:
            prompt = build_fix_prompt(caption, fig_type, panels, anchors, "Output was not valid JSON with q1/q2.", out)
            time.sleep(0.25)
            continue

        q1, q2 = finalize_questions(*parsed)

        # Validaciones duras
        if bad_question(q1, caption, panels) or bad_question(q2, caption, panels):
            prompt = build_fix_prompt(caption, fig_type, panels, anchors, "One or both questions were invalid/generic/ambiguous.", out)
            time.sleep(0.25)
            continue

        if too_similar(q1, q2):
            prompt = build_fix_prompt(caption, fig_type, panels, anchors, "The two questions are too similar; make them distinct.", out)
            time.sleep(0.25)
            continue

        # Anclaje semántico: al menos una palabra “de caption” o concepto fuerte
        if not has_anchor(q1, anchors) or not has_anchor(q2, anchors):
            prompt = build_fix_prompt(caption, fig_type, panels, anchors, "Questions were not anchored to caption concepts; make them more specific.", out)
            time.sleep(0.25)
            continue

        q1 = normalize_question(q1)
        q2 = normalize_question(q2)
        if not q1 or not q2:
            prompt = build_fix_prompt(caption, fig_type, panels, anchors, "Questions were empty after normalization.", out)
            time.sleep(0.25)
            continue

        return [q1, q2], used_model, used_image

    return fallback_general_questions(caption, fig_type), used_model, used_image

# ================== MAIN ==================
def main():
    base_dir = resolve_base_dir()
    if base_dir is None:
        print("[WARN] No manifests found. Checked:")
        for p in BASE_DIR_CANDIDATES:
            print("  -", p)
        return

    manifests = sorted(base_dir.glob("*/manifest.json"))
    if not manifests:
        print("[WARN] No manifests found in", base_dir)
        return

    print("[INFO] Using manifests in", base_dir)

    # Preflight: elige una imagen cualquiera para probar soporte multimodal
    test_imgs = (
        list(base_dir.glob("*/images/*.png"))
        + list(base_dir.glob("*/images/*.jpg"))
        + list(base_dir.glob("*/images/*.jpeg"))
        + list(base_dir.glob("*/crops/*.png"))
        + list(base_dir.glob("*/crops/*.jpg"))
        + list(base_dir.glob("*/crops/*.jpeg"))
    )
    test_img = test_imgs[0] if test_imgs else None

    img_ok_primary = False
    img_ok_secondary = False
    if test_img is not None:
        img_ok_primary = supports_images_once(MODEL_VISION_PRIMARY, test_img)
        img_ok_secondary = supports_images_once(MODEL_VISION_SECONDARY, test_img) if not img_ok_primary else True

    print(f"[PRE] image support: {MODEL_VISION_PRIMARY}={img_ok_primary} | {MODEL_VISION_SECONDARY}={img_ok_secondary} | resize_max_w={RESIZE_MAX_W}")

    total = 0
    missing_imgs = 0

    for mpath in manifests:
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ERR] Cannot read {mpath}: {e}")
            continue

        changed = False

        for item in data:
            removed_model = item.pop("questions_model", None)
            removed_used_image = item.pop("questions_used_image", None)
            if (removed_model is not None) or (removed_used_image is not None):
                changed = True

            caption = (item.get("caption") or "").strip()
            if not caption:
                item[FIELD_OUT] = [
                    "What main idea does the figure present?",
                    "What overall structure or pattern is shown in the figure?",
                ]
                changed = True
                continue

            if not FORCE_REGEN and isinstance(item.get(FIELD_OUT), list) and len(item[FIELD_OUT]) == 2 and all(x.strip() for x in item[FIELD_OUT]):
                continue

            img_path = resolve_image_path(item, mpath)
            if img_path is None:
                missing_imgs += 1

            fig_type = (item.get("figure_type") or "").strip() or None

            try:
                qs, _, _ = generate_questions(
                    caption, img_path, fig_type,
                    img_ok_primary=img_ok_primary,
                    img_ok_secondary=img_ok_secondary,
                    tries=TRIES
                )
                item[FIELD_OUT] = qs
                if not isinstance(item[FIELD_OUT], list) or len(item[FIELD_OUT]) != 2 or not all((x or "").strip() for x in item[FIELD_OUT]):
                    item[FIELD_OUT] = fallback_general_questions(caption, fig_type)

                changed = True
                total += 1
                print(f"[OK] {item.get('doc_id')} p{item.get('page')} id{item.get('id')} -> questions")
                time.sleep(SLEEP_BETWEEN)
            except Exception as e:
                print(f"[ERR] {mpath.name} id{item.get('id')}: {e}")

        if changed:
            mpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[SAVE] {mpath}")

    print(f"\n[FIN] Updated: {total} | Missing images: {missing_imgs}")

if __name__ == "__main__":
    main()
