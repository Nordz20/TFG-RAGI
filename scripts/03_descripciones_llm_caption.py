# gen_description3.py
# - Lee cada manifest.json en ./data/* (fallback: ./extract_figures_only_caption/*)
# - Para cada item, carga el crop (imagen) + caption y genera description3 usando Ollama UPM (OpenAI-compatible)
# - Guarda de vuelta el manifest.json (sobrescribe description3 siempre)
#
# Mejoras:
# - Detecta nº de paneles (1/2/3/4) mirando la imagen (gaps blancos).
# - Si hay multi-panel, fuerza "Two-panel/Three-panel..." al inicio.
# - Traza modelo y si usó imagen.

import base64
import json
import re
import time
from pathlib import Path

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

# ✅ Gemma3 es el único claro multimodal en tu lista
MODEL_VISION_PRIMARY = "gemma3:12b"
MODEL_VISION_SECONDARY = "gemma3:4b"
MODEL_TEXT_FALLBACK = "qwen3:8b"

FIELD_OUT = "description3"
MAX_WORDS = 28
SLEEP_BETWEEN = 0.35
FORCE_REGEN = True  # siempre sobrescribe description3

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

# ================== TEXT UTILS ==================
META_PREFIX_RE = re.compile(
    r"^(here is (the )?(revised|rewritten) output:\s*|revised output:\s*|rewritten output:\s*|output:\s*)",
    flags=re.I,
)

def clean_llm_output(txt: str) -> str:
    t = (txt or "").strip()
    t = " ".join(t.replace("\n", " ").split())
    t = re.sub(r'^(["“”])(.+)(["“”])$', r"\2", t).strip()
    t = META_PREFIX_RE.sub("", t).strip()
    return t

def word_count(text: str) -> int:
    return len(re.findall(r"\S+", (text or "").strip()))

def is_one_sentence(text: str) -> bool:
    t = (text or "").strip()
    parts = [p for p in re.split(r"[.!?]+", t) if p.strip()]
    return len(parts) == 1

# Evita referencias a figuras por número
FIG_REF_RE = re.compile(r"\b(fig\.?|figure)\s*\d+\b", flags=re.I)
def mentions_other_figures(text: str) -> bool:
    return bool(FIG_REF_RE.search(text or ""))

# Evita frases relleno
GENERIC_RE = re.compile(
    r"(is presented|is shown|is depicted|shown in this figure|presented in this figure|"
    r"the figure shows|this figure shows|the figure depicts|this figure depicts|"
    r"the figure illustrates|this figure illustrates)",
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

# Si multi-panel, evitamos "single axis/single plot"
BAD_SINGLE_RE = re.compile(r"\b(single axis|single plot|single panel)\b", flags=re.I)
def contradicts_panels(text: str, panels: int) -> bool:
    if panels and panels > 1:
        return bool(BAD_SINGLE_RE.search(text or ""))
    return False

# ================== IMAGE UTILS ==================
def to_data_url(img_path: Path) -> str:
    b = img_path.read_bytes()
    ext = img_path.suffix.lower().lstrip(".")
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "png"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(b).decode('utf-8')}"

def resolve_image_path(item: dict, manifest_path: Path) -> Path | None:
    # 0) image_path (formato actual del manifest)
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

    # 1) path_rel si existe
    pr = item.get("path_rel")
    if pr:
        p = Path(pr)
        if p.exists():
            return p
        try2 = Path(".") / p
        if try2.exists():
            return try2
        try3 = Path(str(p).replace("\\", "/"))
        if try3.exists():
            return try3

    # 2) doc_id + file
    doc_id = item.get("doc_id")
    fname = item.get("file")
    if doc_id and fname:
        p_crops = manifest_path.parent / "crops" / fname
        if p_crops.exists():
            return p_crops
        p_images = manifest_path.parent / "images" / fname
        if p_images.exists():
            return p_images

    # 3) relativo al manifest (misma carpeta)
    if fname:
        p = manifest_path.parent / "crops" / fname
        if p.exists():
            return p

    return None

# -------- Panel detection (sin OCR) --------
# Heurística: busca "gaps" verticales/horizontales muy blancos (separaciones entre subplots).
def estimate_panels(img_path: Path) -> int:
    try:
        from PIL import Image
    except Exception:
        return 1

    try:
        im = Image.open(img_path).convert("L")
        # downscale para acelerar
        max_w = 800
        if im.width > max_w:
            new_h = int(im.height * (max_w / im.width))
            im = im.resize((max_w, max(1, new_h)))
        w, h = im.size
        pix = list(im.getdata())  # len = w*h

        # helpers para ratio blanco por columna/fila
        def col_white_ratio(x: int) -> float:
            # sample todos los y
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

        # compute ratios con muestreo (cada 2 px) para acelerar
        step = 2 if w > 500 else 1
        col = [col_white_ratio(x) for x in range(0, w, step)]
        row = [row_white_ratio(y) for y in range(0, h, step)]

        def count_gaps(profile, total_len):
            # gap = tramo largo con ratio altísimo (casi todo blanco)
            thr = 0.995
            min_run = max(6, int(len(profile) * 0.02))  # ~2% del ancho/alto
            # ignorar bordes (márgenes)
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

        v_gaps = count_gaps(col, w)  # separaciones verticales -> paneles en horizontal
        h_gaps = count_gaps(row, h)  # separaciones horizontales -> paneles en vertical

        panels_h = min(4, v_gaps + 1) if v_gaps > 0 else 1
        panels_v = min(4, h_gaps + 1) if h_gaps > 0 else 1

        panels = max(panels_h, panels_v)
        return max(1, min(4, panels))
    except Exception:
        return 1

def panel_prefix(panels: int) -> str:
    return {2: "Two-panel", 3: "Three-panel", 4: "Four-panel"}.get(panels, "")

# ================== PROMPTS ==================
def build_prompt(caption: str, panels: int, fig_type: str | None) -> str:
    pfx = panel_prefix(panels)
    panel_rule = ""
    if panels and panels > 1 and pfx:
        panel_rule = f"- The image has {panels} panels; START the sentence with '{pfx}'.\n"

    type_hint = f"Figure type hint: {fig_type}.\n" if fig_type else ""

    return f"""
Write the description in ENGLISH ONLY.
Use BOTH the image and the caption, but DO NOT invent details you cannot see.
Be concrete about the figure structure (e.g., line chart, multi-panel plot, flowchart, taxonomy).
If axis labels or small text are not readable, describe them generically (do not guess).
{type_hint}
Rules:
{panel_rule}- ONE sentence, max {MAX_WORDS} words.
- Do NOT start with "This figure shows".
- Do NOT mention other figure numbers (e.g., "Figure 3").
- Avoid generic filler like "is presented/depicted/illustrates".
Return ONLY the sentence.

Caption:
<<<{caption}>>>
""".strip()

def build_fix_prompt(caption: str, panels: int, fig_type: str | None, reason: str, prev: str) -> str:
    pfx = panel_prefix(panels)
    panel_rule = ""
    if panels and panels > 1 and pfx:
        panel_rule = f"- The image has {panels} panels; START the sentence with '{pfx}'.\n"

    type_hint = f"Figure type hint: {fig_type}.\n" if fig_type else ""

    return f"""
ENGLISH ONLY.
Rewrite because: {reason}

Use image+caption, but no guessing.
{type_hint}
Rules:
{panel_rule}- ONE sentence, max {MAX_WORDS} words.
- No cross-references to other figures.
- Avoid generic filler ("is presented/depicted/illustrates") and avoid starting with "This figure shows".
Return ONLY the sentence.

Previous output:
<<<{prev}>>>

Caption:
<<<{caption}>>>
""".strip()

# ================== LLM CALL ==================
def _post_chat(payload: dict, timeout: int):
    r = session.post(URL, json=payload, headers=HEADERS, timeout=timeout)
    if not r.ok:
        # intenta imprimir error útil
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text[:500]}
        raise requests.HTTPError(f"HTTP {r.status_code}: {err}", response=r)
    return r

def call_ollama(prompt: str, image_path: Path | None, model: str, timeout: int = 180) -> str:
    if image_path is not None:
        data_url = to_data_url(image_path)

        # intento A (objeto con url)
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
            "max_tokens": 160,
        }

        try:
            r = _post_chat(payload_a, timeout)
            out = r.json()["choices"][0]["message"]["content"]
            return clean_llm_output(out)
        except requests.HTTPError as e:
            # intento B (image_url como string) por compatibilidad
            resp = getattr(e, "response", None)
            sc = resp.status_code if resp is not None else None
            if sc in (400, 422):
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
                    "max_tokens": 160,
                }
                r = _post_chat(payload_b, timeout)
                out = r.json()["choices"][0]["message"]["content"]
                return clean_llm_output(out)
            raise

    # texto-only
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 160,
    }
    r = _post_chat(payload, timeout)
    out = r.json()["choices"][0]["message"]["content"]
    return clean_llm_output(out)

def generate_desc3(caption: str, img_path: Path | None, fig_type: str | None, tries: int = 3):
    panels = estimate_panels(img_path) if img_path else 1
    prompt = build_prompt(caption, panels, fig_type)
    prev = ""
    used_model = ""
    used_image = False

    for _ in range(tries):
        out = ""

        # 1) Vision primary
        if img_path is not None:
            try:
                out = call_ollama(prompt, img_path, MODEL_VISION_PRIMARY)
                used_model = MODEL_VISION_PRIMARY
                used_image = True
            except Exception:
                out = ""

        # 2) Vision secondary
        if (not out) and img_path is not None:
            try:
                out = call_ollama(prompt, img_path, MODEL_VISION_SECONDARY)
                used_model = MODEL_VISION_SECONDARY
                used_image = True
            except Exception:
                out = ""

        # 3) Text fallback
        if not out:
            out = call_ollama(prompt, None, MODEL_TEXT_FALLBACK)
            used_model = MODEL_TEXT_FALLBACK
            used_image = False

        # Validaciones
        if not out.strip():
            prev, prompt = out, build_fix_prompt(caption, panels, fig_type, "Empty output.", out)
            time.sleep(0.4)
            continue

        if mentions_other_figures(out):
            prev, prompt = out, build_fix_prompt(caption, panels, fig_type, "Mentions other figure numbers.", out)
            time.sleep(0.4)
            continue

        if not is_one_sentence(out):
            prev, prompt = out, build_fix_prompt(caption, panels, fig_type, "Not a single sentence.", out)
            time.sleep(0.4)
            continue

        if word_count(out) > MAX_WORDS:
            prev, prompt = out, build_fix_prompt(caption, panels, fig_type, f"More than {MAX_WORDS} words.", out)
            time.sleep(0.4)
            continue

        if looks_generic(out):
            prev, prompt = out, build_fix_prompt(caption, panels, fig_type, "Too generic / filler phrasing.", out)
            time.sleep(0.4)
            continue

        if contradicts_panels(out, panels):
            prev, prompt = out, build_fix_prompt(caption, panels, fig_type, "Contradicts multi-panel layout (e.g., says single plot/axis).", out)
            time.sleep(0.4)
            continue

        # Si hay multi-panel, aseguramos prefijo (si el modelo no obedeció)
        pfx = panel_prefix(panels)
        if panels > 1 and pfx and not out.lower().startswith(pfx.lower()):
            # NO añadimos otra frase: solo prefijamos la misma oración
            out = f"{pfx} {out[0].lower() + out[1:]}" if out else out

        return out, used_model, used_image, panels

    return (prev or ""), used_model, used_image, panels

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
            removed_model = item.pop("description3_model", None)
            removed_used_image = item.pop("description3_used_image", None)
            removed_panels = item.pop("description3_panels", None)
            if (removed_model is not None) or (removed_used_image is not None) or (removed_panels is not None):
                changed = True

            caption = (item.get("caption") or "").strip()
            if not caption:
                item[FIELD_OUT] = ""
                changed = True
                continue

            if not FORCE_REGEN and (item.get(FIELD_OUT) or "").strip():
                continue

            img_path = resolve_image_path(item, mpath)
            if img_path is None:
                missing_imgs += 1

            fig_type = (item.get("figure_type") or "").strip() or None

            try:
                desc3, _, _, panels = generate_desc3(caption, img_path, fig_type, tries=3)
                item[FIELD_OUT] = desc3

                changed = True
                total += 1
                print(f"[OK] {item.get('doc_id')} p{item.get('page')} id{item.get('id')} -> {FIELD_OUT} | panels={panels}")
                time.sleep(SLEEP_BETWEEN)
            except Exception as e:
                print(f"[ERR] {mpath.name} id{item.get('id')}: {e}")

        if changed:
            mpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[SAVE] {mpath}")

    print(f"\n[FIN] Updated: {total} | Missing images: {missing_imgs}")

if __name__ == "__main__":
    main()
