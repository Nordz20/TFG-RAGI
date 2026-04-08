import os, json, math, hashlib, re, argparse, tempfile, shutil
import fitz
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------- CONFIG ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PDFS_FOLDER = os.path.join(PROJECT_ROOT, "pdfs")
OUT_BASE_DIR  = os.path.join(PROJECT_ROOT, "data")
    
DPI   = 250
IMGSZ = 1280
CONF  = 0.15
PAD_PX = 4

# Regex estricta pero que acepta letras
CAPTION_FIG_RE = re.compile(r'^\s*(?:Figure|Fig\.?|Figura)\s*[A-Za-z0-9]+[\.\:\-]\s+', re.IGNORECASE)
CAPTION_TAB_RE = re.compile(r'^\s*(?:Table|Tabla)\b', re.IGNORECASE)
MAX_CAPTION_WORDS = 150

# Caption match
MIN_HOVER_FIG = 0.35
MIN_HOVER_CAP = 0.60
MAX_GAP_MAIN_CAPTION_PX = 450
FALLBACK_HOVER_FIG = 0.15
FALLBACK_HOVER_CAP = 0.35

DEDUP_IOU_THR = 0.80
DROP_CONTAINED_IOS = 0.90

ENABLE_LABEL_MERGE = True
LABEL_NEAR_GAP_PX = 18
MIN_LABEL_VOVER = 0.65
MAX_LABEL_WORDS = 10
MAX_LABEL_W_FRAC = 0.25
MAX_LABEL_H_FRAC = 0.30
MAX_LABEL_BLOCKS = 3

INCLUDE_INTERMEDIATE_SUBCAPTIONS = True
MAX_INTER_WORDS = 60
MAX_INTER_BOX_H_PX = 140
MIN_HOVER_INTER = 0.15
MAX_INTER_BOXES = 8

SAFE_CONTENT_EXPAND = True
WHITE_THR = 245

TOP_MAX_EXPAND_PX   = 1200
LEFT_MAX_EXPAND_PX  = 1000
RIGHT_MAX_EXPAND_PX = 1000

STRIP_H = 10
STRIP_W = 10
STEP_PX = 4
CONTENT_FRAC_THR = 0.002
# Aumentado a 200 para que alcance textos flotantes en diagramas dispersos
ALLOW_GAP_PX     = 200 

BARRIER_MARGIN_PX = 12
BARRIER_MIN_WIDTH_FRAC = 0.40 
BARRIER_MIN_H_PX = 22
BARRIER_MIN_VOVER = 0.20
BARRIER_MIN_HOVER = 0.20

ANTI_TABLE_TOP_CLIP = True
TOP_CLIP_MIN_HOVER = 0.22 
TOP_CLIP_PAD_PX    = 2

# ---------------- DocStructBench IDs ----------------
ID_TO_NAME = {
    0: "title", 1: "plain_text", 2: "abandon", 3: "figure",
    4: "figure_caption", 5: "table", 6: "table_caption",
    7: "table_footnote", 8: "isolate_formula", 9: "formula_caption",
}
TITLE_ID = 0
PLAIN_TEXT_ID = 1
FIGURE_ID = 3
FIGURE_CAPTION_ID = 4
TABLE_ID = 5
TABLE_CAPTION_ID = 6
TABLE_FOOTNOTE_ID = 7

ALLOW_TABLE_AS_FIG_CANDIDATE = True
CANDIDATE_CLASS_IDS = {FIGURE_ID} | ({TABLE_ID} if ALLOW_TABLE_AS_FIG_CANDIDATE else set())
CAPTION_CLASS_IDS   = {FIGURE_CAPTION_ID, TABLE_CAPTION_ID, TABLE_FOOTNOTE_ID}
TEXTLIKE_CLASS_IDS  = {TITLE_ID, PLAIN_TEXT_ID} | CAPTION_CLASS_IDS

# ---------------- I/O ----------------
os.makedirs(OUT_BASE_DIR, exist_ok=True)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------- Utils ----------------
def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def clamp(v, lo, hi): return max(lo, min(hi, v))
def clean_text(t: str) -> str: return " ".join((t or "").replace("\n", " ").split()).strip()
def to_posix_path(path: str) -> str: return path.replace("\\", "/")
def arxiv_pdf_url(doc_id: str) -> str: return f"https://arxiv.org/pdf/{doc_id}.pdf"

def render_pdf_pages(pdf_path: str, out_dir: str, dpi: int):
    doc = fitz.open(pdf_path)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    paths = []
    for pno in range(len(doc)):
        pix = doc[pno].get_pixmap(matrix=mat, alpha=False)
        p = os.path.join(out_dir, f"page_{pno+1:04d}.png")
        pix.save(p)
        paths.append(p)
    doc.close()
    return paths

def px_box_to_pt_rect(px_box, dpi):
    x0,y0,x1,y1 = px_box
    s = 72.0 / dpi
    return fitz.Rect(x0*s, y0*s, x1*s, y1*s)

def pt_rect_to_px_box(rect, dpi):
    s = dpi / 72.0
    return [rect.x0 * s, rect.y0 * s, rect.x1 * s, rect.y1 * s]

def v_overlap_ratio(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter = max(0.0, min(ay1, by1) - max(ay0, by0))
    return inter / max(1.0, min(ay1-ay0, by1-by0))

def box_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0,bx0), max(ay0,by0)
    ix1, iy1 = min(ax1,bx1), min(ay1,by1)
    inter = max(0, ix1-ix0) * max(0, iy1-iy0)
    if inter <= 0: return 0.0
    area_a = max(0, ax1-ax0) * max(0, ay1-ay0)
    area_b = max(0, bx1-bx0) * max(0, by1-by0)
    return inter / max(1e-9, (area_a + area_b - inter))

def dedup_keep_best(items, iou_thr=0.80):
    keep = [True]*len(items)
    for i in range(len(items)):
        if not keep[i]: continue
        for j in range(i+1, len(items)):
            if not keep[j]: continue
            if box_iou(items[i]["bbox_px"], items[j]["bbox_px"]) >= iou_thr:
                if items[i]["score"] >= items[j]["score"]: keep[j] = False
                else: keep[i] = False
    return [it for it,k in zip(items, keep) if k]

def intersection_over_small(a, b):
    ix0, iy0 = max(a[0],b[0]), max(a[1],b[1])
    ix1, iy1 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, ix1-ix0) * max(0, iy1-iy0)
    area_a = max(1e-9, (a[2]-a[0])*(a[3]-a[1]))
    area_b = max(1e-9, (b[2]-b[0])*(b[3]-b[1]))
    return inter / min(area_a, area_b)

def drop_contained_boxes(items, ios_thr=0.90):
    keep = [True]*len(items)
    areas = [max(1e-9, (it["bbox_px"][2]-it["bbox_px"][0])*(it["bbox_px"][3]-it["bbox_px"][1])) for it in items]
    for i in range(len(items)):
        if not keep[i]: continue
        for j in range(len(items)):
            if i == j or not keep[j]: continue
            if areas[j] <= areas[i]: continue
            if intersection_over_small(items[i]["bbox_px"], items[j]["bbox_px"]) >= ios_thr:
                keep[i] = False
                break
    return [it for it,k in zip(items, keep) if k]

def looks_like_label(text: str) -> bool:
    t = clean_text(text)
    if not t: return False
    if t.lower().startswith(("figure", "fig.", "fig ", "table", "tabla")): return False
    if len(t.split()) > MAX_LABEL_WORDS: return False
    if re.fullmatch(r"\[\d+\]", t): return False
    return True

def expand_with_small_plaintext_labels(fig_box_px, text_boxes_all):
    fx0,fy0,fx1,fy1 = fig_box_px
    fw, fh = max(1.0, fx1 - fx0), max(1.0, fy1 - fy0)
    union = [float(fx0), float(fy0), float(fx1), float(fy1)]
    added = 0

    for tb_dict in text_boxes_all:
        if len(tb_dict["text"].split()) > MAX_LABEL_WORDS: continue
        tb = tb_dict["bbox_px"]
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

        if tw > MAX_LABEL_W_FRAC * fw or th > MAX_LABEL_H_FRAC * fh: continue
        if v_overlap_ratio(fig_box_px, tb) < MIN_LABEL_VOVER: continue

        dx = max(0, union[0] - tb[2], tb[0] - union[2])
        dy = max(0, union[1] - tb[3], tb[1] - union[3])
        if dx > LABEL_NEAR_GAP_PX or dy > LABEL_NEAR_GAP_PX: continue

        if not looks_like_label(tb_dict["text"]): continue

        union = [min(union[0], tb[0]), min(union[1], tb[1]), max(union[2], tb[2]), max(union[3], tb[3])]
        added += 1
        if added >= MAX_LABEL_BLOCKS: break
    return union

def find_caption_below(fig_box, caption_boxes, min_hov_fig=0.35, min_hov_cap=0.60, max_gap=160):
    fx0,fy0,fx1,fy1 = fig_box
    best, best_key = None, None

    for cb in caption_boxes:
        bx0,by0,bx1,by1 = cb["bbox_px"]
        inter = max(0.0, min(fx1, bx1) - max(fx0, bx0))
        hov_fig, hov_cap = inter / max(1.0, fx1 - fx0), inter / max(1.0, bx1 - bx0)

        if hov_fig < min_hov_fig or hov_cap < min_hov_cap: continue
        gap = by0 - fy1
        if gap < -10 or gap > max_gap: continue

        key = (gap, -hov_cap, -hov_fig)
        if best is None or key < best_key:
            best = cb
            best_key = key
    return best

def extend_bottom_with_intermediate_text(fig_box, main_cap, text_boxes):
    if not INCLUDE_INTERMEDIATE_SUBCAPTIONS: return fig_box
    fx0,fy0,fx1,fy1 = map(float, fig_box)
    cap_y0 = float(main_cap["bbox_px"][1])
    extended, count = float(fy1), 0

    for tb in sorted(text_boxes, key=lambda d: d["bbox_px"][1]):
        bx0,by0,bx1,by1 = map(float, tb["bbox_px"])
        txt = clean_text(tb.get("text", ""))

        if by0 < fy1 - 10 or by1 > cap_y0 + 1: continue
        if (by1 - by0) > MAX_INTER_BOX_H_PX or len(txt.split()) > MAX_INTER_WORDS: continue
        if CAPTION_FIG_RE.match(txt) or CAPTION_TAB_RE.match(txt): continue

        inter = max(0.0, min(fx1, bx1) - max(fx0, bx0))
        if inter / max(1.0, (fx1 - fx0)) < MIN_HOVER_INTER: continue

        extended = max(extended, by1 + 2.0)
        count += 1
        if count >= MAX_INTER_BOXES: break
    return [fx0, fy0, fx1, extended]

# --------- Integral image (content mask) ----------
def build_integral(gray_u8, white_thr=245):
    return (gray_u8 < white_thr).astype(np.uint8).cumsum(axis=0).cumsum(axis=1)

def rect_sum(ii, x0, y0, x1, y1):
    if x1 <= x0 or y1 <= y0: return 0
    A = ii[y1-1, x1-1]
    B = ii[y0-1, x1-1] if y0 > 0 else 0
    C = ii[y1-1, x0-1] if x0 > 0 else 0
    D = ii[y0-1, x0-1] if (y0 > 0 and x0 > 0) else 0
    return int(A - B - C + D)

def compute_barriers(fig_bbox, boxes_np, cls_np, text_boxes_all, page_w, current_cap_box, caption_boxes_fig, is_plan_b, drawing_rects_px):
    fx0,fy0,fx1,fy1 = map(float, fig_bbox)
    fw = max(1.0, fx1 - fx0)
    fh = max(1.0, fy1 - fy0)

    top_barrier_y  = 0.0
    left_barrier_x = 0.0
    right_barrier_x = float(page_w)
    TOP_NEAR_PX = 120

    for fcap in caption_boxes_fig:
        if current_cap_box and box_iou(fcap["bbox_px"], current_cap_box) > 0.3: continue
        cbx0, cby0, cbx1, cby1 = fcap["bbox_px"]
        hov_x = max(0.0, min(fx1, cbx1) - max(fx0, cbx0)) / fw
        hov_y = max(0.0, min(fy1, cby1) - max(fy0, cby0)) / fh
        if cby1 <= fy0 + TOP_NEAR_PX and hov_x >= 0.1: top_barrier_y = max(top_barrier_y, cby1 + 5)
        if cbx1 <= fx0 + 8 and hov_y >= 0.1: left_barrier_x = max(left_barrier_x, cbx1 + 5)
        if cbx0 >= fx1 - 8 and hov_y >= 0.1: right_barrier_x = min(right_barrier_x, cbx0 - 5)

    # Tablas como barreras
    for b, c in zip(boxes_np, cls_np):
        c = int(c)
        if c == FIGURE_ID:
            tb = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            if box_iou(tb, fig_bbox) > 0.3: continue 
            hov = (max(0.0, min(fx1, tb[2]) - max(fx0, tb[0]))) / fw
            vover = v_overlap_ratio([fx0,fy0,fx1,fy1], tb)
            if tb[3] <= fy0 + TOP_NEAR_PX and hov >= 0.1: top_barrier_y = max(top_barrier_y, tb[3] + 5)
            if tb[2] <= fx0 + 8 and vover >= 0.1: left_barrier_x = max(left_barrier_x, tb[2] + 5)
            if tb[0] >= fx1 - 8 and vover >= 0.1: right_barrier_x = min(right_barrier_x, tb[0] - 5)
        elif c in {TABLE_ID, TABLE_CAPTION_ID, TABLE_FOOTNOTE_ID} | CAPTION_CLASS_IDS:
            tb = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            if current_cap_box and box_iou(tb, current_cap_box) > 0.3: continue
            hov = (max(0.0, min(fx1, tb[2]) - max(fx0, tb[0]))) / fw
            vover = v_overlap_ratio([fx0,fy0,fx1,fy1], tb)
            if tb[3] <= fy0 + TOP_NEAR_PX and hov >= BARRIER_MIN_HOVER: top_barrier_y = max(top_barrier_y, tb[3] + BARRIER_MARGIN_PX)
            if tb[2] <= fx0 + 8 and vover >= BARRIER_MIN_VOVER: left_barrier_x = max(left_barrier_x, tb[2] + BARRIER_MARGIN_PX)
            if tb[0] >= fx1 - 8 and vover >= BARRIER_MIN_VOVER: right_barrier_x = min(right_barrier_x, tb[0] - BARRIER_MARGIN_PX)

    # Texto como barreras (usando PyMuPDF para máxima precisión)
    if not is_plan_b:
        for tb_dict in text_boxes_all:
            tb = tb_dict["bbox_px"]
            txt = tb_dict["text"]
            cls_id = tb_dict.get("cls", PLAIN_TEXT_ID)  # ✅ Usar la clase
            wc = len(txt.split())
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            if th < 10: continue

            hov = (max(0.0, min(fx1, tb[2]) - max(fx0, tb[0]))) / fw
            vover = v_overlap_ratio([fx0,fy0,fx1,fy1], tb)
            width_frac = tw / fw

            area_a = max(0, tb[2]-tb[0]) * max(0, tb[3]-tb[1])
            in_drawing = False
            if area_a > 0:
                for bx0_d, by0_d, bx1_d, by1_d in drawing_rects_px:
                    ix0, iy0 = max(tb[0], bx0_d), max(tb[1], by0_d)
                    ix1, iy1 = min(tb[2], bx1_d), min(tb[3], by1_d)
                    inter = max(0, ix1-ix0) * max(0, iy1-iy0)
                    if inter / area_a > 0.4:
                        in_drawing = True
                        break
            if in_drawing: continue 
            
            # ✅ Los títulos sagrados y cabeceras actúan SIEMPRE como barrera sin importar su longitud
            is_large_text = (width_frac >= 0.45 or wc >= 45 or cls_id == TITLE_ID)
            
            if tb[3] <= fy0 + 40 and hov >= BARRIER_MIN_HOVER and is_large_text:
                top_barrier_y = max(top_barrier_y, tb[3] + BARRIER_MARGIN_PX)
            if tb[2] <= fx0 + 20 and vover >= BARRIER_MIN_VOVER and is_large_text:
                left_barrier_x = max(left_barrier_x, tb[2] + BARRIER_MARGIN_PX)
            if tb[0] >= fx1 - 20 and vover >= BARRIER_MIN_VOVER and is_large_text:
                right_barrier_x = min(right_barrier_x, tb[0] - BARRIER_MARGIN_PX)

    return top_barrier_y, left_barrier_x, right_barrier_x

def expand_by_content(ii, bbox, page_w, page_h, top_max, left_max, right_max, strip_h, strip_w, step, frac_thr, allow_gap, top_min_y=0, left_min_x=0, right_max_x=None):
    x0,y0,x1,y1 = map(int, bbox)
    if right_max_x is None: right_max_x = page_w

    # ---- TOP ----
    best_y0, gap, max_up = y0, 0, min(top_max, y0 - int(top_min_y))
    y = y0 - strip_h
    while max_up > 0 and y >= y0 - max_up:
        yy0, yy1 = max(int(top_min_y), y), min(page_h, max(int(top_min_y), y) + strip_h)
        s = rect_sum(ii, x0, yy0, x1, yy1)
        if (s / max(1, (x1-x0)*(yy1-yy0))) >= frac_thr: best_y0, gap = yy0, 0
        else:
            gap += step
            if gap > allow_gap: break
        y -= step
    y0 = best_y0

    # ---- LEFT ----
    best_x0, gap, max_left = x0, 0, min(left_max, x0 - int(left_min_x))
    x = x0 - strip_w
    while max_left > 0 and x >= x0 - max_left:
        xx0, xx1 = max(int(left_min_x), x), min(page_w, max(int(left_min_x), x) + strip_w)
        s = rect_sum(ii, xx0, y0, xx1, y1)
        if (s / max(1, (xx1-xx0)*(y1-y0))) >= frac_thr: best_x0, gap = xx0, 0
        else:
            gap += step
            if gap > allow_gap: break
        x -= step
    x0 = best_x0

    # ---- RIGHT ----
    best_x1, gap, max_r = x1, 0, min(right_max, int(right_max_x) - x1)
    x = x1
    while max_r > 0 and x <= x1 + max_r:
        xx0, xx1 = x, min(page_w, x + strip_w)
        s = rect_sum(ii, xx0, y0, xx1, y1)
        if (s / max(1, (xx1-xx0)*(y1-y0))) >= frac_thr: best_x1, gap = xx1, 0
        else:
            gap += step
            if gap > allow_gap: break
        x += step
    x1 = best_x1

    return [x0,y0,x1,y1]

def clip_top_using_y0_base(bbox, table_boxes, y0_base, min_hov=0.22, pad=2):
    x0,y0,x1,y1 = map(int, bbox)
    new_y0 = y0
    for tb in table_boxes:
        bx0,by0,bx1,by1 = map(float, tb)
        if by1 > (y0_base + 12): continue
        if (max(0.0, min(x1, bx1) - max(x0, bx0)) / max(1, x1 - x0)) < min_hov: continue
        if not (by1 <= new_y0 or by0 >= y1): new_y0 = max(new_y0, int(by1) + pad)
    return [x0, new_y0, x1, y1]

def prevent_text_slicing(fig_box, text_boxes, pad, max_w, max_h, drawing_rects_px):
    x0, y0, x1, y1 = fig_box
    changed = True
    iters = 0
    max_iters = 10
    
    while changed and iters < max_iters:
        changed = False
        iters += 1
        
        for tb_dict in text_boxes:
            bx0, by0, bx1, by1 = tb_dict["bbox_px"]
            txt = tb_dict["text"]
            cls_id = tb_dict.get("cls", PLAIN_TEXT_ID)
            wc = len(txt.split())
            
            ix0, iy0 = max(x0, bx0), max(y0, by0)
            ix1, iy1 = min(x1, bx1), min(y1, by1)
            
            inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            tb_area = max(1e-9, (bx1 - bx0) * (by1 - by0))
            overlap_ratio = inter_area / tb_area
            
            if overlap_ratio > 0.10:
                area_a = max(0, bx1-bx0) * max(0, by1-by0)
                in_drawing = False
                if area_a > 0:
                    for bx0_d, by0_d, bx1_d, by1_d in drawing_rects_px:
                        ix0_d, iy0_d = max(bx0, bx0_d), max(by0, by0_d)
                        ix1_d, iy1_d = min(bx1, bx1_d), min(by1, by1_d)
                        inter_d = max(0, ix1_d - ix0_d) * max(0, iy1_d - iy0_d)
                        if inter_d / area_a > 0.4:
                            in_drawing = True
                            break
                
                # ✅ Los TITLE_ID NUNCA son expandibles, siempre se escupen.
                is_expandable = (wc <= 45 or in_drawing or cls_id in CAPTION_CLASS_IDS) and cls_id != TITLE_ID
                
                if is_expandable:
                    if overlap_ratio < 0.98: # Expande para tragar
                        new_x0 = min(x0, bx0 - pad)
                        new_y0 = min(y0, by0 - pad)
                        new_x1 = max(x1, bx1 + pad)
                        new_y1 = max(y1, by1 + pad)
                        if new_x0 < x0 or new_y0 < y0 or new_x1 > x1 or new_y1 > y1:
                            x0, y0, x1, y1 = new_x0, new_y0, new_x1, new_y1
                            changed = True
                else:
                    # Es un párrafo grande o un TITLE_ID -> Rebánalo desde el borde más cercano
                    cy_fig = (y0 + y1) / 2
                    cy_tb = (by0 + by1) / 2
                    cx_fig = (x0 + x1) / 2
                    cx_tb = (bx0 + bx1) / 2
                    
                    dx = abs(cx_fig - cx_tb) / max(1, x1 - x0)
                    dy = abs(cy_fig - cy_tb) / max(1, y1 - y0)
                    
                    if dy > dx:
                        if cy_tb < cy_fig and y0 < by1: # Rebana por arriba
                            new_y0 = by1 + pad
                            if y1 > new_y0 > y0:
                                y0 = new_y0
                                changed = True
                        elif cy_tb > cy_fig and y1 > by0: # Rebana por abajo
                            new_y1 = by0 - pad
                            if y0 < new_y1 < y1:
                                y1 = new_y1
                                changed = True
                    else:
                        if cx_tb < cx_fig and x0 < bx1: # Rebana por la izq
                            new_x0 = bx1 + pad
                            if x1 > new_x0 > x0:
                                x0 = new_x0
                                changed = True
                        elif cx_tb > cx_fig and x1 > bx0: # Rebana por la der
                            new_x1 = bx0 - pad
                            if x0 < new_x1 < x1:
                                x1 = new_x1
                                changed = True
                                
    return [clamp(x0, 0, max_w-1), clamp(y0, 0, max_h-1), clamp(x1, 0, max_w-1), clamp(y1, 0, max_h-1)]

# ---------------- DEVICE ----------------
try:
    import torch
    device_arg = "cpu"
except Exception:
    device_arg = "cpu"
print("[INFO] device:", device_arg)

# ---------------- Model ----------------
weights_path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501",
    filename="doclayout_yolo_docstructbench_imgsz1280_2501.pt",
    local_dir=MODELS_DIR,
)
from doclayout_yolo import YOLOv10

try:
    import torch.serialization
    import dill
    from doclayout_yolo.nn.tasks import YOLOv10DetectionModel
    torch.serialization.add_safe_globals([YOLOv10DetectionModel, dill._dill._load_type])
except Exception: pass

model = YOLOv10(weights_path)

# ---------------- Process all PDFs ----------------
import glob
pdf_files = glob.glob(os.path.join(PDFS_FOLDER, "*.pdf"))

if not pdf_files:
    print(f"[WARNING] No se encontraron PDFs en {PDFS_FOLDER}")
    exit()

parser = argparse.ArgumentParser()
parser.add_argument("--save_pages", action="store_true", default=False)
args, _ = parser.parse_known_args()
SAVE_PAGES = args.save_pages

for pdf_idx, PDF_PATH in enumerate(pdf_files, start=1):
    doc_id = os.path.splitext(os.path.basename(PDF_PATH))[0]
    print(f"\n{'='*60}\n[{pdf_idx}/{len(pdf_files)}] Procesando: {doc_id}\n{'='*60}")
    
    OUT_DIR = os.path.join(OUT_BASE_DIR, doc_id)
    IMAGES_DIR = os.path.join(OUT_DIR, "images")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    if SAVE_PAGES:
        PAGES_DIR = os.path.join(OUT_DIR, "pages")
        os.makedirs(PAGES_DIR, exist_ok=True)
        temp_pages_dir = None
    else:
        temp_pages_dir = tempfile.mkdtemp()
        PAGES_DIR = temp_pages_dir
    
    page_paths = render_pdf_pages(PDF_PATH, PAGES_DIR, DPI)
    pdf_doc = fitz.open(PDF_PATH)
    manifest, crop_id = [], 0

    for page_num, page_img in enumerate(page_paths, start=1):
        pdf_page = pdf_doc[page_num - 1]
        res = model.predict(page_img, imgsz=IMGSZ, conf=CONF, device=device_arg)[0]
        boxes_np, cls_np, conf_np = res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy()

        im = Image.open(page_img).convert("RGB")
        W, H = im.size

        if SAFE_CONTENT_EXPAND:
            gray = np.array(im.convert("L"), dtype=np.uint8)
            ii = build_integral(gray, white_thr=WHITE_THR)

        arxiv_watermark_x = 0
        if page_num == 1:
            for keyword in ["arXiv:", "arXiv"]:
                for r in pdf_page.search_for(keyword):
                    px_x0, px_x1 = r.x0 * (DPI / 72.0), r.x1 * (DPI / 72.0)
                    if px_x0 < W * 0.08 and (px_x1 - px_x0) < W * 0.05: 
                        arxiv_watermark_x = max(arxiv_watermark_x, px_x1 + 15)

        drawing_rects_px = []
        scale_fact = DPI / 72.0
        page_area_pt = (W / scale_fact) * (H / scale_fact)
        for d in pdf_page.get_drawings():
            r = d["rect"]
            area = (r.x1 - r.x0) * (r.y1 - r.y0)
            if 50 < area < 0.95 * page_area_pt:
                drawing_rects_px.append([r.x0 * scale_fact, r.y0 * scale_fact, r.x1 * scale_fact, r.y1 * scale_fact])

        # ✅ OBTENER FIGURAS DE YOLO PRIMERO PARA VERIFICAR SOLAPES
        yolo_figures = []
        for b, c in zip(boxes_np, cls_np):
            if int(c) in CANDIDATE_CLASS_IDS:
                yolo_figures.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])

        text_boxes_all = []
        for b in pdf_page.get_text("blocks"):
            if b[6] == 0:
                txt = clean_text(b[4])
                if txt:
                    bb = pt_rect_to_px_box(fitz.Rect(b[:4]), DPI)
                    cls_id = PLAIN_TEXT_ID
                    
                    if CAPTION_FIG_RE.match(txt): 
                        cls_id = FIGURE_CAPTION_ID
                    elif CAPTION_TAB_RE.match(txt): 
                        cls_id = TABLE_CAPTION_ID
                    else:
                        # ✅ FILTRO DE METADATOS: Si no está en un diagrama, proteger enlaces y cabeceras
                        area_a = max(0, bb[2]-bb[0]) * max(0, bb[3]-bb[1])
                        in_visual = False
                        if area_a > 0:
                            for bx0_d, by0_d, bx1_d, by1_d in drawing_rects_px:
                                ix0, iy0 = max(bb[0], bx0_d), max(bb[1], by0_d)
                                ix1, iy1 = min(bb[2], bx1_d), min(bb[3], by1_d)
                                if (max(0, ix1-ix0) * max(0, iy1-iy0)) / area_a > 0.4:
                                    in_visual = True
                                    break
                            if not in_visual:
                                for yf in yolo_figures:
                                    ix0, iy0 = max(bb[0], yf[0]), max(bb[1], yf[1])
                                    ix1, iy1 = min(bb[2], yf[2]), min(bb[3], yf[3])
                                    if (max(0, ix1-ix0) * max(0, iy1-iy0)) / area_a > 0.4:
                                        in_visual = True
                                        break
                        
                        if not in_visual:
                            is_meta = bool(re.search(r'(https?://|www\.|github\.com|@|\.org|\.app|\.edu)', txt.lower()))
                            is_header = (page_num == 1 and bb[1] < H * 0.22)
                            is_margin = (bb[1] < H * 0.05 or bb[3] > H * 0.95)
                            if is_meta or is_header or is_margin:
                                cls_id = TITLE_ID

                    text_boxes_all.append({"bbox_px": bb, "text": txt, "cls": cls_id})

        caption_boxes_fig = []
        for tb in text_boxes_all:
            if tb["cls"] == FIGURE_CAPTION_ID and len(tb["text"].split()) < MAX_CAPTION_WORDS: 
                caption_boxes_fig.append(tb)

        table_boxes = [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b, c in zip(boxes_np, cls_np) if int(c) in {TABLE_ID, TABLE_CAPTION_ID, TABLE_FOOTNOTE_ID}]

        raw_candidates = []
        for b, c, s in zip(boxes_np, cls_np, conf_np):
            if int(c) not in CANDIDATE_CLASS_IDS: continue
            raw_box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]

            if ENABLE_LABEL_MERGE:
                raw_box = expand_with_small_plaintext_labels(raw_box, text_boxes_all)

            cap = find_caption_below(raw_box, caption_boxes_fig, min_hov_fig=MIN_HOVER_FIG, min_hov_cap=MIN_HOVER_CAP, max_gap=MAX_GAP_MAIN_CAPTION_PX)
            if cap is None: cap = find_caption_below(raw_box, caption_boxes_fig, min_hov_fig=FALLBACK_HOVER_FIG, min_hov_cap=FALLBACK_HOVER_CAP, max_gap=MAX_GAP_MAIN_CAPTION_PX)
            if cap is None or CAPTION_TAB_RE.match(cap["text"]) or not CAPTION_FIG_RE.match(cap["text"]): continue

            raw_box = extend_bottom_with_intermediate_text(raw_box, cap, text_boxes_all)
            y1_cut = min(raw_box[3], cap["bbox_px"][1] - 2.0)
            fig_box = [raw_box[0], raw_box[1], raw_box[2], y1_cut]

            x0, y0, x1, y1 = fig_box
            x0, y0 = clamp(int(math.floor(x0)) - PAD_PX, 0, W-1), clamp(int(math.floor(y0)) - PAD_PX, 0, H-1)
            x1, y1 = clamp(int(math.ceil (x1)) + PAD_PX, 0, W-1), clamp(int(math.ceil (y1)) + PAD_PX, 0, H-1)

            if page_num == 1 and arxiv_watermark_x > 0: x0 = max(x0, int(arxiv_watermark_x))
            if x1 <= x0 + 10 or y1 <= y0 + 10: continue

            y0_base = y0
            if SAFE_CONTENT_EXPAND:
                top_barrier, left_barrier, right_barrier = compute_barriers([x0,y0,x1,y1], boxes_np, cls_np, text_boxes_all, W, cap["bbox_px"], caption_boxes_fig, False, drawing_rects_px)
                if page_num == 1 and arxiv_watermark_x > 0: left_barrier = max(left_barrier, arxiv_watermark_x)

                x0,y0,x1,y1 = expand_by_content(
                    ii, [x0,y0,x1,y1], W, H,
                    top_max=TOP_MAX_EXPAND_PX, left_max=LEFT_MAX_EXPAND_PX, right_max=RIGHT_MAX_EXPAND_PX,
                    strip_h=STRIP_H, strip_w=STRIP_W, step=STEP_PX,
                    frac_thr=CONTENT_FRAC_THR, allow_gap=ALLOW_GAP_PX,
                    top_min_y=int(top_barrier), left_min_x=int(left_barrier), right_max_x=int(right_barrier)
                )
                y1 = clamp(min(y1, int(cap["bbox_px"][1]) - 2), 0, H-1)
                if ANTI_TABLE_TOP_CLIP and table_boxes: x0,y0,x1,y1 = clip_top_using_y0_base([x0,y0,x1,y1], table_boxes, y0_base, TOP_CLIP_MIN_HOVER, TOP_CLIP_PAD_PX)

            # ✅ APLICAR EL SISTEMA ANTI-CORTES
            x0, y0, x1, y1 = prevent_text_slicing([x0, y0, x1, y1], text_boxes_all, PAD_PX, W, H, drawing_rects_px)

            if x1 <= x0 + 10 or y1 <= y0 + 10: continue
            raw_candidates.append({"label": "figure", "score": float(s), "bbox_px": [x0,y0,x1,y1], "caption": cap["text"], "caption_bbox_px": cap["bbox_px"]})

        # --- PLAN B ---
        matched_captions = {cand["caption"] for cand in raw_candidates}
        for cap in caption_boxes_fig:
            if cap["text"] in matched_captions: continue

            cx0, cy0, cx1, cy1 = cap["bbox_px"]
            x0, y0, x1, y1 = map(float, [cx0, max(0, cy0 - 25), cx1, max(0, cy0 - 2)])
            
            if page_num == 1 and arxiv_watermark_x > 0: x0 = max(x0, int(arxiv_watermark_x))
            y0_base = y0

            if SAFE_CONTENT_EXPAND:
                top_barrier, left_barrier, right_barrier = compute_barriers([x0,y0,x1,y1], boxes_np, cls_np, text_boxes_all, W, cap["bbox_px"], caption_boxes_fig, True, drawing_rects_px)
                if page_num == 1 and arxiv_watermark_x > 0: left_barrier = max(left_barrier, arxiv_watermark_x)

                x0, y0, x1, y1 = expand_by_content(
                    ii, [x0,y0,x1,y1], W, H,
                    top_max=TOP_MAX_EXPAND_PX + 300, left_max=LEFT_MAX_EXPAND_PX, right_max=RIGHT_MAX_EXPAND_PX,
                    strip_h=STRIP_H, strip_w=STRIP_W, step=STEP_PX,
                    frac_thr=CONTENT_FRAC_THR, allow_gap=ALLOW_GAP_PX + 60,
                    top_min_y=int(top_barrier), left_min_x=int(left_barrier), right_max_x=int(right_barrier)
                )
                y1 = clamp(min(y1, int(cy0) - 2), 0, H-1)
                if ANTI_TABLE_TOP_CLIP and table_boxes: x0, y0, x1, y1 = clip_top_using_y0_base([x0,y0,x1,y1], table_boxes, y0_base, TOP_CLIP_MIN_HOVER, TOP_CLIP_PAD_PX)

            # ✅ APLICAR EL SISTEMA ANTI-CORTES AL PLAN B
            x0, y0, x1, y1 = prevent_text_slicing([x0, y0, x1, y1], text_boxes_all, PAD_PX, W, H, drawing_rects_px)

            if x1 <= x0 + 10 or y1 <= y0 + 30: continue
            raw_candidates.append({"label": "figure", "score": 0.50, "bbox_px": [x0, y0, x1, y1], "caption": cap["text"], "caption_bbox_px": cap["bbox_px"]})

        merged = {}
        for cand in raw_candidates:
            key = clean_text(cand["caption"]).lower()
            if key not in merged: merged[key] = cand
            else:
                a, b = merged[key]["bbox_px"], cand["bbox_px"]
                merged[key]["bbox_px"] = [min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3])]
                merged[key]["score"] = max(merged[key]["score"], cand["score"])

        page_candidates = drop_contained_boxes(dedup_keep_best(list(merged.values()), DEDUP_IOU_THR), DROP_CONTAINED_IOS)

        for cand in page_candidates:
            crop_id += 1
            fpath = os.path.join(IMAGES_DIR, f"p{page_num:04d}_figure_{crop_id:05d}.png")
            im.crop((cand["bbox_px"])).save(fpath, "PNG")
            manifest.append({
                "id": crop_id, "doc_id": doc_id,
                "source_pdf": to_posix_path(os.path.relpath(PDF_PATH, ".")),
                "source_pdf_url": arxiv_pdf_url(doc_id), "page": page_num,
                "image_path": to_posix_path(os.path.relpath(fpath, ".")),
                "caption": cand["caption"], "md5": md5_file(fpath),
            })

    pdf_doc.close()
    if temp_pages_dir: shutil.rmtree(temp_pages_dir)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f: json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}\n[OK] Procesamiento completado\n{'='*60}")