#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Config (env overridable)
# -----------------------------
IN_DIR   = Path(os.environ.get("IN_DIR", "./input_images"))
OUT_DIR  = Path(os.environ.get("OUT_DIR", "./output_labeled"))
WEIGHTS  = os.environ.get("YOLO_WEIGHTS", "yolo12s.pt")
DEVICE   = os.environ.get("DEVICE", "cpu")     # 推荐 cpu（5090 sm_120 可能与 torch 不兼容）
CONF_THR = float(os.environ.get("CONF", "0.25"))
IOU_THR  = float(os.environ.get("IOU", "0.7"))

BENCH_START_ID = int(os.environ.get("BENCH_START_ID", "1"))
STOP_START_ID  = int(os.environ.get("STOP_START_ID", "1"))

# Label layout
PAD    = int(os.environ.get("PAD", "10"))      # label 内边距
MARGIN = int(os.environ.get("MARGIN", "6"))    # clamp within image
STOP_Y_OFFSET = float(os.environ.get("STOP_Y_OFFSET", "0.15"))  # stop badge 在 bbox 中心略偏下

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# -----------------------------
# Geometry
# -----------------------------
Box = Tuple[int, int, int, int]  # (x0,y0,x1,y1)

def center_of(b: Box) -> Tuple[float, float]:
    x0, y0, x1, y1 = b
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

# -----------------------------
# Fonts & drawing
# -----------------------------
def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

def _text_bbox(text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    dummy = Image.new("RGB", (8, 8))
    d = ImageDraw.Draw(dummy)
    b = d.textbbox((0, 0), text, font=font)
    return (b[2] - b[0], b[3] - b[1])  # (w, h)

def fit_font_for_box(text: str, max_w: int, max_h: int, start_size: int, pad: int) -> ImageFont.ImageFont:
    """
    找到一个字号，使得 (text + 2*pad) <= (max_w, max_h)
    """
    size = max(8, start_size)
    while size >= 8:
        font = load_font(size)
        tw, th = _text_bbox(text, font)
        if tw + 2 * pad <= max_w and th + 2 * pad <= max_h:
            return font
        size -= 1
    return load_font(8)

def render_bench_label(img: Image.Image, top_left: Tuple[int, int], w: int, h: int, text: str, font: ImageFont.ImageFont):
    x, y = top_left
    plate = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(plate)

    border = max(3, h // 9)
    d.rounded_rectangle(
        (0, 0, w - 1, h - 1),
        radius=max(10, h // 3),
        fill=(255, 230, 0, 245),      # bright yellow
        outline=(0, 0, 0, 255),
        width=border,
    )

    tb = d.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = (w - tw) // 2 - tb[0]
    ty = (h - th) // 2 - tb[1]

    stroke = max(2, int(getattr(font, "size", 20) * 0.10))
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx == 0 and dy == 0:
                continue
            d.text((tx + dx, ty + dy), text, font=font, fill=(255, 255, 255, 255))
    d.text((tx, ty), text, font=font, fill=(0, 0, 0, 255))

    img.alpha_composite(plate, (x, y))

def render_stop_badge(img: Image.Image, top_left: Tuple[int, int], diam: int, text: str, font: ImageFont.ImageFont):
    x, y = top_left
    badge = Image.new("RGBA", (diam, diam), (0, 0, 0, 0))
    d = ImageDraw.Draw(badge)

    border = max(3, diam // 12)
    d.ellipse(
        (0, 0, diam - 1, diam - 1),
        fill=(220, 20, 60, 245),      # crimson/red
        outline=(255, 255, 255, 255),
        width=border,
    )
    inset = border + 2
    d.ellipse(
        (inset, inset, diam - 1 - inset, diam - 1 - inset),
        outline=(0, 0, 0, 220),
        width=max(2, border // 2),
    )

    tb = d.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = (diam - tw) // 2 - tb[0]
    ty = (diam - th) // 2 - tb[1]

    stroke = max(2, int(getattr(font, "size", 18) * 0.12))
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx == 0 and dy == 0:
                continue
            d.text((tx + dx, ty + dy), text, font=font, fill=(0, 0, 0, 255))
    d.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))

    img.alpha_composite(badge, (x, y))

# -----------------------------
# Placement: inside bbox (your request)
# -----------------------------
def place_inside_box(
    obj_box: Box,
    label_w: int,
    label_h: int,
    W: int,
    H: int,
    y_offset_ratio: float = 0.0,
) -> Tuple[int, int]:
    x0, y0, x1, y1 = obj_box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0 + y_offset_ratio * bh

    px = int(round(cx - label_w / 2.0))
    py = int(round(cy - label_h / 2.0))

    px = max(MARGIN, min(px, W - MARGIN - label_w))
    py = max(MARGIN, min(py, H - MARGIN - label_h))
    return px, py

# -----------------------------
# YOLO helpers
# -----------------------------
def norm_class_name(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def is_stop_name(n: str) -> bool:
    # 兼容常见写法
    return n in {"stop_sign", "stopsign", "stop"} or "stop_sign" in n or "stopsign" in n

# -----------------------------
# Main
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "ultralytics not found. Install: pip install ultralytics\n"
            f"Original error: {e}"
        )

    model = YOLO(WEIGHTS)

    paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not paths:
        print(f"[WARN] No images found in: {IN_DIR}")
        return

    print(f"[INFO] Input : {IN_DIR}  ({len(paths)} images)")
    print(f"[INFO] Output: {OUT_DIR}")
    print(f"[INFO] YOLO  : {WEIGHTS}, device={DEVICE}, conf={CONF_THR}, iou={IOU_THR}")

    for img_path in paths:
        im = Image.open(img_path).convert("RGBA")
        W, H = im.size

        res = model.predict(
            source=str(img_path),
            device=DEVICE,
            conf=CONF_THR,
            iou=IOU_THR,
            verbose=False,
        )[0]

        benches: List[Box] = []
        stops: List[Box] = []

        if res.boxes is not None and len(res.boxes) > 0:
            names = model.names  # id->name
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls  = res.boxes.cls.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()

            for b, c, s in zip(xyxy, cls, conf):
                name = names.get(int(c), str(int(c))) if isinstance(names, dict) else names[int(c)]
                name = norm_class_name(name)

                x0, y0, x1, y1 = [int(round(v)) for v in b.tolist()]
                x0 = max(0, min(x0, W - 1))
                y0 = max(0, min(y0, H - 1))
                x1 = max(0, min(x1, W))
                y1 = max(0, min(y1, H))
                if x1 <= x0 or y1 <= y0:
                    continue

                if name == "bench":
                    benches.append((x0, y0, x1, y1))
                elif is_stop_name(name):
                    stops.append((x0, y0, x1, y1))

        # sort left->right by center x, tiebreak y
        def sort_lr(b: Box):
            cx, cy = center_of(b)
            return (cx, cy)

        benches = sorted(benches, key=sort_lr)
        stops = sorted(stops, key=sort_lr)

        placed = 0

        # -------- bench: ID in bbox center --------
        bench_id = BENCH_START_ID
        for b in benches:
            x0, y0, x1, y1 = b
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)

            text = str(bench_id)

            # label 的最大尺寸（必须能塞进 bench bbox）
            max_w = max(24, int(0.92 * bw))
            max_h = max(20, int(0.55 * bh))

            start_font = max(16, int(min(W, H) * 0.040))
            font = fit_font_for_box(text, max_w, max_h, start_font, PAD)
            tw, th = _text_bbox(text, font)

            lw = min(max_w, tw + 2 * PAD)
            lh = min(max_h, th + 2 * PAD)

            px, py = place_inside_box(b, lw, lh, W, H, y_offset_ratio=0.0)
            render_bench_label(im, (px, py), lw, lh, text, font)

            bench_id += 1
            placed += 1

        # -------- stop: ID in bbox center, slightly lower --------
        stop_id = STOP_START_ID
        for b in stops:
            x0, y0, x1, y1 = b
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)

            text = str(stop_id)

            # badge 直径必须能塞进 stop bbox
            max_d = max(28, int(0.85 * min(bw, bh)))

            start_font = max(14, int(min(W, H) * 0.036))
            # 对圆形：用 max_d x max_d 当作约束
            font = fit_font_for_box(text, max_d, max_d, start_font, PAD)
            tw, th = _text_bbox(text, font)
            diam = min(max_d, max(tw, th) + 2 * PAD)

            px, py = place_inside_box(b, diam, diam, W, H, y_offset_ratio=STOP_Y_OFFSET)
            render_stop_badge(im, (px, py), diam, text, font)

            stop_id += 1
            placed += 1

        out_path = OUT_DIR / img_path.name
        im.convert("RGB").save(out_path, quality=95)
        print(f"[OK] {img_path.name} -> {out_path.name} | benches={len(benches)} stops={len(stops)} placed={placed}")

if __name__ == "__main__":
    main()