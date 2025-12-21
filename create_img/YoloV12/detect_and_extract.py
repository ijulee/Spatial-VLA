#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from typing import List, Tuple, Dict

import cv2
import numpy as np
from ultralytics import YOLO


# -------------------------
# Helpers: class id lookup
# -------------------------
def find_class_id(model_names: Dict[int, str], targets: List[str]) -> int:
    """
    Find class id by matching any target string in class name (case-insensitive).
    Return -1 if not found.
    """
    targets_l = [t.lower() for t in targets]
    for k, v in model_names.items():
        name = str(v).lower()
        if any(t in name for t in targets_l):
            return int(k)
    return -1


# -------------------------
# Sorting (left->right, tiebreak top->bottom)
# -------------------------
def sort_bboxes_left_to_right(bboxes_xyxy: List[Tuple[float, float, float, float]],
                              x_eps_ratio: float = 0.02) -> List[int]:
    """
    Sort by center-x; if nearly aligned in x, sort by center-y (top->bottom).
    x_eps_ratio is relative to image width; we will convert outside if needed.
    """
    # This function returns indices; actual eps computed outside if desired.
    # Here we simply sort by (cx, cy) and let the caller supply pre-grouping if needed.
    centers = []
    for i, (x0, y0, x1, y1) in enumerate(bboxes_xyxy):
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        centers.append((i, cx, cy))
    centers.sort(key=lambda t: (t[1], t[2]))
    return [t[0] for t in centers]


# -------------------------
# Badge drawing
# -------------------------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def draw_rounded_rect(img: np.ndarray,
                      x: int, y: int, w: int, h: int,
                      radius: int,
                      fill_bgr: Tuple[int, int, int],
                      border_bgr: Tuple[int, int, int],
                      border_thickness: int) -> None:
    """
    Draw rounded rectangle by combining rectangles + circles.
    """
    H, W = img.shape[:2]
    x = clamp(x, 0, W - 1)
    y = clamp(y, 0, H - 1)
    w = max(1, w)
    h = max(1, h)
    r = max(0, min(radius, min(w, h) // 2))

    # filled
    overlay = img.copy()
    # center rects
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), fill_bgr, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), fill_bgr, -1)
    # four corners
    cv2.circle(overlay, (x + r, y + r), r, fill_bgr, -1)
    cv2.circle(overlay, (x + w - r, y + r), r, fill_bgr, -1)
    cv2.circle(overlay, (x + r, y + h - r), r, fill_bgr, -1)
    cv2.circle(overlay, (x + w - r, y + h - r), r, fill_bgr, -1)

    img[:] = overlay

    # border
    # draw border using same primitives
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), border_bgr, border_thickness)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), border_bgr, border_thickness)
    cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, border_bgr, border_thickness)
    cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, border_bgr, border_thickness)
    cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, border_bgr, border_thickness)
    cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, border_bgr, border_thickness)


def put_centered_text(img: np.ndarray,
                      text: str,
                      x: int, y: int, w: int, h: int,
                      font_scale: float,
                      thickness: int,
                      text_bgr: Tuple[int, int, int],
                      stroke_bgr: Tuple[int, int, int],
                      stroke: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2  # baseline-ish

    # stroke
    if stroke > 0:
        for dx in range(-stroke, stroke + 1):
            for dy in range(-stroke, stroke + 1):
                if dx == 0 and dy == 0:
                    continue
                cv2.putText(img, text, (tx + dx, ty + dy), font, font_scale, stroke_bgr,
                            thickness=thickness + 1, lineType=cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, font_scale, text_bgr,
                thickness=thickness, lineType=cv2.LINE_AA)


def place_label_box(img_w: int, img_h: int,
                    bbox: Tuple[int, int, int, int],
                    label_w: int, label_h: int,
                    gap: int,
                    side: str = "auto") -> Tuple[int, int, str]:
    """
    Place label tightly to left or right of bbox. Return (px, py, used_side).
    """
    x0, y0, x1, y1 = bbox
    # prefer align to top
    py = clamp(y0, 0, img_h - label_h)

    if side == "left":
        px = x0 - label_w - gap
        if px < 0:
            px = x1 + gap
            used = "right"
        else:
            used = "left"
    elif side == "right":
        px = x1 + gap
        if px + label_w > img_w:
            px = x0 - label_w - gap
            used = "left"
        else:
            used = "right"
    else:
        # auto: pick the side with more room
        left_room = x0
        right_room = img_w - x1
        if left_room >= (label_w + gap):
            px = x0 - label_w - gap
            used = "left"
        elif right_room >= (label_w + gap):
            px = x1 + gap
            used = "right"
        else:
            # neither fits well -> clamp near left/right
            if right_room >= left_room:
                px = clamp(x1 + gap, 0, img_w - label_w)
                used = "right"
            else:
                px = clamp(x0 - label_w - gap, 0, img_w - label_w)
                used = "left"

    px = clamp(px, 0, img_w - label_w)
    return int(px), int(py), used


def draw_bench_id(img: np.ndarray, number: int, bbox: Tuple[int, int, int, int], side: str = "auto") -> None:
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    # bigger, very visible
    label_w = max(64, int(0.55 * bw))
    label_h = max(44, int(0.55 * bh))
    gap = max(3, int(0.03 * bw))

    H, W = img.shape[:2]
    px, py, _ = place_label_box(W, H, bbox, label_w, label_h, gap, side=side)

    # yellow rounded badge
    border = max(3, label_h // 8)
    radius = max(10, label_h // 2)
    draw_rounded_rect(
        img, px, py, label_w, label_h,
        radius=radius,
        fill_bgr=(0, 230, 255),      # BGR: yellow-ish
        border_bgr=(0, 0, 0),
        border_thickness=border
    )

    text = str(number)
    font_scale = max(0.8, label_h / 55.0)
    thickness = max(2, label_h // 10)
    put_centered_text(
        img, text, px, py, label_w, label_h,
        font_scale=font_scale,
        thickness=thickness,
        text_bgr=(0, 0, 0),
        stroke_bgr=(255, 255, 255),
        stroke=max(1, thickness // 2)
    )


def draw_stop_id(img: np.ndarray, number: int, bbox: Tuple[int, int, int, int], side: str = "auto") -> None:
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    # different style: red circular badge
    d = max(46, int(0.60 * min(bw, bh)))
    label_w = d
    label_h = d
    gap = max(3, int(0.03 * bw))

    H, W = img.shape[:2]
    px, py, _ = place_label_box(W, H, bbox, label_w, label_h, gap, side=side)

    cx = px + label_w // 2
    cy = py + label_h // 2
    r = min(label_w, label_h) // 2 - 1

    # red fill + black border
    cv2.circle(img, (cx, cy), r, (0, 0, 255), -1)          # red
    cv2.circle(img, (cx, cy), r, (0, 0, 0), max(3, r // 8))  # black border

    text = str(number)
    font_scale = max(0.8, label_h / 60.0)
    thickness = max(2, label_h // 10)

    # centered text with black stroke for legibility
    # (white text)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = cx - tw // 2
    ty = cy + th // 2
    stroke = max(2, thickness // 2)
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx == 0 and dy == 0:
                continue
            cv2.putText(img, text, (tx + dx, ty + dy), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# -------------------------
# Main processing
# -------------------------
def process_folder(input_dir: str,
                   output_dir: str,
                   weights: str,
                   device: str,
                   conf: float,
                   iou: float,
                   side: str,
                   bench_start: int,
                   stop_start: int,
                   save_yolo_boxes: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(weights)

    # robust class ids
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    bench_id = find_class_id(names, targets=["bench"])
    stop_id = find_class_id(names, targets=["stop sign", "stop_sign", "stopsign"])

    # fallback to COCO ids if not found
    if bench_id < 0:
        bench_id = 13
    if stop_id < 0:
        stop_id = 11

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(input_dir, e)))
    paths.sort()

    if not paths:
        print(f"[ERR] No images found in: {input_dir}")
        return

    print(f"[INFO] Found {len(paths)} images.")
    print(f"[INFO] bench_id={bench_id} ({names.get(bench_id,'?')})  stop_id={stop_id} ({names.get(stop_id,'?')})")

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read: {p}")
            continue

        H, W = img.shape[:2]

        # YOLO predict
        # NOTE: if your GPU is incompatible, use --device cpu
        results = model.predict(
            source=img,
            verbose=False,
            device=device,
            conf=conf,
            iou=iou
        )
        r = results[0]

        benches = []
        stops = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for (x0, y0, x1, y1), c, cf in zip(xyxy, cls, confs):
                if cf < conf:
                    continue
                x0i = int(clamp(int(round(x0)), 0, W - 1))
                y0i = int(clamp(int(round(y0)), 0, H - 1))
                x1i = int(clamp(int(round(x1)), 0, W))
                y1i = int(clamp(int(round(y1)), 0, H))
                bbox = (x0i, y0i, x1i, y1i)

                if c == bench_id:
                    benches.append(bbox)
                elif c == stop_id:
                    stops.append(bbox)

        # sort + draw IDs
        if benches:
            idxs = sort_bboxes_left_to_right([(b[0], b[1], b[2], b[3]) for b in benches])
            n = bench_start
            for i in idxs:
                draw_bench_id(img, n, benches[i], side=side)
                n += 1

        if stops:
            idxs = sort_bboxes_left_to_right([(b[0], b[1], b[2], b[3]) for b in stops])
            n = stop_start
            for i in idxs:
                draw_stop_id(img, n, stops[i], side=side)
                n += 1

        # optionally draw yolo boxes for debugging
        if save_yolo_boxes:
            # r.plot() returns RGB; we want BGR
            dbg = r.plot()
            dbg = cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR)
            # overlay IDs already drawn on img -> just save both
            out_dbg = os.path.join(output_dir, os.path.splitext(os.path.basename(p))[0] + "_yolo.jpg")
            cv2.imwrite(out_dbg, dbg)

        out_path = os.path.join(output_dir, os.path.basename(p))
        cv2.imwrite(out_path, img)
        print(f"[OK] {os.path.basename(p)}  benches={len(benches)}  stops={len(stops)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Folder containing scene images")
    ap.add_argument("--output_dir", type=str, required=True, help="Output folder for labeled images")
    ap.add_argument("--weights", type=str, default="yolo12s.pt", help="YOLO weights path")
    ap.add_argument("--device", type=str, default="cpu", help="cpu, 0, 1, cuda:0 etc. (use cpu if 5090 incompatible)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--side", type=str, default="auto", choices=["auto", "left", "right"],
                    help="Place ID to left/right of object bbox (auto picks best)")
    ap.add_argument("--bench_start", type=int, default=1, help="bench id start number")
    ap.add_argument("--stop_start", type=int, default=1, help="stop sign id start number")
    ap.add_argument("--save_yolo_boxes", action="store_true", help="Also save YOLO debug images with boxes")
    args = ap.parse_args()

    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        side=args.side,
        bench_start=args.bench_start,
        stop_start=args.stop_start,
        save_yolo_boxes=args.save_yolo_boxes
    )


if __name__ == "__main__":
    main()