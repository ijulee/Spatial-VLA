#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(input_dir: Path) -> List[Path]:
    out = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def sort_left_to_right_with_vertical_tiebreak(
    bboxes: List[Tuple[int, int, int, int]],
    x_eps_ratio: float,
    image_w: int,
) -> List[int]:
    """
    先按中心 x 排序；若两者 x 很接近(<= x_eps_ratio*W)，则按中心 y 排序(从上到下)。
    """
    x_eps = x_eps_ratio * image_w
    centers = []
    for (x0, y0, x1, y1) in bboxes:
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        centers.append((cx, cy))

    idxs = list(range(len(bboxes)))
    idxs.sort(key=lambda i: centers[i][0])  # primary: cx

    out = []
    i = 0
    while i < len(idxs):
        j = i + 1
        group = [idxs[i]]
        while j < len(idxs) and abs(centers[idxs[j]][0] - centers[idxs[i]][0]) <= x_eps:
            group.append(idxs[j])
            j += 1
        group.sort(key=lambda k: centers[k][1])  # tiebreak: cy
        out.extend(group)
        i = j
    return out


def draw_tag(
    img_bgr,
    text: str,
    anchor_bbox: Tuple[int, int, int, int],
    *,
    fill_bgr: Tuple[int, int, int],
    border_bgr: Tuple[int, int, int],
    text_bgr: Tuple[int, int, int],
    prefer_right: bool = True,
    pad: int = 6,
    border: int = 3,
    font_scale: float = 0.9,
    font_thickness: int = 2,
):
    """
    在 bbox 左/右侧紧贴画一个矩形标签（不同颜色区分 bench/stop sign）
    """
    H, W = img_bgr.shape[:2]
    x0, y0, x1, y1 = map(int, anchor_bbox)
    cy = (y0 + y1) // 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    tag_w = tw + 2 * pad
    tag_h = th + 2 * pad
    tag_h = max(tag_h, int(1.2 * th) + 2 * pad)

    # 位置：优先右侧，否则左侧
    right_x = x1 + 6
    left_x = x0 - 6 - tag_w

    if prefer_right and right_x + tag_w <= W - 1:
        tx0 = right_x
    elif left_x >= 0:
        tx0 = left_x
    else:
        tx0 = clamp(right_x, 0, W - tag_w)

    ty0 = clamp(cy - tag_h // 2, 0, H - tag_h)
    tx1 = tx0 + tag_w
    ty1 = ty0 + tag_h

    # 填充 + 边框
    cv2.rectangle(img_bgr, (tx0, ty0), (tx1, ty1), fill_bgr, thickness=-1)
    cv2.rectangle(img_bgr, (tx0, ty0), (tx1, ty1), border_bgr, thickness=border)

    # 文字居中
    text_x = tx0 + (tag_w - tw) // 2
    text_y = ty0 + (tag_h + th) // 2 - 2

    # 伪描边：白描边 + 黑字
    cv2.putText(img_bgr, text, (text_x, text_y), font, font_scale, (255, 255, 255),
                font_thickness + 2, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (text_x, text_y), font, font_scale, text_bgr,
                font_thickness, cv2.LINE_AA)


def detect_bench_and_stop(model: YOLO, img_bgr, device: str, conf: float, iou: float):
    """
    返回 benches / stops 两类 bbox list: [(x0,y0,x1,y1), ...]
    """
    results = model.predict(source=img_bgr, verbose=False, device=device, conf=conf, iou=iou)
    r0 = results[0]
    boxes = r0.boxes
    if boxes is None or len(boxes) == 0:
        return [], []

    names = model.names  # {id: name}
    benches = []
    stops = []

    for b in boxes:
        cls = int(b.cls[0].item())
        name = names.get(cls, str(cls))
        x0, y0, x1, y1 = b.xyxy[0].tolist()

        # 只要类别名对上就收集
        if name == "bench":
            benches.append((int(x0), int(y0), int(x1), int(y1)))
        elif name in ("stop sign", "stop_sign", "stopsign"):
            stops.append((int(x0), int(y0), int(x1), int(y1)))

    return benches, stops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="输入图片文件夹")
    ap.add_argument("--output_dir", type=str, required=True, help="输出图片文件夹")
    ap.add_argument("--weights", type=str, default="yolo12s.pt", help="YOLO 权重路径")
    ap.add_argument("--device", type=str, default="cpu", help="cpu / 0 / cuda:0 等（推荐 cpu）")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--x_eps_ratio", type=float, default=0.03, help="x 近似同列阈值比例，用于 y tiebreak")
    ap.add_argument("--bench_start", type=int, default=1, help="bench 编号起始")
    ap.add_argument("--stop_start", type=int, default=1, help="stop sign 编号起始")
    ap.add_argument("--verbose_per_image", action="store_true", help="每张图都打印检测数量")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    imgs = list_images(in_dir)
    if not imgs:
        print(f"[ERROR] No images found in {in_dir}")
        return

    print(f"[INFO] Found {len(imgs)} images.")
    print(f"[INFO] weights={args.weights}, device={args.device}, conf={args.conf}, iou={args.iou}")
    print(f"[INFO] Saving to: {out_dir}")

    for idx, p in enumerate(imgs, start=1):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Cannot read: {p}")
            continue

        H, W = img.shape[:2]
        benches, stops = detect_bench_and_stop(model, img, args.device, args.conf, args.iou)

        rel = p.relative_to(in_dir)
        # ✅ 每张图都显示数量（默认就显示；也可以用 --verbose_per_image 强制）
        print(f"[IMG] {rel} | benches={len(benches)} | stops={len(stops)}")

        # bench：黄底黑边黑字
        if benches:
            order = sort_left_to_right_with_vertical_tiebreak(benches, args.x_eps_ratio, W)
            for k, bi in enumerate(order):
                bid = args.bench_start + k
                bbox = benches[bi]
                draw_tag(
                    img,
                    text=str(bid),
                    anchor_bbox=bbox,
                    fill_bgr=(0, 255, 255),   # bright yellow (BGR)
                    border_bgr=(0, 0, 0),     # black border
                    text_bgr=(0, 0, 0),       # black text
                    prefer_right=True,
                    font_scale=1.0,
                    font_thickness=2,
                    border=3,
                    pad=7,
                )

        # stop sign：更醒目亮红底 + 白边 + 黑字
        if stops:
            order = sort_left_to_right_with_vertical_tiebreak(stops, args.x_eps_ratio, W)
            for k, si in enumerate(order):
                sid = args.stop_start + k
                bbox = stops[si]
                # stop sign：亮绿色底 + 黑边 + 黑字
                draw_tag(
                    img,
                    text=str(sid),
                    anchor_bbox=bbox,
                    fill_bgr=(0, 255, 0),   # ✅ bright green (BGR)
                    border_bgr=(0, 0, 0),   # ✅ black border
                    text_bgr=(0, 0, 0),     # black text
                    prefer_right=True,
                    font_scale=1.0,
                    font_thickness=2,
                    border=3,
                    pad=7,
                )

        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)

        if idx % 50 == 0 or idx == len(imgs):
            print(f"[INFO] {idx}/{len(imgs)} done.")

    print("[DONE]")


if __name__ == "__main__":
    main()