import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageChops
from ultralytics import YOLO

# ========= TUNABLE PARAMETERS =========

# Paths
IMG_DIR = Path("./src")                 # Directory containing sprites
OUTPUT_DIR = Path("./output")          # Output directory
BACKGROUND_FILE_NAME = "background.jpeg"

# Group counts
ANIMAL_GROUP_COUNT_RANGE = (2, 2)   # 固定 2 个 animal groups
BENCH_COUNT_RANGE = (5, 5)          # 固定 5 个 benches

# Internal counts
ANIMALS_PER_GROUP_RANGE = (4, 4)    # 每个 animal group 固定 4 只动物
PEOPLE_PER_BENCH_RANGE = (0, 2)     # [min, max] people per bench (0 = empty bench)

# Radii of rings (in pixels)
ANIMAL_RING_RADIUS = 370               # Animals around stop sign
PEOPLE_RING_RADIUS = 300               # People around bench

# Overlap control on rings (closer to 1.0 = less overlap)
ANIMAL_WIDTH_SCALE = 0.9
PEOPLE_WIDTH_SCALE = 0.9

# Max attempts to sample a valid angle on the ring
ANIMAL_MAX_TRIES = 100
PEOPLE_MAX_TRIES = 100

# Bench-front arc in radians (fractions of pi)
# Real angle = pi * BENCH_ANGLE_*_FRAC
BENCH_ANGLE_MIN_FRAC = 0.2             # Left-front of bench
BENCH_ANGLE_MAX_FRAC = 0.8             # Right-front of bench

# Canvas margin (objects will stay inside [MARGIN, W-MARGIN])
MARGIN = 30

# Grid layout for placing groups
GRID_ROWS = 3                          # Number of rows in global grid
GRID_COLS = 3                          # Number of columns in global grid
GRID_JITTER_SCALE = 0.25               # Random jitter inside each cell (0 ~ 0.4 is reasonable)

# Number of base layouts
NUM_BASE_SCENES = 100

# 15 张里有 6 张不包含 person
BUS_VARIANTS_PER_SCENE = 15
NO_PERSON_VARIANTS_PER_SCENE = 6

# Bus augmentation parameters
BUS_SPRITE_FILE_NAME = "clock.png"       # Sprite file name for the bus
BUS_ROTATE = True                       # Randomly rotate bus or not
BUS_DIFF_THRESHOLD = 8                  # Threshold for "blank" region detection (vs pure background)
BUS_MAX_PLACEMENT_TRIES = 200           # Max tries to find a blank region for each bus
BUS_MIN_CENTER_DIST = 150.0             # Min distance between bus centers across variants of the same base scene

# YOLO filtering (must satisfy counts to save)
YOLO_WEIGHTS = "yolo12s.pt"
YOLO_CONF = 0.20
YOLO_IMGSZ = 1280
NEED_BENCH = 5
NEED_STOP_SIGN = 2

# =====================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BACKGROUND_FILE = IMG_DIR / BACKGROUND_FILE_NAME
BUS_FILE = IMG_DIR / BUS_SPRITE_FILE_NAME

SPRITE_FILES = {
    "bench": IMG_DIR / "bench.png",
    "person": IMG_DIR / "person.png",
    "stopSign": IMG_DIR / "stopSign.png",
    "zebra": IMG_DIR / "zebra.png",
    "elephant": IMG_DIR / "elephant.png",
    "giraffe": IMG_DIR / "giraffe.png",
}


def load_sprites() -> Dict[str, Image.Image]:
    """Load all main sprites as RGBA images."""
    sprites: Dict[str, Image.Image] = {}
    for name, path in SPRITE_FILES.items():
        img = Image.open(path).convert("RGBA")
        sprites[name] = img
    return sprites


def paste_sprite(bg: Image.Image, sprite: Image.Image, x: int, y: int) -> None:
    """
    Paste sprite onto bg at (x, y), handling alpha and cropping
    if the sprite goes partially outside the canvas.
    """
    w, h = sprite.size
    bg_w, bg_h = bg.size

    if x >= bg_w or y >= bg_h:
        return

    crop_x0 = max(0, -x)
    crop_y0 = max(0, -y)
    crop_x1 = min(w, bg_w - x)
    crop_y1 = min(h, bg_h - y)

    if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
        return

    cropped = sprite.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    bg.alpha_composite(cropped, (x + crop_x0, y + crop_y0))


# ========= Group generation: animal groups / bench groups =========

def build_animal_group(
    sprites: Dict[str, Image.Image]
) -> Tuple[List[Tuple[str, int, int]], Tuple[int, int]]:
    """
    Build an animal group: a stop sign in the center, with several
    animals (zebra/elephant/giraffe) on a ring of radius ANIMAL_RING_RADIUS.
    Avoid heavy overlap on the ring (light overlap allowed).
    """
    objs: List[Tuple[str, int, int]] = []

    stop = sprites["stopSign"]
    sw, sh = stop.size

    stop_x = 0
    stop_y = 0
    objs.append(("stopSign", stop_x, stop_y))

    cx = stop_x + sw / 2
    cy = stop_y + sh / 2

    animal_names = ["zebra", "elephant", "giraffe"]
    num_animals = random.randint(*ANIMALS_PER_GROUP_RANGE)

    R = ANIMAL_RING_RADIUS
    width_scale = ANIMAL_WIDTH_SCALE
    max_tries = ANIMAL_MAX_TRIES

    placed: List[Tuple[str, float, int, int]] = []

    for _ in range(num_animals):
        name = random.choice(animal_names)
        img = sprites[name]
        w, h = img.size

        half_angle_width = (width_scale * w) / (2.0 * R)

        chosen_angle: Optional[float] = None
        for _ in range(max_tries):
            angle = random.uniform(0.0, 2.0 * math.pi)

            ok = True
            for _, a_prev, w_prev, _ in placed:
                half_prev = (width_scale * w_prev) / (2.0 * R)
                dtheta = abs(angle - a_prev)
                dtheta = min(dtheta, 2.0 * math.pi - dtheta)
                if dtheta < (half_angle_width + half_prev):
                    ok = False
                    break

            if ok:
                chosen_angle = angle
                break

        if chosen_angle is None:
            chosen_angle = random.uniform(0.0, 2.0 * math.pi)

        placed.append((name, chosen_angle, w, h))

    for name, angle, w, h in placed:
        x_center = cx + R * math.cos(angle)
        y_bottom = cy + R * math.sin(angle) + h / 2
        x = int(x_center - w / 2)
        y = int(y_bottom - h)
        objs.append((name, x, y))

    # Normalize to top-left at (0, 0)
    xs = [x for _, x, _ in objs]
    ys = [y for _, _, y in objs]
    xe, ye = [], []
    for (name, x, y) in objs:
        img = sprites[name]
        w, h = img.size
        xe.append(x + w)
        ye.append(y + h)

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xe)
    max_y = max(ye)

    norm_objs: List[Tuple[str, int, int]] = []
    for name, x, y in objs:
        norm_objs.append((name, int(x - min_x), int(y - min_y)))

    group_w = int(max_x - min_x)
    group_h = int(max_y - min_y)
    return norm_objs, (group_w, group_h)


def build_bench_group(
    sprites: Dict[str, Image.Image],
    n_people: int,
) -> Tuple[List[Tuple[str, int, int]], Tuple[int, int]]:
    """
    Build a bench group: a bench with n people located on an arc
    in front of the bench at radius PEOPLE_RING_RADIUS.
    If n_people == 0, we get an empty bench.
    """
    objs: List[Tuple[str, int, int]] = []

    bench = sprites["bench"]
    bw, bh = bench.size

    bench_x = 0
    bench_y = 0
    objs.append(("bench", bench_x, bench_y))

    if n_people <= 0:
        return objs, (bw, bh)

    person = sprites["person"]
    pw, ph = person.size

    cx = bench_x + bw / 2
    cy = bench_y + bh / 2

    R = PEOPLE_RING_RADIUS
    width_scale = PEOPLE_WIDTH_SCALE
    max_tries = PEOPLE_MAX_TRIES

    angle_min = math.pi * BENCH_ANGLE_MIN_FRAC
    angle_max = math.pi * BENCH_ANGLE_MAX_FRAC

    placed: List[Tuple[str, float, int, int]] = []

    for _ in range(n_people):
        w, h = pw, ph
        half_angle_width = (width_scale * w) / (2.0 * R)

        chosen_angle: Optional[float] = None
        for _ in range(max_tries):
            angle = random.uniform(angle_min, angle_max)

            ok = True
            for _, a_prev, w_prev, _ in placed:
                half_prev = (width_scale * w_prev) / (2.0 * R)
                dtheta = abs(angle - a_prev)
                if dtheta < (half_angle_width + half_prev):
                    ok = False
                    break

            if ok:
                chosen_angle = angle
                break

        if chosen_angle is None:
            chosen_angle = random.uniform(angle_min, angle_max)

        placed.append(("person", chosen_angle, w, h))

    for name, angle, w, h in placed:
        x_center = cx + R * math.cos(angle)
        y_bottom = cy + R * math.sin(angle) + h / 2
        x = int(x_center - w / 2)
        y = int(y_bottom - h)
        objs.append((name, x, y))

    # Normalize to top-left (0, 0)
    xs = [x for _, x, _ in objs]
    ys = [y for _, _, y in objs]
    xe, ye = [], []
    for (name, x, y) in objs:
        img = sprites[name]
        w, h = img.size
        xe.append(x + w)
        ye.append(y + h)

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xe)
    max_y = max(ye)

    norm_objs: List[Tuple[str, int, int]] = []
    for name, x, y in objs:
        norm_objs.append((name, int(x - min_x), int(y - min_y)))

    group_w = int(max_x - min_x)
    group_h = int(max_y - min_y)
    return norm_objs, (group_w, group_h)


# ========= Main layout generation on the background =========

def generate_scene(seed: Optional[int] = None, index: int = 0) -> Tuple[Image.Image, Image.Image]:
    """
    Return (bg_full, bg_no_person)
    - bg_full: includes everything
    - bg_no_person: same but with all "person" sprites removed
    """
    if seed is not None:
        random.seed(seed)

    bg_full = Image.open(BACKGROUND_FILE).convert("RGBA")
    bg_no_person = Image.open(BACKGROUND_FILE).convert("RGBA")

    bg_w, bg_h = bg_full.size
    sprites = load_sprites()

    n_animal_groups = random.randint(*ANIMAL_GROUP_COUNT_RANGE)
    n_benches = random.randint(*BENCH_COUNT_RANGE)

    bench_people_counts: List[int] = [
        random.randint(*PEOPLE_PER_BENCH_RANGE) for _ in range(n_benches)
    ]

    group_specs: List[Tuple[str, Optional[int]]] = []
    for _ in range(n_animal_groups):
        group_specs.append(("animal", None))
    for c in bench_people_counts:
        group_specs.append(("bench", c))
    random.shuffle(group_specs)

    built_groups: List[Tuple[List[Tuple[str, int, int]], int, int]] = []
    for kind, arg in group_specs:
        if kind == "animal":
            objs, (gw, gh) = build_animal_group(sprites)
        else:
            n_people = int(arg) if arg is not None else 0
            objs, (gw, gh) = build_bench_group(sprites, n_people)
        built_groups.append((objs, gw, gh))

    total_groups = len(built_groups)
    if total_groups == 0:
        return bg_full, bg_no_person

    rows = GRID_ROWS
    cols = GRID_COLS
    total_cells = rows * cols
    groups_to_place = min(total_groups, total_cells)

    random.shuffle(built_groups)
    chosen_groups = built_groups[:groups_to_place]
    cell_indices = random.sample(range(total_cells), groups_to_place)

    cell_w = (bg_w - 2 * MARGIN) / cols
    cell_h = (bg_h - 2 * MARGIN) / rows
    jitter_scale = GRID_JITTER_SCALE

    for (cell_idx, (objs, gw, gh)) in zip(cell_indices, chosen_groups):
        row = cell_idx // cols
        col = cell_idx % cols

        cx_base = MARGIN + (col + 0.5) * cell_w
        cy_base = MARGIN + (row + 0.5) * cell_h

        jx = random.uniform(-jitter_scale, jitter_scale) * cell_w
        jy = random.uniform(-jitter_scale, jitter_scale) * cell_h

        cx = cx_base + jx
        cy = cy_base + jy

        x = int(cx - gw / 2)
        y = int(cy - gh / 2)

        x = max(MARGIN, min(x, bg_w - MARGIN - gw))
        y = max(MARGIN, min(y, bg_h - MARGIN - gh))

        for name, ox, oy in objs:
            sprite = sprites[name]
            paste_sprite(bg_full, sprite, x + ox, y + oy)
            if name != "person":
                paste_sprite(bg_no_person, sprite, x + ox, y + oy)

    return bg_full, bg_no_person


# ========= Step 2: add randomly rotated bus on blank background =========

def is_patch_blank(
    scene: Image.Image,
    plain_bg: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    diff_threshold: int = BUS_DIFF_THRESHOLD,
) -> bool:
    """
    Compare a patch from the final scene and the pure background.
    If the difference is very small, we treat this patch as "blank".
    """
    patch_scene = scene.crop((x, y, x + w, y + h)).convert("RGB")
    patch_bg = plain_bg.crop((x, y, x + w, y + h)).convert("RGB")

    diff = ImageChops.difference(patch_scene, patch_bg)
    bbox = diff.getbbox()
    if bbox is None:
        return True

    extrema = diff.getextrema()
    max_diff = max(ch[1] for ch in extrema)
    return max_diff <= diff_threshold


def yolo_counts_ok(
    model: YOLO,
    pil_img: Image.Image,
    need_bench: int = NEED_BENCH,
    need_stop: int = NEED_STOP_SIGN,
    conf: float = YOLO_CONF,
    imgsz: int = YOLO_IMGSZ,
) -> bool:
    """
    Run YOLO on a PIL image, and check bench / stop sign counts.
    """
    img = np.array(pil_img.convert("RGB"))  # HWC RGB
    results = model.predict(source=img, verbose=False, conf=conf, imgsz=imgsz)
    r = results[0]

    bench_cnt = 0
    stop_cnt = 0

    for box in r.boxes:
        cls = int(box.cls[0])
        name = model.names.get(cls, str(cls))
        if name == "bench":
            bench_cnt += 1
        elif name == "stop sign":  # COCO name
            stop_cnt += 1

    return (bench_cnt == need_bench) and (stop_cnt == need_stop)


def add_bus_variants_for_one_scene(
    model: YOLO,
    base_full: Image.Image,
    base_no_person: Image.Image,
    stem: str,
    num_variants: int = BUS_VARIANTS_PER_SCENE,
) -> None:
    """
    Create variants with a bus sprite placed on blank background regions.
    Additionally:
      - Randomly select NO_PERSON_VARIANTS_PER_SCENE variants to use base_no_person
      - Resize output to long side 1280
      - Save JPEG
      - Only save if YOLO detects bench=NEED_BENCH and stop sign=NEED_STOP_SIGN
    """
    bus_sprite = Image.open(BUS_FILE).convert("RGBA")
    plain_bg = Image.open(BACKGROUND_FILE).convert("RGB")

    W, H = base_full.size
    used_centers: List[Tuple[float, float]] = []

    no_person_set = set(random.sample(range(num_variants), NO_PERSON_VARIANTS_PER_SCENE))

    for k in range(num_variants):
        base = base_no_person if k in no_person_set else base_full

        if BUS_ROTATE:
            angle = random.uniform(0.0, 360.0)
            rotated = bus_sprite.rotate(angle, expand=True)
        else:
            rotated = bus_sprite.copy()

        ow, oh = rotated.size

        placed = False
        cx = cy = 0.0

        for _ in range(BUS_MAX_PLACEMENT_TRIES):
            x = random.randint(MARGIN, max(MARGIN, W - MARGIN - ow))
            y = random.randint(MARGIN, max(MARGIN, H - MARGIN - oh))

            if not is_patch_blank(base, plain_bg, x, y, ow, oh):
                continue

            cx_try = x + ow / 2.0
            cy_try = y + oh / 2.0

            ok = True
            for (px, py) in used_centers:
                dx = cx_try - px
                dy = cy_try - py
                if math.sqrt(dx * dx + dy * dy) < BUS_MIN_CENTER_DIST:
                    ok = False
                    break

            if not ok:
                continue

            cx, cy = cx_try, cy_try
            used_centers.append((cx, cy))
            placed = True
            break

        if not placed:
            print(f"[warn] cannot find blank region for {stem} variant {k}")
            continue

        out = base.copy()
        out.alpha_composite(rotated, (int(cx - ow / 2.0), int(cy - oh / 2.0)))

        # Resize: long side -> 1280
        w, h = out.size
        scale = 1280 / max(w, h)
        out = out.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)

        # Save JPEG requires RGB
        out = out.convert("RGB")

        # ===== YOLO filter: only save if counts match =====
        if not yolo_counts_ok(model, out):
            continue
        # ===============================================

        out_path = OUTPUT_DIR / f"{stem}_bus_{k:02d}.jpg"
        out.save(out_path, format="JPEG", quality=95)
        print("Saved", out_path)


# ========= Entry point =========

if __name__ == "__main__":
    model = YOLO(YOLO_WEIGHTS)

    for i in range(NUM_BASE_SCENES):
        base_full, base_no_person = generate_scene(index=i)
        add_bus_variants_for_one_scene(
            model=model,
            base_full=base_full,
            base_no_person=base_no_person,
            stem=f"scene_{i:03d}",
            num_variants=BUS_VARIANTS_PER_SCENE,
        )