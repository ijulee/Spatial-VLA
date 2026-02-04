#!/usr/bin/env python
"""
run_zoo_bus.py

Use GRAID to generate a VQA dataset on a custom "zoo + bus" image folder.

Usage:
    python run_zoo_bus.py                  # 默认使用 ./zoo_bus_config.json
    python run_zoo_bus.py path/to/config.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T  # 如果没有装 torchvision，需要先 pip 安装

from graid.data.generate_dataset import HuggingFaceDatasetBuilder
from graid.models.Ultralytics import Yolo
from graid.questions.ObjectDetectionQ import ALL_QUESTION_CLASSES

logger = logging.getLogger("zoo_bus")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# =======================
# 0. 顶层 collate_fn（关键修复点）
# =======================

def identity_collate(batch):
    """Simple collate_fn that just returns the list of samples as-is."""
    return batch

# =======================
# 1. 自定义 Dataset
# =======================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class ZooBusDataset(Dataset):
    """
    Minimal custom dataset for a flat image folder.

    __getitem__ 返回:
        (image_tensor, [])
    这样在 GRAID 的 _get_batch_predictions 里会走 tuple 分支，
    ground_truth_labels 虽然为空但不会用到（因为我们有模型预测）。
    """

    def __init__(self, root: Path, resize_longest_side: Optional[int] = None):
        self.root = root
        self.paths: List[Path] = sorted(
            [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        )
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")

        self.resize_longest_side = resize_longest_side
        # 关键改动：保持 0–255 的 uint8，不做 /255 归一化，让 Ultralytics 自己处理
        self.to_tensor = T.PILToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def _resize_keep_aspect(self, img: Image.Image) -> Image.Image:
        if self.resize_longest_side is None:
            return img
        w, h = img.size
        longest = max(w, h)
        if longest <= self.resize_longest_side:
            return img
        scale = self.resize_longest_side / float(longest)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), resample=Image.BILINEAR)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, list]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self._resize_keep_aspect(img)
        tensor = self.to_tensor(img)  # uint8, (C, H, W), 范围 [0,255]
        # 第二个位置留空列表，表示没有 GT 标注
        return tensor, []

# =======================
# 2. 自定义 Builder
# =======================

class ZooBusBuilder(HuggingFaceDatasetBuilder):
    """
    小改版的 HuggingFaceDatasetBuilder：
    - 不用 GRAID 内置的 DatasetLoaderFactory
    - 自己在 _create_data_loader 里构建 DataLoader(ZooBusDataset)
    其它 QA 生成逻辑完全沿用 GRAID。
    """

    def __init__(
        self,
        image_root: Path,
        resize_longest_side: Optional[int],
        *args,
        **kwargs,
    ):
        self._image_root = image_root
        self._resize_longest_side = resize_longest_side
        super().__init__(*args, **kwargs)

    # 覆盖掉父类里对 BDD/NuImages/Waymo 的 transform 选择
    def _get_dataset_transform(self):
        # 我们在自定义 Dataset 里已经做了 to_tensor/resize，这里返回一个空操作即可
        return lambda image, labels: (image, labels)

    # 跳过 DatasetLoaderFactory
    def _init_dataset_loader(self):
        # 我们完全不用它，只是占个位
        self.dataset_loader = None

    # 用我们自己的 Dataset + DataLoader
    def _create_data_loader(self) -> DataLoader:
        dataset = ZooBusDataset(
            root=self._image_root,
            resize_longest_side=self._resize_longest_side,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=identity_collate,  # 顶层函数，避免 lambda 无法 pickle
        )
        return loader

# =======================
# 3. 配置解析 & 模型/问题构建
# =======================

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_models(model_cfgs: List[Dict[str, Any]]) -> List[Any]:
    models: List[Any] = []
    for cfg in model_cfgs:
        backend = cfg["backend"]
        if backend != "ultralytics":
            raise ValueError(f"Only 'ultralytics' backend is handled here, got {backend}")
        # 允许两种写法：model_path 或 model_name
        model_path = cfg.get("model_path") or cfg.get("model_name")
        if model_path is None:
            raise ValueError("Ultralytics model needs 'model_path' or 'model_name'")
        m = Yolo(model_path)
        # 覆盖置信度阈值
        if "confidence_threshold" in cfg:
            m.threshold = float(cfg["confidence_threshold"])
        models.append(m)
    return models

def build_questions(question_cfgs: List[Dict[str, Any]]):
    questions = []
    for qc in question_cfgs:
        name = qc["name"]
        params = qc.get("params", {})
        cls = ALL_QUESTION_CLASSES.get(name)
        if cls is None:
            raise ValueError(f"Unknown question type in config: {name}")
        questions.append(cls(**params))
    return questions

# =======================
# 4. 主入口
# =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="zoo_bus_config.json",
        help="Path to JSON config file (default: zoo_bus_config.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    dataset_name: str = cfg.get("dataset_name", "zoo_bus")
    split: str = cfg.get("split", "train")

    image_root = Path(cfg["image_root"]).expanduser().resolve()
    save_path = Path(cfg["save_path"]).expanduser().resolve()

    models = build_models(cfg["models"])
    allowable_set = cfg.get("allowable_set")
    question_cfgs = cfg.get("questions", [])
    questions = build_questions(question_cfgs)

    # 如果 config 里没写 transforms，这里会变成 {}，resize_longest_side = None
    resize_longest_side: Optional[int] = None
    transforms_cfg = cfg.get("transforms", {})
    if isinstance(transforms_cfg, dict):
        resize_longest_side = transforms_cfg.get("resize_longest_side")

    conf_threshold = float(cfg.get("confidence_threshold", 0.2))
    batch_size = int(cfg.get("batch_size", 1))
    num_workers = int(cfg.get("num_workers", 4))   # 如遇到别的多进程问题可以先改成 0
    qa_workers = int(cfg.get("qa_workers", 4))
    num_samples = cfg.get("num_samples")  # 可以是 None

    upload_to_hub = bool(cfg.get("upload_to_hub", False))
    hub_repo_id = cfg.get("hub_repo_id") or None
    hub_private = bool(cfg.get("hub_private", True))

    logger.info(f"Image root: {image_root}")
    logger.info(f"Saving dataset to: {save_path}")
    logger.info(f"Using {len(models)} detection model(s)")
    logger.info(f"Questions: {[q.__class__.__name__ for q in questions]}")

    # ====== 构建自定义 Builder ======
    builder = ZooBusBuilder(
        image_root=image_root,
        resize_longest_side=resize_longest_side,
        dataset_name=dataset_name,
        split=split,
        models=models,
        use_wbf=bool(cfg.get("use_wbf", False)),
        wbf_config=cfg.get("wbf_config"),
        conf_threshold=conf_threshold,
        batch_size=batch_size,
        device=None,  # 让 GRAID 自动挑设备
        allowable_set=allowable_set,
        question_configs=question_cfgs,
        num_workers=num_workers,
        qa_workers=qa_workers,
        num_samples=num_samples,
        save_path=str(save_path),
        use_original_filenames=True,
        filename_prefix=cfg.get("filename_prefix", "zoo_bus"),
    )

    # ====== 生成 HF DatasetDict ======
    dataset_dict = builder.build()

    # 本地保存到磁盘
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving DatasetDict to disk at {save_path} ...")
    dataset_dict.save_to_disk(str(save_path))
    logger.info("✅ Finished generating zoo_bus VQA dataset")

    # 可选：上传到 Hub
    if upload_to_hub:
        if not hub_repo_id:
            raise ValueError("hub_repo_id is required when upload_to_hub=True")
        from huggingface_hub import create_repo

        logger.info(f"Uploading dataset to HuggingFace Hub: {hub_repo_id}")
        create_repo(
            hub_repo_id,
            repo_type="dataset",
            private=hub_private,
            exist_ok=True,
        )
        dataset_dict.push_to_hub(
            repo_id=hub_repo_id,
            private=hub_private,
            commit_message=f"Upload {dataset_name} {split} dataset",
            max_shard_size="5GB",
        )
        logger.info("✅ Pushed dataset to HuggingFace Hub")

if __name__ == "__main__":
    main()