#!/usr/bin/env python3
"""
ElectroCom61 数据集准备脚本占位：
- 解压/组织数据到 datasets/yolo/route_a
- 生成 train/val 划分列表
- 保留标签为 YOLO txt
"""
import argparse
import pathlib
import random
import shutil
from typing import List

RANDOM_SEED = 42

def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_images_and_labels(src_images: pathlib.Path, src_labels: pathlib.Path, dst_root: pathlib.Path) -> List[pathlib.Path]:
    """将 images 与 labels 复制到目标目录，返回图像路径列表。"""
    ensure_dir(dst_root / "images")
    ensure_dir(dst_root / "labels")
    image_paths = []
    for img in src_images.glob("**/*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        rel = img.relative_to(src_images)
        label = (src_labels / rel).with_suffix(".txt")
        dst_img = dst_root / "images" / rel
        dst_lbl = dst_root / "labels" / rel.with_suffix(".txt")
        ensure_dir(dst_img.parent)
        ensure_dir(dst_lbl.parent)
        shutil.copy2(img, dst_img)
        if label.exists():
            shutil.copy2(label, dst_lbl)
        else:
            print(f"[WARN] 未找到标签文件: {label}")
        image_paths.append(dst_img)
    return image_paths


def write_split_lists(image_paths: List[pathlib.Path], dst_root: pathlib.Path, train_ratio: float, val_ratio: float) -> None:
    random.seed(RANDOM_SEED)
    random.shuffle(image_paths)
    n = len(image_paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_list = image_paths[:n_train]
    val_list = image_paths[n_train:n_train + n_val]
    test_list = image_paths[n_train + n_val:]

    def dump(name: str, paths: List[pathlib.Path]):
        with open(dst_root / name, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(str(p.resolve()) + "\n")
        print(f"[INFO] 写入 {name}: {len(paths)} 张")

    dump("train.txt", train_list)
    dump("val.txt", val_list)
    dump("test.txt", test_list)


def main():
    parser = argparse.ArgumentParser(description="准备 ElectroCom61 数据集为 YOLO 格式")
    parser.add_argument("--src-images", type=pathlib.Path, required=True, help="原始 images 目录")
    parser.add_argument("--src-labels", type=pathlib.Path, required=True, help="原始 labels 目录 (YOLO txt 或需先转换)")
    parser.add_argument("--dst-root", type=pathlib.Path, default=pathlib.Path("../../datasets/yolo/route_a"), help="输出根目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    args = parser.parse_args()

    ensure_dir(args.dst_root)
    image_paths = copy_images_and_labels(args.src_images, args.src_labels, args.dst_root)
    if not image_paths:
        print("[ERROR] 未找到任何图像，请检查 --src-images 路径与后缀")
        return
    write_split_lists(image_paths, args.dst_root, args.train_ratio, args.val_ratio)
    print(f"[DONE] 数据已整理到 {args.dst_root}")


if __name__ == "__main__":
    main()
