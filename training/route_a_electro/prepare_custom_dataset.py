#!/usr/bin/env python3
"""
自采数据整理脚本占位：
- 支持从 LabelMe/COCO 转 YOLO（当前仅示意，需按数据格式补充）
- 输出至 datasets/yolo/route_a
"""
import argparse
import pathlib
import shutil
from typing import List

SUPPORTED_LABELME_SUFFIX = {".json"}


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_images(src: pathlib.Path) -> List[pathlib.Path]:
    return [p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def convert_labelme_to_yolo(labelme_file: pathlib.Path, dst_label: pathlib.Path, class_map: dict) -> None:
    """占位转换函数，请根据实际标注格式补充。"""
    # TODO: 解析 labelme json，生成 YOLO txt: class x_center y_center w h (归一化)
    # 当前写入空文件作为占位，避免训练时报错。
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    if not dst_label.exists():
        dst_label.touch()


def main():
    parser = argparse.ArgumentParser(description="整理自采数据为 YOLO 格式")
    parser.add_argument("--src", type=pathlib.Path, required=True, help="含 images/ 与 labelme 标注的目录")
    parser.add_argument("--dst-root", type=pathlib.Path, default=pathlib.Path("../../datasets/yolo/route_a"), help="输出根目录")
    parser.add_argument("--class-map", type=str, default="resistor:0,color_resistor:1,capacitor:2,diode:3,transistor:4,ic:5,led:6,other:7", help="类别映射，格式 name:id 逗号分隔")
    args = parser.parse_args()

    class_map = {item.split(":")[0]: int(item.split(":")[1]) for item in args.class_map.split(",")}
    images = collect_images(args.src / "images")
    ensure_dir(args.dst_root / "images")
    ensure_dir(args.dst_root / "labels")

    for img in images:
        rel = img.relative_to(args.src / "images")
        dst_img = args.dst_root / "images" / rel
        dst_lbl = args.dst_root / "labels" / rel.with_suffix(".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_img)
        labelme_file = (args.src / "labels" / rel).with_suffix(".json")
        if labelme_file.exists():
            convert_labelme_to_yolo(labelme_file, dst_lbl, class_map)
        else:
            print(f"[WARN] 未找到标注 {labelme_file}，生成空标签")
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.touch()

    print(f"[DONE] 已复制 {len(images)} 张图片到 {args.dst_root}")
    print("[TODO] 根据实际标注格式完善 convert_labelme_to_yolo")


if __name__ == "__main__":
    main()
