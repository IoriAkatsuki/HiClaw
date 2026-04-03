#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 MindSpore Lite 的 YOLOv8 推理封装，面向 Ascend NPU。
特性：
- 加载 MindIR 模型并绑定 Ascend 设备；
- 简单预处理（resize/归一化/NCHW）；
- 后处理包含置信度过滤与 NMS；
- 支持从 data.yaml 读取类别名；未找到则回退类别编号。
"""
from pathlib import Path
from typing import List, Optional, Tuple
import time

import cv2
import numpy as np
import yaml

try:
    from mindspore_lite import AscendDeviceInfo, Context, Model, ModelType
except ImportError as exc:
    raise ImportError("缺少 mindspore-lite，请先安装：pip install mindspore-lite") from exc


def load_class_names(data_yaml: Path) -> List[str]:
    """从 YOLO data.yaml 读取类别列表，若失败则返回空列表。"""
    try:
        with data_yaml.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", [])
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
    except Exception:
        pass
    return []


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """纯 numpy NMS，输入为 xyxy 与分数。"""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


class Yolov8MsInfer:
    """MindSpore Lite YOLOv8 推理包装。"""

    def __init__(
        self,
        mindir_path: Path,
        device_id: int = 0,
        input_size: int = 640,
        conf_thr: float = 0.3,
        iou_thr: float = 0.45,
        num_classes: int = 61,
        data_yaml: Optional[Path] = None,
    ):
        self.mindir_path = Path(mindir_path)
        self.device_id = device_id
        self.input_size = input_size
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.num_classes = num_classes
        self.class_names = load_class_names(data_yaml) if data_yaml else []

        if not self.mindir_path.exists():
            raise FileNotFoundError(f"未找到 MindIR 模型: {self.mindir_path}")

        ctx = Context()
        ascend_info = AscendDeviceInfo(device_id=device_id)
        # Ascend Lite 依赖 ge provider
        ascend_info.provider = "ge"
        ctx.append_device_info(ascend_info)

        self.model = Model()
        build_ok = self.model.build_from_file(str(self.mindir_path), ModelType.MINDIR, ctx)
        if not build_ok:
            raise RuntimeError(f"构建 MindIR 模型失败: {self.mindir_path}")

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """BGR HWC -> NCHW，保持等比缩放到固定尺寸。"""
        h0, w0 = frame.shape[:2]
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # C,H,W
        img = np.expand_dims(img, 0)  # 1,C,H,W
        scale_x = w0 / float(self.input_size)
        scale_y = h0 / float(self.input_size)
        return img, scale_x, scale_y

    def _postprocess(
        self, preds: np.ndarray, scale_x: float, scale_y: float
    ) -> List[dict]:
        """解析输出并执行 NMS。支持 [1, N, 4+cls] 或 [1, 4+cls, N]。"""
        if preds.ndim != 3 or preds.shape[0] != 1:
            return []
        # 兼容两种布局
        if preds.shape[1] == self.num_classes + 4:
            pred = preds[0]  # N x (4+cls)
        elif preds.shape[2] == self.num_classes + 4:
            pred = np.transpose(preds[0], (1, 0))  # N x (4+cls)
        else:
            return []

        boxes_xyxy = []
        scores = []
        cls_ids = []
        for row in pred:
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            score = float(cls_scores[cls_id])
            if score < self.conf_thr:
                continue
            x, y, w, h = row[:4]
            x1 = (x - w * 0.5) * scale_x
            y1 = (y - h * 0.5) * scale_y
            x2 = (x + w * 0.5) * scale_x
            y2 = (y + h * 0.5) * scale_y
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(score)
            cls_ids.append(cls_id)

        if not boxes_xyxy:
            return []
        boxes = np.array(boxes_xyxy, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        keep_indices = nms_xyxy(boxes, scores_np, self.iou_thr)

        results: List[dict] = []
        for idx in keep_indices:
            c = cls_ids[idx]
            results.append(
                {
                    "cls": c,
                    "label": self.class_names[c] if self.class_names else f"cls{c}",
                    "score": float(scores_np[idx]),
                    "box": [float(v) for v in boxes[idx].tolist()],
                }
            )
        return results

    def infer(self, frame: np.ndarray) -> Tuple[List[dict], float]:
        """执行推理，返回检测结果与推理耗时（毫秒）。"""
        input_tensor, scale_x, scale_y = self._preprocess(frame)
        start = time.time()
        outputs = self.model.predict([input_tensor])
        ms = (time.time() - start) * 1000.0
        if not outputs:
            return [], ms
        preds = outputs[0].get_data_to_numpy()
        dets = self._postprocess(preds, scale_x, scale_y)
        return dets, ms


__all__ = ["Yolov8MsInfer", "load_class_names"]
