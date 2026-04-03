# 路线 B 训练端

包含紧固件检测与可选异常检测的训练/导出脚本。

## 目录规划
- prepare_fastener_dataset.py：整理自采数据为 YOLO 标注
- prepare_mvtec_screw.py：导出 MVTec AD 相关类（可选）
- train_yolo_route_b.py：MindYOLO 检测训练
- train_patchcore_fastener.py：异常检测预训练（可选）
- eval_route_b.py：检测/异常检测评估
- export_mindir_route_b.py：导出 MindIR + 转 OM

## TODO
- [ ] 编写脚本骨架与默认超参
- [ ] 记录数据集路径与版本
