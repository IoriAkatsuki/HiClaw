# 路线 A 训练端

包含数据准备、训练、导出与评估脚本。

## 目录规划
- configs/route_a_yolov5n.yaml：示例配置（路径/超参/类别）
- prepare_electrocom61.py：复制/划分公共数据集（需提供已转换 YOLO 标签）
- prepare_custom_dataset.py：自采数据整理（LabelMe→YOLO 占位转换）
- train_yolo_route_a.py：封装调用 `mindyolo.tools.train`
- eval_route_a.py：封装调用 `mindyolo.tools.eval`
- export_mindir_route_a.py：封装调用 `mindyolo.tools.export`
- convert_to_ascend_om.sh：ATC 转 OM

## TODO
- [ ] 编写上述脚本骨架并补充配置参数
- [ ] 记录依赖版本（MindSpore、MindYOLO、Python）
- [ ] 完善 LabelMe→YOLO 转换逻辑；支持 COCO/其他来源
- [ ] 在 Ascend 310B 上验证训练与导出链路
