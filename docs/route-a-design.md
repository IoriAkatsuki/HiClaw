# 路线 A：电子元器件激光可视分拣设计稿

## 场景与目标
- 顶视相机 + 托盘元器件；YOLO 检测后通过振镜标注
- 目标类别：resistor/color_resistor、capacitor、diode、transistor、ic、led、other

## 数据与标注
- 公共数据：ElectroCom61 等，转换为 YOLO 格式
- 自采数据：实物拍摄 + LabelMe/LabelImg 标注；推荐保持 640×640 训练尺寸

## 模型与训练
- 方案：MindYOLO YOLOv5n/YOLOv8n；可选色环电阻子模型
- 脚本占位：
  - training/route_a_electro/prepare_electrocom61.py
  - training/route_a_electro/train_yolo_route_a.py
  - training/route_a_electro/export_mindir_route_a.py

## 端侧流程
- edge/common：相机采集、MindSpore Lite 推理封装、标定、振镜控制
- edge/route_a_app：业务逻辑（检测→坐标映射→图形生成→发送振镜；可选 depth/语音接口）

## 待办清单
- [ ] 整理数据集说明与下载脚本
- [ ] 补充训练/导出脚本模板
- [ ] 完成标定工具与振镜通信封装
- [ ] 端侧 demo：实时检测 + 激光标注 + Web UI
