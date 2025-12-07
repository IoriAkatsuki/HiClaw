# 数据集说明

## 结构建议
- raw/：原始下载或采集数据
- yolo/：转换后的 YOLO 标注（images/、labels/）
- annotations/：中间格式（COCO/LabelMe）

## 路线 A
- 公共：ElectroCom61 等；使用 training/route_a_electro/prepare_electrocom61.py 下载与转换
- 自采：拍摄托盘元器件，统一分辨率与光照；标注类别同 `docs/route-a-design.md`

## 路线 B
- 自采紧固件与异物；保持类别：long_screw、short_screw、nut、washer、foreign_object
- 可选：MVTec AD screw/metal_nut 数据用于异常检测预训练

## 标注规范
- YOLO txt：`class x_center y_center width height`，归一化坐标
- 建议保留数据集版本号与采集日期，确保可复现
