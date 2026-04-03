# 路线 A 端侧应用

职责：电子元器件检测→坐标映射→激光打标→结果可视化/语音接口。

## 建议模块
- pipeline.py：主循环（采集→推理→后处理→振镜指令）
- postprocess.py：NMS、类别映射、depth 辅助判定
- shapes.py：类别→激光图形映射
- service_api.py：与 Web UI/语音 agent 的数据交换
  
## 快速跑通（摄像头实时检测 + WebUI）
1. 准备 MindIR（选择其一）  
   - 使用 mindyolo 导出：`python training/route_a_electro/export_mindir_route_a.py --config training/route_a_electro/configs/route_a_yolov5n.yaml --checkpoint runs/detect/train_electro61/weights/best.pt --file-name runs/detect/train_electro61/weights/yolov8_electro61`  
   - 或使用 MindSpore Lite 转换 ONNX：`converter_lite --fmk=ONNX --modelFile=runs/detect/train_electro61/weights/best.onnx --outputFile=runs/detect/train_electro61/weights/yolov8_electro61 --device=Ascend`
2. 安装依赖：`pip install fastapi uvicorn pyyaml opencv-python numpy mindspore-lite`  
3. 启动服务：  
   ```bash
   python edge/route_a_app/pipeline.py \
     --mindir runs/detect/train_electro61/weights/yolov8_electro61.mindir \
     --data-yaml "ElectroCom61 A Multiclass Dataset for Detection of Electronic Components/ElectroCom-61_v2/data.yaml" \
     --cam /dev/video4 --host 0.0.0.0 --port 8000
   ```
4. 浏览器访问 `http://<设备IP>:8000` 查看视频流、检测框与调试 JSON；`/api/state` 可供上位机轮询。

## 待办
- [ ] 业务规则与图形映射表
- [ ] 振镜/语音接口集成测试
