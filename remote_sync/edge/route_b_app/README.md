# 路线 B 端侧应用

职责：紧固件/异物检测→坐标映射→激光打标→异常高亮→结果发布。

## 建议模块
- pipeline.py：主循环
- postprocess.py：类别与异常检测融合，规则判定堆叠/长度
- shapes.py：long_screw/short_screw/nut/washer/foreign_object 图形映射
- service_api.py：向 Web UI/语音 agent 发布结果

## 待办
- [ ] 接入 edge/common 推理与标定
- [ ] 异常检测结果融合与图形生成
- [ ] demo 启动脚本与配置
