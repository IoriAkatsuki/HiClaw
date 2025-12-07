# 标定工具

目标：完成相机像素坐标→物理坐标→振镜坐标的映射标定。

## 功能规划
- 振镜打点阵，自动识别像素坐标
- 计算 Homography 或 3D 外参，写入 edge/common/hw_config.yaml
- 可视化误差与保存标定报告

## TODO
- [ ] 设计 UI（PyQt/简易 Web 均可）
- [ ] 实现点阵检测与矩阵求解
- [ ] 集成到 edge/common/calibration 调用
