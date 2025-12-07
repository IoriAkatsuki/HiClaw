# ICT 目录清理报告

## 已清理的文件

✓ Python缓存 (__pycache__, *.pyc)
✓ 日志文件 (remote_sync/*.log)
✓ ElectroCom61 数据集压缩包 (138MB) - 已解压，删除压缩包
✓ Ascend SDK 压缩包 (198MB) - 已解压，删除压缩包
✓ 临时文件 (tmp_samples/, 粘贴的图像.png)

**节省空间:** ~340MB

## 剩余大文件（建议手动处理）

### 1. 操作系统镜像 (~37GB)
```
25GB  opiaipro_20t_openEuler22.03_desktop_aarch64_20250909.img
6.5GB opiaipro_20t_openEuler22.03_desktop_aarch64_20250909.img.xz
6.1GB opiaipro_ubuntu22.04_desktop_aarch64_20250925.img.xz
```
**建议:**
- 如果系统已经安装完成，可以删除或移动到外部存储
- 使用命令: `rm -f *.img *.img.xz`

### 2. remote_sync 目录 (91GB)
```
91GB  remote_sync/  # 板卡rsync同步数据
```
**建议:**
- 这是正常的同步目录，包含板卡上的所有代码和数据
- 可以定期使用 `rsync --delete` 清理旧文件
- 或者设置.gitignore排除不需要的大文件

### 3. 备份目录 (~197MB)
```
183MB backup_modules_rootfs_20241204/
14MB  backup_hwhiaiuser_rootfs_20241204/
```
**建议:**
- 如果确认备份不再需要，可删除
- 使用命令: `rm -rf backup_*`

### 4. 训练输出 (48MB)
```
48MB  runs/  # YOLO训练日志和权重
```
**建议:**
- 如果不需要查看训练历史，可删除
- 使用命令: `rm -rf runs/`

### 5. 模型文件 (~26MB)
```
13MB  yolov8n-pose.onnx
6.6MB yolov8n-pose.pt
6.3MB yolov8n.pt
```
**建议:**
- 板卡上已有转换后的.om文件
- 本地开发机可保留用于重新转换
- 如需节省空间可删除: `rm -f *.onnx *.pt`

## 快速清理脚本

已创建交互式清理脚本: `/home/oasis/Documents/ICT/cleanup.sh`

使用方法:
```bash
cd /home/oasis/Documents/ICT
./cleanup.sh
```

## 当前目录占用

```
总大小: 127GB

主要占用:
91GB   remote_sync/          # 板卡同步数据
25GB   系统镜像 (openEuler)
6.5GB  系统镜像压缩包
6.1GB  系统镜像压缩包 (Ubuntu)
198MB  Ascend SDK
183MB  backup目录
149MB  ElectroCom61数据集
48MB   runs/
```

## 建议的清理命令

如果确认不需要，可以执行:

```bash
cd /home/oasis/Documents/ICT

# 删除操作系统镜像 (节省 ~37GB)
rm -f *.img *.img.xz

# 删除备份目录 (节省 ~197MB)
rm -rf backup_*

# 删除训练输出 (节省 48MB)
rm -rf runs/

# 删除本地模型文件 (节省 ~26MB)
rm -f *.onnx *.pt

# 清理remote_sync中的旧文件（谨慎）
# 可以使用rsync --delete重新同步最新状态
```

## 注意事项

1. **remote_sync** 是最大的目录(91GB)，包含板卡的完整副本
   - 可以设置rsync只同步需要的目录
   - 例如: `rsync -avz --exclude 'samples/' HwHiAiUser@ict.local:~/ICT/ remote_sync/`

2. **系统镜像** 如果还需要重新安装系统，建议保留或移动到外部存储

3. **备份目录** 确认不需要后再删除

4. **模型文件** 如果需要重新转换或训练，建议保留

## 预计可节省空间

- 保守估计: ~340MB (已完成)
- 删除镜像: +37GB
- 删除备份: +197MB
- 删除runs: +48MB
- 删除模型: +26MB

**总计可节省:** ~38GB

---

**生成时间:** 2025-12-07
