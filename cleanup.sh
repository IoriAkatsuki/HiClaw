#!/bin/bash
# ICT目录清理脚本

set -e
cd /home/oasis/Documents/ICT

echo "========================================"
echo "ICT 目录清理工具"
echo "========================================"
echo ""

# 统计当前占用
TOTAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)
echo "当前目录总大小: $TOTAL_SIZE"
echo ""

# 1. 清理Python缓存
echo "1. 清理Python缓存文件..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -exec rm -f {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
echo "✓ Python缓存已清理"

# 2. 清理日志文件
echo ""
echo "2. 清理日志文件..."
find ./remote_sync -name "*.log" -exec rm -f {} + 2>/dev/null || true
echo "✓ 日志文件已清理"

# 3. 清理已解压的压缩包
echo ""
echo "3. 处理压缩包..."
if [ -f "ElectroCom61 A Multiclass Dataset for Detection of Electronic Components.zip" ]; then
    if [ -d "ElectroCom61 A Multiclass Dataset for Detection of Electronic Components" ]; then
        echo "  数据集已解压，删除压缩包 (138MB)..."
        rm -f "ElectroCom61 A Multiclass Dataset for Detection of Electronic Components.zip"
        echo "  ✓ 已删除: ElectroCom61...zip"
    fi
fi

if [ -f "Ascend-hdk-310b-npu-soc_25.3.rc1_linux-aarch64.zip" ]; then
    if [ -d "Ascend-hdk-310b-npu-soc_25.3.rc1_linux-aarch64" ]; then
        echo "  Ascend SDK已解压，删除压缩包 (198MB)..."
        rm -f "Ascend-hdk-310b-npu-soc_25.3.rc1_linux-aarch64.zip"
        echo "  ✓ 已删除: Ascend-hdk...zip"
    fi
fi

# 4. 清理临时目录
echo ""
echo "4. 清理临时目录..."
if [ -d "tmp_samples" ]; then
    rm -rf tmp_samples
    echo "  ✓ 已删除: tmp_samples/ (2.1MB)"
fi

# 5. 清理临时图片
echo ""
echo "5. 清理临时图片..."
if [ -f "粘贴的图像.png" ]; then
    rm -f "粘贴的图像.png"
    echo "  ✓ 已删除: 粘贴的图像.png"
fi

# 6. 清理训练runs目录（可选）
echo ""
echo "6. 训练输出目录 (runs/) - 48MB"
read -p "   是否清理训练输出目录? (y/N): " clean_runs
if [[ "$clean_runs" =~ ^[Yy]$ ]]; then
    rm -rf runs
    echo "  ✓ 已删除: runs/"
else
    echo "  - 保留: runs/"
fi

# 7. 处理操作系统镜像（询问）
echo ""
echo "7. 操作系统镜像文件"
echo "   以下文件占用大量空间，但可能需要用于系统安装："
echo ""
ls -lh *.img *.img.xz 2>/dev/null | awk '{print "   ", $9, "("$5")"}'
echo ""
echo "   建议: 如已完成系统安装，可移动到备份目录或外部存储"
read -p "   是否删除所有镜像文件? (y/N): " clean_images
if [[ "$clean_images" =~ ^[Yy]$ ]]; then
    rm -f *.img *.img.xz
    echo "  ✓ 已删除所有镜像文件"
else
    echo "  - 保留镜像文件"
fi

# 8. 处理模型文件
echo ""
echo "8. ONNX/PT模型文件"
ls -lh *.onnx *.pt 2>/dev/null | awk '{print "   ", $9, "("$5")"}'
echo ""
read -p "   是否删除本地训练模型文件? (板卡上已有.om文件) (y/N): " clean_models
if [[ "$clean_models" =~ ^[Yy]$ ]]; then
    rm -f *.onnx *.pt
    echo "  ✓ 已删除模型文件"
else
    echo "  - 保留模型文件"
fi

# 9. 清理backup目录
echo ""
echo "9. 备份目录"
if [ -d "backup_modules_rootfs_20241204" ]; then
    echo "   backup_modules_rootfs_20241204/ (183MB)"
    read -p "   是否删除备份目录? (y/N): " clean_backup
    if [[ "$clean_backup" =~ ^[Yy]$ ]]; then
        rm -rf backup_modules_rootfs_20241204
        echo "  ✓ 已删除备份目录"
    else
        echo "  - 保留备份目录"
    fi
fi

if [ -d "backup_hwhiaiuser_rootfs_20241204" ]; then
    echo "   backup_hwhiaiuser_rootfs_20241204/ (14MB)"
    read -p "   是否删除备份目录? (y/N): " clean_backup2
    if [[ "$clean_backup2" =~ ^[Yy]$ ]]; then
        rm -rf backup_hwhiaiuser_rootfs_20241204
        echo "  ✓ 已删除备份目录"
    else
        echo "  - 保留备份目录"
    fi
fi

# 10. 统计清理后大小
echo ""
echo "========================================"
echo "清理完成！"
echo "========================================"
FINAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)
echo "清理后大小: $FINAL_SIZE"
echo ""
echo "目录结构:"
du -sh */ 2>/dev/null | sort -hr | head -10
echo ""
echo "建议: remote_sync 目录(91G)主要是板卡同步数据"
echo "      可定期清理或使用 rsync --delete 同步最新状态"
