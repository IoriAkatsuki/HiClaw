#!/bin/bash
# YOLO 检测阈值调整脚本
# 用法: ./adjust_threshold.sh 0.55

if [ $# -eq 0 ]; then
    echo "当前阈值设置："
    ssh HwHiAiUser@ict.local "grep 'conf_thres=' /home/HwHiAiUser/ICT/edge/route_a_app/rtsp_detect_aipp.py | grep 'def postprocess'"
    echo ""
    echo "用法: $0 <新阈值>"
    echo "示例: $0 0.52  # 降低阈值，增加检测（可能增加误检）"
    echo "      $0 0.60  # 提高阈值，减少误检（可能漏检）"
    echo ""
    echo "推荐范围: 0.50-0.70"
    exit 0
fi

NEW_THRES=$1
echo "正在将阈值设置为 $NEW_THRES ..."

# 修改阈值
ssh HwHiAiUser@ict.local "sed -i 's/conf_thres=[0-9.]\+/conf_thres=$NEW_THRES/' /home/HwHiAiUser/ICT/edge/route_a_app/rtsp_detect_aipp.py"

# 重启服务
echo "重启检测服务..."
ssh HwHiAiUser@ict.local 'pkill -f rtsp_detect_aipp.py || true'
sleep 2
ssh HwHiAiUser@ict.local "cd /home/HwHiAiUser/ICT && source /usr/local/Ascend/ascend-toolkit/set_env.sh && export PYTHONPATH=/home/HwHiAiUser/ICT/pybind_venc/build:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages: && nohup python3 /home/HwHiAiUser/ICT/edge/route_a_app/rtsp_detect_aipp.py --model /home/HwHiAiUser/ICT/runs/detect/train_electro61/weights/yolov8_electro61_aipp.om --data-yaml '/home/HwHiAiUser/ICT/config/electro61.yaml' > /home/HwHiAiUser/ICT/rtsp_aipp.log 2>&1 &"

echo "完成！等待 3 秒后查看状态..."
sleep 3
ssh HwHiAiUser@ict.local 'cat /home/HwHiAiUser/ICT/webui_http/state.json' | python3 -m json.tool 2>/dev/null | grep -E '"(fps|cand|kept|max_score)"'
