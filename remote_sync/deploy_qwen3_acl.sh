#!/bin/bash
#
# Qwen3-8B ACL Deployment Script
# Automated deployment for Ascend 310B (OrangePi AIpro)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
WORK_DIR="/home/HwHiAiUser/ICT"
MODEL_DIR="$WORK_DIR/models/qwen3-8b"
ONNX_DIR="$WORK_DIR/qwen3_onnx"
OM_FILE="$WORK_DIR/qwen3_fp16.om"

echo "========================================"
echo "Qwen3-8B ACL Deployment"
echo "Ascend 310B (OrangePi AIpro)"
echo "========================================"

cd "$WORK_DIR"

# Step 1: Check if model is downloaded
echo -e "\n${YELLOW}[Step 1/5]${NC} Checking Qwen3-8B model..."
if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/config.json" ]; then
    echo -e "${RED}✗${NC} Model not found at $MODEL_DIR"
    echo "Please run: python3 download_qwen3_8b.py"
    exit 1
fi
echo -e "${GREEN}✓${NC} Model found: $MODEL_DIR"

# Step 2: Export to ONNX
echo -e "\n${YELLOW}[Step 2/5]${NC} Exporting Qwen3-8B to ONNX..."
if [ -f "$ONNX_DIR/qwen3_model.onnx" ]; then
    echo -e "${YELLOW}!${NC} ONNX file already exists. Skipping export."
else
    python3 export_qwen3_onnx.py
    if [ ! -f "$ONNX_DIR/qwen3_model.onnx" ]; then
        echo -e "${RED}✗${NC} ONNX export failed!"
        exit 1
    fi
fi
echo -e "${GREEN}✓${NC} ONNX model ready: $ONNX_DIR/qwen3_model.onnx"

# Step 3: Convert ONNX to OM using ATC
echo -e "\n${YELLOW}[Step 3/5]${NC} Converting ONNX to OM (Ascend format)..."
if [ -f "$OM_FILE" ]; then
    echo -e "${YELLOW}!${NC} OM file already exists. Skipping conversion."
    read -p "Overwrite existing OM file? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$OM_FILE"
        bash convert_qwen3_to_om.sh
    fi
else
    bash convert_qwen3_to_om.sh
fi

if [ ! -f "$OM_FILE" ]; then
    echo -e "${RED}✗${NC} OM conversion failed!"
    exit 1
fi
echo -e "${GREEN}✓${NC} OM model ready: $OM_FILE"
ls -lh "$OM_FILE"

# Step 4: Test ACL inference
echo -e "\n${YELLOW}[Step 4/5]${NC} Testing ACL inference..."
python3 qwen3_acl_inference.py
if [ $? -ne 0 ]; then
    echo -e "${RED}✗${NC} ACL inference test failed!"
    exit 1
fi
echo -e "${GREEN}✓${NC} ACL inference working!"

# Step 5: Launch Web UI
echo -e "\n${YELLOW}[Step 5/5]${NC} Launching Web UI..."
echo "Starting Streamlit on port 8501..."
echo "Access at: http://$(hostname -I | awk '{print $1}'):8501"
echo ""

# Kill existing streamlit if running
pkill -f streamlit || true

# Start in background
nohup streamlit run qwen3_webui_acl.py --server.port 8501 --server.address 0.0.0.0 > webui_acl.log 2>&1 &

sleep 3

if pgrep -f streamlit > /dev/null; then
    echo -e "${GREEN}✓${NC} Web UI started successfully!"
    echo "Logs: tail -f $WORK_DIR/webui_acl.log"
else
    echo -e "${RED}✗${NC} Web UI failed to start. Check logs."
    cat webui_acl.log
    exit 1
fi

echo ""
echo "========================================"
echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo "========================================"
echo "Model: Qwen3-8B"
echo "Backend: ACL (Ascend 310B NPU)"
echo "Web UI: http://$(hostname -I | awk '{print $1}'):8501"
echo "========================================"
