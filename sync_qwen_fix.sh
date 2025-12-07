#!/bin/bash
# Sync fixed Qwen3 files to remote board

REMOTE_HOST="HwHiAiUser@ict.local"
REMOTE_PATH="/home/HwHiAiUser/ICT"
LOCAL_PATH="/home/oasis/Documents/ICT/remote_sync"

echo "==================================="
echo "Syncing Qwen3 Fixed Files"
echo "==================================="

# Upload fixed inference engine
echo "[1/2] Uploading qwen3_acl_inference.py..."
scp "${LOCAL_PATH}/qwen3_acl_inference.py" "${REMOTE_HOST}:${REMOTE_PATH}/" || {
    echo "Failed to upload inference engine"
    exit 1
}

# Upload fixed WebUI
echo "[2/2] Uploading qwen3_webui_acl.py..."
scp "${LOCAL_PATH}/qwen3_webui_acl.py" "${REMOTE_HOST}:${REMOTE_PATH}/" || {
    echo "Failed to upload WebUI"
    exit 1
}

echo ""
echo "==================================="
echo "Sync completed successfully!"
echo "==================================="
echo ""
echo "Next steps on remote server:"
echo "1. Kill existing WebUI process:"
echo "   pkill -f qwen3_webui_acl.py"
echo ""
echo "2. Restart WebUI:"
echo "   cd /home/HwHiAiUser/ICT"
echo "   streamlit run qwen3_webui_acl.py --server.port 8502"
echo ""
