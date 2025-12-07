#!/usr/bin/env bash
# 在 Orange Pi AI Pro (Ascend 310B) 上编译 fastllm（Ascend 后端）。
# 需提前刷好官方镜像并确保 /usr/local/Ascend/ascend-toolkit/set_env.sh 存在。

set -euo pipefail

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "[WARN] 未检测到 npu-smi，请确认已安装 Ascend 驱动与 CANN。"
fi

echo "[INFO] 加载 Ascend 环境变量 ..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "[INFO] 安装编译依赖 (若已安装会跳过) ..."
sudo apt-get update
sudo apt-get install -y cmake g++ python3-dev git

WORKDIR="$(pwd)"
SRC_DIR="${WORKDIR}/fastllm_src"

if [ ! -d "${SRC_DIR}" ]; then
  echo "[INFO] 克隆 fastllm 源码 ..."
  git clone --depth 1 https://github.com/ztxz16/fastllm.git "${SRC_DIR}"
else
  echo "[INFO] 已存在 fastllm 源码目录，跳过克隆。"
fi

cd "${SRC_DIR}"
mkdir -p build
cd build

echo "[INFO] 配置 CMake (USE_ASCEND=ON) ..."
cmake .. -DUSE_ASCEND=ON

echo "[INFO] 开始编译 ..."
make -j"$(nproc)"

echo "[INFO] 安装 Python 绑定 ..."
cd ../python
pip3 install .

echo "[INFO] fastllm 安装完成，可运行 run_qwen25_fastllm.py 测试。"
