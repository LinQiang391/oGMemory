#!/usr/bin/env bash
################################################################################
# OpenClaw + oG-Memory 安装脚本
#
# 功能：下载部署脚本和配置文件到本地
#
# 使用方法：
#   远程一键安装（推荐）
#     curl -fsSL https://gitcode.com/opengauss/oG-Memory/raw/dev/deploy/install.sh | bash
#
# 执行后会下载以下文件到 ./ogmem-deploy/ 目录：
#   - deploy.env: 配置文件
#   - deploy.sh:  部署脚本
################################################################################
set -euo pipefail

REPO_RAW="https://gitcode.com/opengauss/oG-Memory/raw/dev"
TARGET_DIR="./ogmem-deploy"

echo "下载 OpenClaw + oG-Memory 一键部署脚本..."
mkdir -p "${TARGET_DIR}"

curl -fsSL "${REPO_RAW}/deploy/deploy.env" -o "${TARGET_DIR}/deploy.env"
curl -fsSL "${REPO_RAW}/deploy/deploy.sh"  -o "${TARGET_DIR}/deploy.sh"
chmod +x "${TARGET_DIR}/deploy.sh"

echo ""
echo "下载完成！文件已保存到 ${TARGET_DIR}/"
echo ""
echo "架构说明："
echo "  本部署脚本会启动三个容器（如果启用 openGauss）："
echo "    1. openGauss 数据库容器（可选）"
echo "    2. ogmemory 容器：独立的 oG-Memory 服务（HTTP API，端口 8090）"
echo "    3. openclaw-ogmemory 容器：OpenClaw Gateway（Web 界面，端口 18789）"
echo ""
echo "使用方法："
echo "  1. cd ${TARGET_DIR}"
echo "  2. 编辑 deploy.env 填写必要配置"
echo ""
echo "  必填配置项："
echo "    OpenClaw Gateway 配置："
echo "     - LLM_PROVIDER: LLM提供商标识（dashscope/openai/volcengine等）"
echo "     - LLM_API_KEY: API密钥"
echo "     - LLM_BASE_URL: API地址"
echo "     - LLM_MODEL: 模型名称"
echo ""
echo "    oG-Memory 服务配置："
echo "     - EMBEDDING_MODEL: Embedding模型名称"
echo "     - OPENAI_API_KEY: oG-Memory使用的API密钥（可选，默认使用OpenClaw的）"
echo "     - OPENAI_BASE_URL: oG-Memory使用的API地址（可选，默认使用OpenClaw的）"
echo ""
echo "    openGauss 配置："
echo "     - ENABLE_OPENGAUSS: 是否启用openGauss（true/false）"
echo "     - OG_HOST_PORT: openGauss宿主机端口"
echo ""
echo "    容器配置："
echo "     - HOST_IP: 宿主机IP地址"
echo "     - AGFS_DATA_DIR: AGFS数据存储目录（用于存储记忆文件）"
echo ""
echo "  3. 部署方式："
echo "     a) 完整部署（openGauss + ogmemory + openclaw-ogmemory）:"
echo "        bash deploy.sh -password <openGauss密码>"
echo "        密码要求：至少8字符，包含大写、小写、数字、特殊符号中的至少3种"
echo ""
echo "     b) 仅部署 ogmemory + openclaw-ogmemory:"
echo "        修改 deploy.env 中 ENABLE_OPENGAUSS=false"
echo "        bash deploy.sh"
echo ""
echo "首次访问："
echo "  oG-Memory API:  http://<HOST_IP>:8090"
echo "  OpenClaw Web:   http://<HOST_IP>:18789"
echo "  如果设置了 OPENCLAW_GATEWAY_TOKEN，需要输入 Token 完成配对"
echo ""
echo "配置文件说明："
echo "  deploy.env 包含五部分配置："
echo "    1. OpenClaw Gateway 配置：用于 OpenClaw Gateway 服务"
echo "    2. oG-Memory 服务配置：用于 oG-Memory 记忆服务"
echo "    3. openGauss 配置：用于 openGauss 数据库容器"
echo "    4. 容器配置：ogmemory 和 openclaw-ogmemory 容器的配置"
echo "    5. 部署控制参数：控制部署行为的参数"
echo ""
echo "详细配置说明请参考 deploy.env 中的注释"
