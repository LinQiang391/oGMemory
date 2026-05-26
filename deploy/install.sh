#!/usr/bin/env bash
################################################################################
# OpenClaw + oGMemory 安装脚本
#
# 功能：下载部署脚本和配置文件到本地
#
# 使用方法：
#   远程一键安装（推荐）
#     curl -fsSL https://raw.gitcode.com/opengauss/oGMemory/raw/dev/deploy/install.sh | bash
#
# 执行后会下载以下文件到 ./ogmem-deploy/ 目录：
#   - deploy.env: 配置文件（OpenClaw + 基础设施配置）
#   - ogmemory.example.yaml: oGMemory 服务配置模板
#   - deploy.sh:  部署脚本
################################################################################
set -euo pipefail

REPO_RAW="https://raw.gitcode.com/opengauss/oGMemory/raw/dev"
TARGET_DIR="./ogmem-deploy"

echo "下载 OpenClaw + oGMemory 一键部署脚本..."
mkdir -p "${TARGET_DIR}"

curl -fsSL "${REPO_RAW}/deploy/deploy.env" -o "${TARGET_DIR}/deploy.env"
curl -fsSL "${REPO_RAW}/deploy/ogmemory.example.yaml" -o "${TARGET_DIR}/ogmemory.example.yaml"
curl -fsSL "${REPO_RAW}/deploy/deploy.sh"  -o "${TARGET_DIR}/deploy.sh"
chmod +x "${TARGET_DIR}/deploy.sh"

echo ""
echo "下载完成！文件已保存到 ${TARGET_DIR}/"
echo ""
echo "架构说明："
echo "  本部署脚本会启动三个容器（如果启用 openGauss）："
echo "    1. openGauss 数据库容器（可选）"
echo "    2. ogmemory 容器：独立的 oGMemory 服务（HTTP API，端口 8090）"
echo "    3. openclaw-ogmemory 容器：OpenClaw Gateway（Web 界面，端口 18789）"
echo ""
echo "使用方法："
echo "  1. cd ${TARGET_DIR}"
echo "  2. 编辑 deploy.env 填写 OpenClaw 和基础设施配置"
echo "  3. 复制并编辑 oGMemory 配置："
echo "     cp ogmemory.example.yaml ogmemory.yaml"
echo "     编辑 ogmemory.yaml 填写 LLM API Key、Embedding 模型等"
echo ""
echo "  必填配置项："
echo "    deploy.env 中："
echo "     - LLM_PROVIDER: LLM提供商标识（dashscope/openai/volcengine等）"
echo "     - LLM_API_KEY: API密钥"
echo "     - LLM_BASE_URL: API地址"
echo "     - LLM_MODEL: 模型名称"
echo ""
echo "    ogmemory.yaml 中："
echo "     - llm.api_key: oGMemory 使用的 API 密钥"
echo "     - llm.model: oGMemory 使用的模型"
echo "     - embedding.model: Embedding 模型名称"
echo "     - vector_db.dimension: 向量维度（需与 Embedding 模型匹配）"
echo ""
echo "  4. 部署方式："
echo "     a) 完整部署（openGauss + ogmemory + openclaw-ogmemory）:"
echo "        bash deploy.sh -password <openGauss密码>"
echo "        密码要求：至少8字符，包含大写、小写、数字、特殊符号中的至少3种"
echo ""
echo "     b) 仅部署 ogmemory + openclaw-ogmemory:"
echo "        修改 deploy.env 中 ENABLE_OPENGAUSS=false"
echo "        bash deploy.sh"
echo ""
echo "首次访问："
echo "  oGMemory API:  http://<OPENCLAW_HOST_IP>:8090"
echo "  OpenClaw Web:   http://<OPENCLAW_HOST_IP>:18789"
echo "  如果设置了 OPENCLAW_GATEWAY_TOKEN，需要输入 Token 完成配对"
echo ""
echo "配置文件说明："
echo "  deploy.env: OpenClaw Gateway + 基础设施（openGauss、容器、部署参数）配置"
echo "  ogmemory.yaml: oGMemory 服务配置（LLM、Embedding、向量数据库、身份认证等）"
echo ""
echo "详细配置说明请参考各文件中的注释"
