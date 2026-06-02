#!/bin/bash
# 停止推理服务
pkill -f "server.py" 2>/dev/null && echo "推理服务已停止" || echo "未发现运行中的推理服务"
