# LiveTalking 推理服务

独立的口型推理服务（wav2lip / musetalk），部署在 GPU 服务器上，自包含模型代码，不依赖主项目。主项目通过 HTTP API 调用（`base_url + model + api_key`，类似调 LLM）。

## 部署

```bash
# 1. 环境
conda create -n nerfstream python=3.10 && conda activate nerfstream
conda install pytorch==2.4.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt

# 2. 下载模型
bash scripts/download_models.sh        # 或 --musetalk / --wav2lip
bash scripts/check_models.sh

# 3. 放置 avatar 素材到 data/avatars/<avatar_id>/

# 4. 编辑 conf.ini（端口、模型、avatar、api_key）

# 5. 启动
bash scripts/start_bg.sh               # 后台；前台用 start.sh；停止用 stop.sh
```

模型与 avatar 的存储路径可在 `conf.ini` 的 `[path]` 段自定义。

## 配置 conf.ini

```ini
[server]
port = 8020
api_key = sk-your-secret-key     # 留空则不鉴权

[model]
model = musetalk                 # musetalk / wav2lip
avatar_id = your_avatar
batch_size = 8

[path]
models_dir = ./models
avatars_dir = ./data/avatars
```

命令行可覆盖：`python server.py --model wav2lip --port 9000 --config other.ini`

## API

| 接口 | 方法 | 鉴权 | 说明 |
|------|------|------|------|
| `/health` | GET | ❌ | 健康检查 |
| `/avatar_info` | GET | ✅ | avatar 信息 |
| `/switch_avatar` | POST | ✅ | 切换 avatar |
| `/audio_feature` | POST | ✅ | 音频 → 特征 |
| `/inference` | POST | ✅ | 特征 → 脸帧 |

鉴权：设置 `api_key` 后，除 `/health` 外需 `Authorization: Bearer <key>`。

```bash
curl http://localhost:8020/health
curl http://localhost:8020/avatar_info -H "Authorization: Bearer <key>"
```

## 主项目客户端配置

```ini
[inference]
remote = true
base_url = http://<GPU_IP>:8020
model = musetalk
api_key = sk-your-secret-key
```
