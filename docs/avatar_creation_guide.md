# LiveTalking 数字人制作完整指南（MuseTalk + 姿态动作编排）

本指南覆盖两件事：

1. **制作会说话的主数字人形象**（口型同步驱动）
2. **加上姿态/手势动作**，并让**不同信息或语言触发不同动作**

> 阅读前先理解一个核心概念，后面所有内容都建立在它之上。

---

## 0. 先理解：数字人的"动作"从哪来？

LiveTalking 里数字人有**两种来源完全不同的动作**，搞混会走弯路。

### 动作来源 A：说话时的自然身体动作 → 来自「源视频本身」

口型同步的原理是：**只把生成的嘴部区域贴回原始视频帧**，其余画面（头、肩、手、身体）**原封不动地播放源视频**。

代码佐证（`avatars/musetalk_avatar.py` 的 `paste_back_frame`）：把推理出的脸 `res_frame` 用 mask 贴回 `frame_list_cycle[idx]`（即源视频原帧）。

**这意味着：**
- 如果你的源视频里人物在自然地点头、微笑、手势、身体轻微晃动，那么数字人说话时**就会有这些动作**——因为它在重放源视频帧，只换了嘴。
- 想要"说话时有姿态"，**最有效的办法是录一段本身就有自然肢体动作的源视频**，而不是去配置动作编排。

### 动作来源 B：触发式特定动作 → 来自「动作编排（custom action）」

这是一套独立机制：你预先录好若干**完整的动作视频片段**（如挥手问好、思考、再见、待机），运行时通过 API **手动触发**某个片段播放。

特点（代码佐证 `avatars/base_avatar.py`）：
- 触发的片段是**整帧直接播放**的（`custom_img_cycle[audiotype]`），**不做口型同步**。
- 片段可以带自己的音频（预录的 wav），也可以纯画面（待机循环）。
- 播放完自动回到待机/正常状态。

**适用场景：** 固定的、可预录的动作——开场问好、"让我想想"、结束语、不说话时的待机循环。

> 一句话总结：**动态回答的内容用 A（源视频自带动作 + 口型同步）；固定的仪式性动作用 B（动作编排触发）。** 二者配合，就是一个有姿态、会互动的数字人。

---

## 1. 前置准备

### 1.1 下载模型

```bash
bash download_models.sh   # 下载 MuseTalk 全部模型到 models/
bash check_models.sh      # 校验完整性，全绿才继续
```

### 1.2 安装 mmpose（生成形象必需）

MuseTalk 的形象生成脚本依赖 `mmpose`（`avatars/musetalk/utils/preprocessing.py` 里 `from mmpose.apis import ...`），但 `requirements.txt` 没有，需手动装：

```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"
```

---

## 2. Part A —— 制作主数字人形象

### 2.1 录制/准备源视频（决定最终质量的关键）

| 要求 | 说明 |
|------|------|
| **帧率 25fps** | 项目硬性要求（`--fps` 必须 25），否则音画不同步 |
| **正脸面向镜头** | 避免大幅转头、低头、侧脸 |
| **光线均匀、背景固定** | 背景动/光线闪会穿帮 |
| **时长 30s ~ 5min** | 太短循环会看出重复 |
| **嘴部动作自然丰富** | 最好就是真人在自然说话 |
| **★ 自带自然肢体动作** | 想要"说话有姿态"，源视频里就要有自然的点头、微笑、轻微手势、身体晃动（见第 0 节动作来源 A） |
| **人物不要移出画面** | 身体可轻微动，但别大幅位移 |

> 提示：录一段"自然讲话状态"的视频（像在跟人聊天那样有表情和小幅手势），生成出来的数字人说话时就会有这些姿态，比任何配置都自然。

把视频放到例如 `data/video/myface.mp4`。

### 2.2 生成形象（命令行）

```bash
python -m avatars.musetalk.genavatar \
  --file data/video/myface.mp4 \
  --avatar_id my_musetalk1 \
  --version v15 \
  --extra_margin 10 \
  --parsing_mode jaw \
  --bbox_shift 0
```

> 注意：MuseTalk 的生成脚本参数名是 `--file`（不是 wav2lip 的 `--video_path`）。

### 2.3 生成形象（网页 / API）

- **网页**：启动服务后打开 `http://<服务器IP>:8010/avatar.html`，model 选 `musetalk`，上传视频，填 `avatar_id`，提交看进度。
- **API**：
  ```bash
  curl -X POST http://<服务器IP>:8010/api/avatar/task \
    -F "model=musetalk" \
    -F "avatar_id=my_musetalk1" \
    -F "video_file=@data/video/myface.mp4" \
    -F "version=v15" -F "extra_margin=10" -F "parsing_mode=jaw"
  # 返回 task_id，轮询 GET /api/avatar/task/{task_id} 看 progress
  ```

### 2.4 生成产物

`data/avatars/my_musetalk1/` 下会有：

```
my_musetalk1/
├── full_imgs/          # 源视频每帧完整画面（说话时身体动作就来自这里）
├── mask/               # 每帧人脸贴回用的 mask
├── coords.pkl          # 人脸坐标
├── mask_coords.pkl     # mask 坐标
├── latents.pt          # VAE 隐变量（口型推理输入）
└── avator_info.json    # 元信息
```

### 2.5 启动

```bash
python app.py --transport webrtc \
  --model musetalk --avatar_id my_musetalk1 \
  --batch_size 16 --fps 25
```

浏览器开 `http://<服务器IP>:8010/index.html` 测试说话。

### 2.6 调优 bbox_shift

生成时日志会打印一行建议范围：

```
Manually adjust range : [ -9~9 ] , the current value: 0
```

如果默认 `bbox_shift=0` 出来嘴型偏上/偏下，就在这个范围内取值（正值嘴部区域下移，负值上移），重新生成。`extra_margin` 增大可改善下巴贴回不自然。

---

## 3. Part B —— 制作姿态/手势动作（动作编排）

### 3.1 audiotype 的语义（核心规则）

动作编排用一个整数 `audiotype` 标识每个动作。代码里有明确约定（`base_avatar.py` + `base_asr.py`）：

| audiotype | 含义 | 是否带音频 | 触发方式 |
|-----------|------|-----------|----------|
| `0` | 正常模式（TTS 口型同步说话） | — | 默认 |
| `1` | **待机循环**：不说话时自动播放的动作 | 通常不带 | 静音时**自动**播放 |
| `≥2` | **触发式动作**：完整片段（画面+音频）播一遍后自动回到待机 | 带音频 | 调 API `/set_audiotype` |

机制要点：
- `audiotype=1` 是特殊待机态，只要定义了，**数字人不说话时就自动循环播放它**（不用手动触发）。
- `audiotype≥2` 必须带音频（`audiopath`），调 API 触发后播放对应"画面帧+音频"，播完自动切回 `1`。
- 帧序列用 `mirror_index` 做**乒乓循环**（正放→倒放→正放），所以即使片段首尾不衔接也能平滑循环。

### 3.2 录制动作片段

为每个动作录一段独立视频，例如：

| 动作 | 用途 | 建议 audiotype |
|------|------|----------------|
| 待机微动 | 不说话时循环 | 1 |
| 挥手问好 | 用户进入/打招呼 | 2 |
| 托腮思考 | 检索/等待 LLM 时 | 3 |
| 挥手再见 | 结束对话 | 4 |

录制要求与主形象一致：**25fps、同一人物、同样的机位/景别/分辨率**。

> ⚠️ 关键：动作片段的**分辨率必须和主形象源视频一致**，因为它们会被直接推送到输出（`output.push_video_frame`），尺寸不一致会出错或错位。

### 3.3 把视频转成帧序列 + 音频

动作片段不需要跑 genavatar，只要拆成**帧图片**（+可选音频）。用 ffmpeg：

```bash
# 拆帧（命名必须是纯数字、可带前导零）
mkdir -p data/customvideo/wave
ffmpeg -i wave.mp4 -r 25 data/customvideo/wave/%08d.png

# 提取音频：必须 16kHz 单声道 wav（代码不做重采样，直接按 16k 读）
ffmpeg -i wave.mp4 -ar 16000 -ac 1 data/customvideo/wave.wav
```

命名规则（代码 `__loadcustom` 按 `int(文件名)` 排序）：`00000000.png, 00000001.png, ...` 都可以，关键是去掉扩展名后是整数。

**音频格式硬要求**：16kHz、单声道、PCM wav。代码 `get_custom_audio_stream` 直接按 `self.chunk`(320 samples=20ms@16k) 切片，不重采样，采样率不对会变速/噪音。

### 3.4 目录结构示例

```
data/customvideo/
├── image/            # audiotype=1 待机帧（纯画面，无音频）
│   ├── 00000000.png
│   └── ...
├── wave/             # audiotype=2 挥手帧
│   └── ...
├── wave.wav          # audiotype=2 配套音频(16k mono)
├── think/            # audiotype=3 思考帧
│   └── ...
├── think.wav
├── leave/            # audiotype=4 再见帧
│   └── ...
└── leave.wav
```

### 3.5 编写 custom_config.json

参考项目自带的 `data/custom_config.json`：

```json
[
    {
        "audiotype": 1,
        "imgpath": "data/customvideo/image"
    },
    {
        "audiotype": 2,
        "imgpath": "data/customvideo/wave",
        "audiopath": "data/customvideo/wave.wav"
    },
    {
        "audiotype": 3,
        "imgpath": "data/customvideo/think",
        "audiopath": "data/customvideo/think.wav"
    },
    {
        "audiotype": 4,
        "imgpath": "data/customvideo/leave",
        "audiopath": "data/customvideo/leave.wav"
    }
]
```

- `audiotype=1` 只给 `imgpath`（待机循环，无音频）。
- `audiotype≥2` 同时给 `imgpath` 和 `audiopath`。

### 3.6 启动时加载动作编排

**全局方式**（所有会话生效）：

```bash
python app.py --transport webrtc \
  --model musetalk --avatar_id my_musetalk1 \
  --batch_size 16 --fps 25 \
  --customvideo_config data/custom_config.json
```

**按会话方式**（每个连接不同动作）：在 WebRTC `/offer` 请求里传 `custom_config` 字段（JSON 字符串），见 `app.py` 的 `build_avatar_session`。

---

## 4. Part C —— 让不同信息/语言触发不同动作

### 4.1 触发 API

```
POST /set_audiotype
Content-Type: application/json

{ "sessionid": "<会话ID>", "audiotype": 2 }
```

调用后，对应 `audiotype` 的动作片段立即播放，播完自动回到待机(1)。这是**唯一的触发入口**（`server/routes.py` → `set_custom_state`）。

> 重要：项目**不会自动**根据文字内容或语言去触发动作。"什么内容触发什么动作"的判断逻辑需要你自己实现，然后调这个 API。下面给三种实现策略。

### 4.2 策略一：客户端按关键词/语言触发（最简单，零改后端）

在网页/客户端发送消息前，先判断内容，再决定是否触发动作。示例（JS）：

```javascript
async function sendMessage(text, sessionid) {
  // 1. 规则判断 → 选 audiotype
  let audiotype = null;
  if (/^(你好|hi|hello|こんにちは)/i.test(text)) audiotype = 2;      // 问好→挥手
  else if (/(再见|bye|さようなら)/i.test(text))   audiotype = 4;      // 告别→再见
  else if (text.length > 30)                       audiotype = 3;      // 长问题→思考

  // 2. 先触发动作（可选）
  if (audiotype) {
    await fetch('/set_audiotype', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ sessionid, audiotype })
    });
  }

  // 3. 再发文本驱动数字人回答
  await fetch('/human', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ text, type: 'chat', sessionid })
  });
}
```

### 4.3 策略二：按语言自动切换动作 / 音色

检测语言后触发不同动作（也可顺便切 TTS 音色）。示例用一个简单的语言判断：

```javascript
function detectLang(text) {
  if (/[\u3040-\u30ff]/.test(text)) return 'ja';   // 日文假名
  if (/[\u4e00-\u9fff]/.test(text)) return 'zh';   // 中文
  if (/[a-zA-Z]/.test(text))        return 'en';
  return 'unknown';
}

const LANG_ACTION = { zh: 2, en: 5, ja: 6 };  // 不同语言对应不同问候动作
const lang = detectLang(text);
if (LANG_ACTION[lang]) {
  await fetch('/set_audiotype', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ sessionid, audiotype: LANG_ACTION[lang] })
  });
}
```

### 4.4 策略三：在 llm.py 里按回答内容触发（服务端，最灵活）

如果想根据 **LLM 生成的回答**来触发动作（比如回答里含致歉就播放鞠躬），在 `llm.py` 的 `llm_response` 里调用触发逻辑。`avatar_session` 已经在手上，可直接调 `set_custom_state`：

```python
# llm.py 中（示意）
def maybe_trigger_action(text, avatar_session):
    if any(k in text for k in ["抱歉", "对不起"]):
        avatar_session.set_custom_state(7)   # 鞠躬致歉
    elif any(k in text for k in ["再见", "拜拜"]):
        avatar_session.set_custom_state(4)   # 再见

# 在流式拼句、put_msg_txt 之前/之后按需调用
maybe_trigger_action(result, avatar_session)
```

> 注意：`set_custom_state(n)` 的 `n` 必须是 config 里已注册且带 `audiopath` 的 `audiotype`（≥2），否则会被 `set_custom_state` 里的 `if self.custom_audio_index.get(audiotype) is None: return` 直接忽略。

### 4.5 触发与打断的关系

- 动作片段播放期间，用户若发新消息并带 `interrupt:true`，`/human` 会调 `flush_talk()` 清空队列并把 `custom_audiotype` 重置为 0（正常态）。互动场景务必接上打断。
- 触发式动作（≥2）播完会自动回到待机(1)，无需手动复位。

---

## 5. 常见坑速查

| 现象 | 原因 | 解决 |
|------|------|------|
| 音画不同步 | 源视频/动作片段不是 25fps | 全部用 `-r 25` 处理 |
| 动作音频变速/有噪音 | 配套 wav 不是 16k 单声道 | `ffmpeg -ar 16000 -ac 1` |
| 触发动作无反应 | audiotype 没在 config 注册，或 ≥2 却没给 audiopath | 检查 `custom_config.json` |
| 动作画面错位/报错 | 动作片段分辨率与主形象不一致 | 统一分辨率/机位重录 |
| 待机动作不播放 | 没定义 audiotype=1，或没加 `--customvideo_config` | 补 config 并在启动加参数 |
| 帧顺序乱 | 帧文件名不是纯数字 | 用 `%08d.png` 命名 |
| 说话时身体僵硬不动 | 源视频本身是定格/不动的 | 录一段自带自然肢体动作的源视频（见第 0 节） |
| 生成形象报缺 mmpose | 未装 mmpose | 见 1.2 |

---

## 6. 端到端完整示例

```bash
# ① 模型
bash download_models.sh && bash check_models.sh

# ② mmpose
pip install -U openmim
mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"

# ③ 生成主形象（源视频自带自然肢体动作）
python -m avatars.musetalk.genavatar \
  --file data/video/myface.mp4 --avatar_id my_musetalk1 \
  --version v15 --extra_margin 10 --parsing_mode jaw --bbox_shift 0

# ④ 准备动作片段（示例：待机 + 挥手）
ffmpeg -i idle.mp4 -r 25 data/customvideo/image/%08d.png
ffmpeg -i wave.mp4 -r 25 data/customvideo/wave/%08d.png
ffmpeg -i wave.mp4 -ar 16000 -ac 1 data/customvideo/wave.wav
# 编辑 data/custom_config.json 注册 audiotype 1/2...

# ⑤ 启动（带动作编排 + 流式 TTS）
python app.py --transport webrtc \
  --model musetalk --avatar_id my_musetalk1 \
  --batch_size 16 --fps 25 \
  --customvideo_config data/custom_config.json \
  --tts gpt-sovits --TTS_SERVER http://127.0.0.1:9880 \
  --REF_FILE data/ref.wav --REF_TEXT "参考音频文本"
```

之后在客户端按 4.2~4.4 的策略，根据信息/语言调用 `/set_audiotype` 触发对应动作即可。

---

## 7. 设计建议小结

- **会动的说话** → 靠"自带自然动作的源视频" + 口型同步（动作来源 A）。这是性价比最高的"姿态"来源。
- **仪式化动作**（问好/思考/再见/待机）→ 用动作编排（动作来源 B）预录片段 + `/set_audiotype` 触发。
- **不同信息/语言触发不同动作** → 在客户端或 `llm.py` 里做内容/语言判断，映射到 audiotype，调触发 API。
- 动作片段是**预录整段播放、不做口型同步**的，所以它适合固定内容；动态回答仍由主形象口型驱动完成。
