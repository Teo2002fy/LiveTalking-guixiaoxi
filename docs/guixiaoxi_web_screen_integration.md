# GuiXiaoxi Web 大屏接入说明

## 1. Web 大屏是什么

Web 大屏是数字人的展示和交互页面，当前页面为：

```text
/guixiaoxi.html
```

它负责：

- 展示数字人视频画面。
- 建立 WebRTC 音视频连接。
- 支持麦克风语音提问。
- 支持文字兜底输入。
- 接收问题后调用问答服务，并让数字人语音播报答案。
- 支持外部前端通过 `iframe/postMessage` 控制数字人提问或播报。

第一版接入建议：前端同事直接用 `iframe` 或页面跳转接入该 Web 大屏，不需要在业务前端里重写 WebRTC。

## 2. 推荐接入方式

### 方式 A：直接打开 Web 大屏

```text
https://<数字人服务域名>/guixiaoxi.html
```

适合大屏展示页、演示页、独立入口。

### 方式 B：iframe 嵌入

```html
<iframe
  id="guixiaoxiFrame"
  src="https://<数字人服务域名>/guixiaoxi.html"
  style="width: 100%; height: 100vh; border: 0;"
  allow="microphone; autoplay; fullscreen"
></iframe>
```

注意：

- 麦克风能力需要浏览器授权。
- 线上部署建议使用 HTTPS。
- 如果嵌入到其他系统，需要确认 iframe 权限策略允许 `microphone` 和 `autoplay`。

## 3. 外部控制方式

### 3.1 让数字人自己问答并播报

外部前端只传用户问题，数字人服务会调用后端问答接口，然后播报答案。

```js
const frame = document.getElementById("guixiaoxiFrame");

frame.contentWindow.postMessage({
  type: "guixiaoxi-question",
  question: "请介绍一下项目亮点"
}, "*");
```

数字人页面完成问答后，会回传：

```js
window.addEventListener("message", (event) => {
  if (event.data?.type === "guixiaoxi-answer") {
    console.log(event.data.data);
  }
});
```

### 3.2 已有答案时，只让数字人播报

如果业务系统已经完成问答，只需要数字人播报答案，使用：

```js
frame.contentWindow.postMessage({
  type: "guixiaoxi-speak",
  question: "请介绍一下项目亮点",
  answer: "本项目面向农业知识问答场景，提供智能检索、语音播报和数字人交互能力。"
}, "*");
```

数字人页面播报请求提交后，会回传：

```js
window.addEventListener("message", (event) => {
  if (event.data?.type === "guixiaoxi-spoken") {
    console.log(event.data.data);
  }
});
```

### 3.3 主动连接

```js
frame.contentWindow.postMessage({
  type: "guixiaoxi-connect"
}, "*");
```

连接成功后，数字人页面会回传当前会话：

```js
window.addEventListener("message", (event) => {
  if (event.data?.type === "guixiaoxi-session") {
    console.log(event.data.sessionid);
  }
});
```

## 4. 后端接口

### 4.1 问答并播报

```http
POST /api/digital-human/chat
Content-Type: application/json
```

请求：

```json
{
  "sessionid": "数字人页面连接后产生的 sessionid",
  "question": "请介绍一下项目亮点",
  "conversation_id": "",
  "interrupt": true,
  "inputs": {}
}
```

返回：

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "sessionid": "...",
    "question": "请介绍一下项目亮点",
    "answer": "这里是问答服务返回的答案",
    "conversation_id": "...",
    "meta": {}
  }
}
```

### 4.2 只播报已有答案

```http
POST /api/digital-human/speak
Content-Type: application/json
```

请求：

```json
{
  "sessionid": "数字人页面连接后产生的 sessionid",
  "question": "请介绍一下项目亮点",
  "answer": "这里是业务系统已经生成好的答案",
  "interrupt": true
}
```

返回：

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "sessionid": "...",
    "question": "请介绍一下项目亮点",
    "answer": "这里是业务系统已经生成好的答案"
  }
}
```

### 4.3 打断播报

```http
POST /interrupt_talk
Content-Type: application/json
```

请求：

```json
{
  "sessionid": "数字人页面连接后产生的 sessionid"
}
```

## 5. 问答服务配置

数字人后端通过 `C:\lt\conf.ini` 的 `[llm]` 配置连接 Dify 或 OpenAI 兼容接口。

### Dify 示例

```ini
[llm]
provider = dify
base_url = https://<dify-host>/v1
api_key = app-xxxx
model = dify
```

### OpenAI 兼容接口示例

```ini
[llm]
provider = openai
base_url = https://<llm-host>/v1
api_key = sk-xxxx
model = qwen3-14b
```

如果业务系统已有问答接口，也可以把 `qa_client.py` 中的请求格式改成业务接口格式。

## 6. 前端需要确认的信息

前端同事接入前需要确认：

- Web 大屏最终部署域名。
- 是否用 iframe 嵌入，还是直接打开独立页面。
- 是否需要外部系统传问题给数字人。
- 是否由数字人服务调用 Dify/问答接口，还是业务系统先返回答案再让数字人播报。
- 线上环境是否支持 HTTPS、麦克风权限、WebRTC 所需端口或反向代理配置。

