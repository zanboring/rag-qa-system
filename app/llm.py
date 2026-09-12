"""大模型调用：mock / Ollama / DeepSeek / GLM-4 可切换。

面试可讲点：
- RAG 的「生成」环节调用 LLM，把检索到的上下文拼进 Prompt 再生成答案。
- 本项目默认 mock（确定性假模型，便于单测与离线演示），生产改环境变量即可切换：
    LLM_BACKEND=mock      零依赖假模型，用于单测与链路验证
    LLM_BACKEND=ollama    本地模型，数据不出本地（需 ollama serve）
    LLM_BACKEND=deepseek  DeepSeek 云端（OpenAI 兼容，需 DEEPSEEK_API_KEY）
    LLM_BACKEND=glm4      智谱 GLM-4 云端（OpenAI 兼容，需 ZHIPU_API_KEY）

关于可插拔的设计取舍：云端厂商虽然各异，但绝大多数都实现了 OpenAI 兼容协议，
因此用一个 OpenAICompatLLM 覆盖，而不是每个厂商复制一份请求逻辑；
协议确实不同的（如 Ollama 原生接口）才单独实现。
"""

import httpx

from app import config


class LLM:
    async def generate(self, system: str, prompt: str) -> str:
        raise NotImplementedError

    async def aclose(self) -> None:
        """释放后端持有的资源（如 HTTP 连接池）。

        默认空实现：mock 不持有外部连接。持有 httpx.AsyncClient 的后端必须覆写，
        由服务关闭阶段统一调用，否则连接池会驻留到进程退出。
        """
        return None


class MockLLM(LLM):
    """确定性假模型：回显参考资料的来源编号，方便验证 RAG 链路。"""

    async def generate(self, system: str, prompt: str) -> str:
        # 提取 prompt 里的来源编号，模拟「依据资料回答」
        return f"[mock] 已依据参考资料生成回答。上下文结尾：{prompt[-120:]}"


class OllamaLLM(LLM):
    """本地 Ollama 模型（需先 `ollama serve` 并 `ollama pull <model>`）。"""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        # 复用连接池
        self._client = httpx.AsyncClient(timeout=120)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(self, system: str, prompt: str) -> str:
        resp = await self._client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class OpenAICompatLLM(LLM):
    """任何实现 OpenAI 兼容协议的云端模型（DeepSeek / 智谱 GLM-4 / 其它网关）。

    为什么不给每个厂商写一个类：DeepSeek、智谱 GLM-4 以及大多数国产模型服务
    都实现了 OpenAI 的接口协议——请求体是 `messages` 数组，响应体取
    `choices[0].message.content`。差异只收敛在三个参数上：base_url / api_key / model。
    抽出一个实现，新增后端就只是加一行配置映射，而不是复制一整段请求代码。

    与之相对，Ollama 的原生 `/api/chat` 协议不同（响应取 `message.content`），
    因此保留独立的实现，不用一个类去兼容两种协议。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider: str,
        timeout: float = 120.0,
    ):
        if not api_key:
            raise ValueError(
                f"使用 {provider} 后端需先设置对应的 API Key 环境变量"
                f"（如 {provider.upper()}_API_KEY）"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider
        # 复用连接池：避免每次请求都重新建连与 TLS 握手
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(self, system: str, prompt: str) -> str:
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def get_llm() -> LLM:
    """按 LLM_BACKEND 返回生成后端。

    deepseek / glm4 走同一套 OpenAI 兼容实现，ollama 用自己的原生协议，
    mock 用于零依赖离线测试。
    """
    if config.LLM_BACKEND == "deepseek":
        return OpenAICompatLLM(
            config.DEEPSEEK_BASE_URL,
            config.DEEPSEEK_API_KEY,
            config.DEEPSEEK_CHAT_MODEL,
            provider="deepseek",
        )
    if config.LLM_BACKEND == "glm4":
        return OpenAICompatLLM(
            config.ZHIPU_BASE_URL,
            config.ZHIPU_API_KEY,
            config.ZHIPU_CHAT_MODEL,
            provider="glm4",
        )
    if config.LLM_BACKEND == "ollama":
        return OllamaLLM(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
    return MockLLM()
