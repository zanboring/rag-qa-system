"""大模型调用：mock / Ollama / GLM-4 可切换。

面试可讲点：
- RAG 的「生成」环节调用 LLM，把检索到的上下文拼进 Prompt 再生成答案。
- 本项目默认 mock（确定性假模型，便于单测与离线演示），
  生产切 Ollama 本地模型或 GLM-4 云端模型，改环境变量即可。
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


class GLM4LLM(LLM):
    """智谱 GLM-4 云端模型（兼容 OpenAI 协议，需 ZHIPU_API_KEY）。"""

    def __init__(self, api_key: str, model: str = "glm-4-flash"):
        if not api_key:
            raise ValueError("使用 glm4 后端需设置 ZHIPU_API_KEY 环境变量")
        self.api_key = api_key
        self.model = model
        # 复用连接池
        self._client = httpx.AsyncClient(timeout=120)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(self, system: str, prompt: str) -> str:
        resp = await self._client.post(
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
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
    if config.LLM_BACKEND == "ollama":
        return OllamaLLM(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
    if config.LLM_BACKEND == "glm4":
        return GLM4LLM(config.ZHIPU_API_KEY, config.ZHIPU_CHAT_MODEL)
    return MockLLM()
