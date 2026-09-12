"""LLM-as-judge：用大模型评估 RAG 答案质量。

支持的后端
----------
任何 **OpenAI 兼容** 的 chat/completions 接口，通过环境变量配置：

    JUDGE_BASE_URL  默认 https://open.bigmodel.cn/api/paas/v4   （智谱 GLM-4）
    JUDGE_API_KEY   默认复用 ZHIPU_API_KEY
    JUDGE_MODEL     默认 glm-4-flash
    JUDGE_CONCURRENCY 默认 4（并发上限，避免触发服务端限流）

因此同一份代码可以指向：
    - 智谱 GLM-4（默认）
    - Ollama 本地模型：JUDGE_BASE_URL=http://localhost:11434/v1
    - 任何其他 OpenAI 兼容网关

设计原则
--------
1. **judge 失败不能中断评测**：单条样本的裁判调用失败时返回 None，由调用方跳过该条，
   而不是抛异常让整个评测前功尽弃（评测动辄几十次调用，个别失败是常态）。

2. **无 Key 时诚实降级**：没有配置 API Key 时，使用规则模式（RuleBasedJudge）。
   规则模式**不产出** faithfulness / answer_relevancy —— 这两个指标语义上必须由模型
   判断，用关键词覆盖率冒充会得出看似合理、实则误导的数字。规则模式只产出它真正
   能可靠计算的指标（拒答判定、关键词覆盖），报告中会明确标注"生成指标不可用"。

3. **要求给出理由**：只输出分数的裁判无法审计，无法判断评分是否可靠。
   每条评分都要求一句话理由，便于人工抽查裁判质量。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass

import httpx

# ---------------------------------------------------------------------------
# 裁判提示词
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "你是一个严格的 RAG 答案质量评估员。你需要根据给出的参考资料，评估回答的质量。"
    "你的评估必须客观、可复现，不要因为回答的措辞风格、长度或语气而调整分数。"
)

JUDGE_USER_TEMPLATE = """请根据【参考资料】评估【回答】对【问题】的作答质量。

【问题】
{question}

【参考资料】
{contexts}

【回答】
{answer}

请从以下维度评估：

1. faithfulness（忠实度，0~1）：回答中的论断是否能由参考资料支撑。
   - 若回答包含参考资料中不存在的外部事实（编造），应大幅扣分。
   - 若回答只是忠实复述资料，即使不完全等同于标准答案，也应给高分。

2. answer_relevancy（答案相关性，0~1）：回答是否切题。
   - 若答非所问、偏题、只复述资料而未回答问题，应扣分。

3. answered（布尔值）：回答是否给出了实质内容。
   - 若回答明确指出"资料中未找到相关信息""无法回答"等，则 answered = false。

4. reason（字符串）：用一句话说明评分依据。

特别注意：
- 当参考资料确实不包含答案时，**明确指出资料不足是正确行为**，
  此时 faithfulness 与 answer_relevancy 都应给高分（0.8 以上），answered 为 false。
- 当参考资料包含答案而回答却说"未找到"时，属于错误拒答，faithfulness 应大幅扣分。

只输出一个 JSON 对象，不要输出任何其他文字、解释或 markdown 代码块标记：
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "answered": true, "reason": ""}}"""


@dataclass
class JudgeScore:
    """单条样本的裁判结果。字段为 None 表示该指标未产出（而非得 0 分）。"""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    answered: bool | None = None
    reason: str = ""
    error: str | None = None

    @property
    def usable(self) -> bool:
        """是否成功产出可用评分。"""
        return self.faithfulness is not None


# ---------------------------------------------------------------------------
# 规则模式（零依赖降级）
# ---------------------------------------------------------------------------

# 拒答的标志性表述。命中任一即认为模型选择了拒答。
ABSTENTION_MARKERS = (
    "未找到",
    "没有找到",
    "无法回答",
    "不能回答",
    "资料不足",
    "信息不足",
    "无法确定",
    "未提及",
    "没有提及",
    "不包含",
    "无法从",
    "缺少相关",
)

# mock 后端的输出前缀。识别它可用于提示"当前是假模型，生成指标无意义"。
MOCK_MARKER = "[mock]"


class RuleBasedJudge:
    """零依赖的规则裁判：只产出真正可靠的指标。

    它**不**声称能评估忠实度与答案相关性——这两者需要语义理解。它只做两件确定的事：
    1. 拒答判定：答案中是否出现拒答标志词；
    2. 标记 mock 输出：提示生成指标不可用。

    它的定位是"底线哨兵"：在无 API Key 环境下仍然能测出系统的拒答行为，
    并明确告诉使用者"其余生成指标没有数据"，避免报告出现虚构的指标。
    """

    name = "rule-based"

    async def score(self, question: str, answer: str, contexts: list[str]) -> JudgeScore:
        text = (answer or "").strip()
        abstained = any(marker in text for marker in ABSTENTION_MARKERS)
        is_mock = text.startswith(MOCK_MARKER)

        if is_mock:
            # mock 后端不产生真实答案，它的输出既不构成"作答"也不构成"拒答"。
            # 这里 answered 必须返回 None（不适用）而不是 False——若返回 False，
            # 会被上层解读为"拒答"，导致 mock 模式下所有样本都被判为拒答，
            # 拒答率与误拒率同时显示为 1.0，得出自相矛盾的报告。
            return JudgeScore(
                faithfulness=None,
                answer_relevancy=None,
                answered=None,
                reason="当前为 mock 生成后端，答案不构成真实作答，生成与拒答指标均不可用",
                error="mock_backend",
            )

        reason = "答案中检测到拒答表述" if abstained else "答案给出了实质内容（规则模式不评估语义正确性）"
        return JudgeScore(
            faithfulness=None,
            answer_relevancy=None,
            answered=not abstained,
            reason=reason,
            error=None,
        )


# ---------------------------------------------------------------------------
# LLM 裁判
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """从模型输出中稳健地提取 JSON 对象。

    模型常见的"不听话"输出形态：
      - 包裹在 ```json ... ``` 代码块中
      - 在 JSON 前后附带解释性文字
    直接 json.loads 会失败，因此先尝试逐字解析，再退化为提取第一个花括号区间。
    """
    text = text.strip()
    # 去除 markdown 代码块围栏
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 退化路径：截取第一个 { 到最后一个 } 之间的内容
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法从裁判输出中解析 JSON：{text[:200]!r}") from exc
    raise ValueError(f"裁判输出中未找到 JSON：{text[:200]!r}")


def _clamp(value, low: float = 0.0, high: float = 1.0) -> float | None:
    """把裁判给的分数夹到合法区间；非数值则返回 None（视为无效，而不是当 0 分）。"""
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return None


class LLMJudge:
    """调用 OpenAI 兼容接口做裁判。"""

    name = "llm"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        concurrency: int = 4,
        timeout: float = 90.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # 复用连接池；并发用信号量限制，避免触发服务端 QPS 限制
        self._client = httpx.AsyncClient(timeout=timeout)
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        # 统计调用情况，便于在报告中披露裁判的开销与失败率
        self.calls = 0
        self.failures = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _chat(self, system: str, user: str) -> str:
        """调用一次对话接口，带有限次数的指数退避重试。

        只对 429（限流）与 5xx（服务端错误）重试——这些是暂时性故障。
        对 4xx（除 429 外）不重试：参数错误、余额不足、鉴权失败重试多少次都一样。
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,  # 裁判必须确定性，温度拉满会引入随机评分
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._client.post(url, json=payload, headers=headers)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                last_error = exc
                if status != 429 and status < 500:
                    break  # 不可重试的错误，立即放弃
                await asyncio.sleep(2**attempt)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                await asyncio.sleep(2**attempt)

        raise RuntimeError(f"裁判调用失败：{last_error}")

    async def score(self, question: str, answer: str, contexts: list[str]) -> JudgeScore:
        context_text = "\n\n".join(contexts) if contexts else "（无检索结果）"
        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            contexts=context_text,
            answer=answer,
        )

        async with self._semaphore:
            self.calls += 1
            try:
                raw = await self._chat(JUDGE_SYSTEM_PROMPT, user_prompt)
                data = _extract_json(raw)
            except Exception as exc:  # noqa: BLE001 — 裁判失败不应中断评测
                self.failures += 1
                return JudgeScore(reason="", error=str(exc))

        return JudgeScore(
            faithfulness=_clamp(data.get("faithfulness")),
            answer_relevancy=_clamp(data.get("answer_relevancy")),
            answered=bool(data.get("answered")) if "answered" in data else None,
            reason=str(data.get("reason", ""))[:200],
        )


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def get_judge():
    """按环境变量返回裁判实例；未配置 API Key 时降级为规则裁判。

    API Key 的解析顺序：
        JUDGE_API_KEY → ZHIPU_API_KEY → （无）
    这样只配置了项目本身的 ZHIPU_API_KEY 时也能直接用于评测，无需重复配置。
    """
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("ZHIPU_API_KEY") or ""
    if not api_key:
        return RuleBasedJudge()

    return LLMJudge(
        base_url=os.getenv("JUDGE_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        api_key=api_key,
        model=os.getenv("JUDGE_MODEL", "glm-4-flash"),
        concurrency=int(os.getenv("JUDGE_CONCURRENCY", "4")),
    )
