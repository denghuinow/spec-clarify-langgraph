# -*- coding: utf-8 -*-
"""
Multi-Agent SRS Generation with LangGraph
-----------------------------------------
流程：ReqParse -> 串行处理所有原子需求（ReqExplore <-> ReqClarify）-> 汇总 -> DocGenerate -> END

- ReqParse：自然语言 -> 原子需求字符串数组
- ReqExplore：单条原子需求 -> 细化需求列表（支持迭代优化）
- ReqClarify：对当前原子需求的细化需求进行评分
- 串行处理：依次处理每个原子需求，每个原子需求独立进行 ReqExplore <-> ReqClarify 迭代循环
- DocGenerate：输出 Markdown（IEEE Std 830-1998 基本格式）
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field

# 加载 .env 文件
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# 基础工具
# -----------------------------
# 日志级别
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"

def get_log_level() -> str:
    """获取日志级别，支持通过环境变量 LOG_LEVEL 控制（默认 INFO）"""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    if level in (LOG_LEVEL_DEBUG, LOG_LEVEL_INFO):
        return level
    return LOG_LEVEL_INFO


def log(message: str, level: str = LOG_LEVEL_INFO) -> None:
    """Print log messages immediately for real-time feedback.
    
    Args:
        message: 日志消息
        level: 日志级别（DEBUG 或 INFO），默认 INFO
    """
    current_level = get_log_level()
    # DEBUG 级别打印所有日志，INFO 级别只打印 INFO 及以上
    if level == LOG_LEVEL_DEBUG and current_level != LOG_LEVEL_DEBUG:
        return
    print(f"[日志] {message}", flush=True)


def log_debug(message: str) -> None:
    """打印 DEBUG 级别的日志"""
    log(message, level=LOG_LEVEL_DEBUG)


def log_info(message: str) -> None:
    """打印 INFO 级别的日志"""
    log(message, level=LOG_LEVEL_INFO)


def read_text_file(file_path: str, label: str) -> str:
    """Read UTF-8 text from the given file path, raising a helpful error if missing."""
    resolved_path = os.path.expanduser(file_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"{label} 文件不存在: {file_path}")
    log(f"读取 {label} 文件: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as f:
        return f.read()


class StreamingPrinter(BaseCallbackHandler):
    """简单的流式打印回调，便于观测模型输出过程。"""

    def __init__(self) -> None:
        self._buffer: List[str] = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:  # type: ignore[override]
        print(token, end="", flush=True)
        self._buffer.append(token)

    def get_text(self) -> str:
        return "".join(self._buffer)


def _parse_env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v not in (None, "") else default
    except Exception:
        return default


def get_agent_temperature(agent: str) -> float:
    """
    默认温度：
      ReqParse: 0.2   —— 解析与严格 JSON，更稳
      ReqExplore: 0.2 —— 受控挖掘，收敛行为
      ReqClarify: 0.2 —— 判定与对齐，需一致性
      DocGenerate: 0.1 —— 文档成形，稳定输出
    可通过环境变量覆盖：
      OPENAI_TEMP_REQPARSE / OPENAI_TEMP_REQEXPLORE / OPENAI_TEMP_REQCLARIFY / OPENAI_TEMP_DOCGENERATE
    """
    defaults = {
        "ReqParse": 0.2,
        "ReqExplore": 0.2,
        "ReqClarify": 0.2,
        "DocGenerate": 0.1,
    }
    env_key = f"OPENAI_TEMP_{agent.upper()}"
    return _parse_env_float(env_key, defaults.get(agent, 0.7))


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.7,
    streaming: bool = False,
    callbacks: Optional[List[Any]] = None,
):
    """
    包装 ChatOpenAI，兼容私有 base_url；附加合理的重试与超时。
    说明：具体可用参数随 langchain_openai 版本可能变化，未识别参数会被忽略。
    """
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if streaming:
        kwargs["streaming"] = True
    if callbacks:
        kwargs["callbacks"] = callbacks
    # 软性支持：若 SDK 识别，以下可生效；不识别则被忽略，不影响运行
    kwargs.setdefault("timeout", 120)
    kwargs.setdefault("max_retries", 3)
    return ChatOpenAI(**kwargs)


def get_llm_for(
    agent: str,
    model: Optional[str] = None,
    streaming: bool = False,
    callbacks: Optional[List[Any]] = None,
):
    temp = get_agent_temperature(agent)
    llm = get_llm(
        model=model, temperature=temp, streaming=streaming, callbacks=callbacks
    )
    log(f"{agent} LLM 初始化：temperature={temp}")
    return llm


def extract_first_json(text: str) -> Any:
    """
    从模型输出中**尽力**提取第一段 JSON（支持 ```json ...``` 或裸 JSON）。
    - 优先匹配代码围栏；若失败，退化到大括号/中括号的首次匹配尝试。
    - 若全部失败，抛出异常并附加输出片段，便于诊断。
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("模型输出为空，无法解析 JSON。")

    # 优先：```json ... ```
    fence = re.findall(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S)
    candidates = fence if fence else re.findall(r"(\{.*?\}|\[.*?\])", text, flags=re.S)

    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue

    preview = text.strip()
    if len(preview) > 800:
        preview = preview[:800] + "...(截断)"
    raise ValueError("未能从模型输出中解析出有效 JSON。原始输出预览:\n" + preview)


def invoke_with_json_retry(
    llm: ChatOpenAI, messages: List[Dict[str, str]], max_retries: int = 3
) -> Tuple[Any, str]:
    """
    调用 LLM 并自动重试 JSON 解析失败的情况

    Args:
        llm: ChatOpenAI 实例
        messages: 要发送的消息列表（重试时保持完全不变）
        max_retries: 最大重试次数（默认 3）

    Returns:
        (parsed_json, raw_output): 解析后的 JSON 和原始输出

    Raises:
        ValueError: 如果所有重试都失败，包含所有尝试的错误信息
    """
    errors: List[str] = []

    for attempt in range(max_retries):
        try:
            resp = llm.invoke(messages)
            raw_output = (
                resp.content if isinstance(resp.content, str) else str(resp.content)
            )
            parsed = extract_first_json(raw_output)

            if attempt > 0:
                log(f"JSON 解析重试成功（第 {attempt + 1} 次尝试）")

            return parsed, raw_output
        except ValueError as e:
            error_msg = f"第 {attempt + 1} 次尝试失败: {str(e)}"
            errors.append(error_msg)

            if attempt < max_retries - 1:
                log(
                    f"JSON 解析失败，{error_msg}，将进行重试（剩余 {max_retries - attempt - 1} 次）"
                )
            else:
                log(f"JSON 解析失败，{error_msg}，已达到最大重试次数")

    # 所有重试都失败
    all_errors = "\n".join(errors)
    raise ValueError(
        f"JSON 解析失败，已重试 {max_retries} 次均失败。\n"
        f"所有尝试的错误信息：\n{all_errors}"
    )


def record_llm_interaction(
    state: "GraphState",
    *,
    agent: str,
    iteration: int,
    messages: List[Dict[str, str]],
    raw_output: Any,
    parsed_output: Any = None,
) -> None:
    """统一记录 LLM 交互，保证日志可审计。"""
    if isinstance(raw_output, list):
        raw_text = "".join(str(part) for part in raw_output)
    elif isinstance(raw_output, str):
        raw_text = raw_output
    else:
        raw_text = str(raw_output) if raw_output is not None else ""

    entry: Dict[str, Any] = {
        "iteration": iteration,
        "agent": agent,
        "input_messages": messages,
        "raw_output": raw_text,
    }
    if parsed_output is not None:
        entry["parsed_output"] = parsed_output
    state["logs"].append(entry)

    # 根据日志级别决定打印详细内容还是简化摘要
    current_level = get_log_level()
    
    if current_level == LOG_LEVEL_DEBUG:
        # DEBUG 级别：打印完整的输入消息和输出
        pretty_messages = json.dumps(messages, ensure_ascii=False, indent=2)
        log_debug(f"{agent}（第 {iteration} 轮）输入：\n{pretty_messages}")
        if parsed_output is not None:
            pretty_parsed = json.dumps(parsed_output, ensure_ascii=False, indent=2)
            log_debug(f"{agent}（第 {iteration} 轮）输出（解析后）：\n{pretty_parsed}")
        else:
            log_debug(f"{agent}（第 {iteration} 轮）输出（原始）：\n{raw_text}")
    else:
        # INFO 级别：只打印简化摘要
        # 统计输入消息信息
        system_count = sum(1 for msg in messages if msg.get("role") == "system")
        user_count = sum(1 for msg in messages if msg.get("role") == "user")
        assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
        input_summary = f"包含 {system_count} 条系统消息、{user_count} 条用户消息"
        if assistant_count > 0:
            input_summary += f"、{assistant_count} 条助手消息"
        log_info(f"{agent}（第 {iteration} 轮）输入：{input_summary}")
        
        # 统计输出信息
        if parsed_output is not None:
            if isinstance(parsed_output, list):
                output_summary = f"解析为 {len(parsed_output)} 条记录"
                # 根据 agent 类型提供更具体的摘要
                if agent == "ReqExplore":
                    output_summary = f"解析为 {len(parsed_output)} 条需求"
                elif agent == "ReqClarify":
                    output_summary = f"解析为 {len(parsed_output)} 个评分项"
            elif isinstance(parsed_output, dict):
                output_summary = "解析为字典对象"
            else:
                output_summary = "解析为对象"
            log_info(f"{agent}（第 {iteration} 轮）输出（解析后）：{output_summary}")
            
            # 针对 ReqClarify，在 INFO 级别显示每个细化需求的评分
            if agent == "ReqClarify" and isinstance(parsed_output, list):
                for item in parsed_output:
                    rid = str(item.get("id", ""))
                    sc_val = item.get("score")
                    try:
                        sc = int(sc_val)
                        # 格式化评分为 +2, +1, 0, -1, -2
                        score_str = f"+{sc}" if sc > 0 else str(sc)
                        log_info(f"  - ID: {rid}, 评分: {score_str}")
                    except Exception:
                        continue
        else:
            output_length = len(raw_text)
            if output_length > 0:
                output_summary = f"原始输出长度 {output_length} 字符"
            else:
                output_summary = "原始输出为空"
            log_info(f"{agent}（第 {iteration} 轮）输出（原始）：{output_summary}")


# -----------------------------
# 状态结构
# -----------------------------
class GraphState(TypedDict):
    user_input: str
    reference_srs: str
    atomic_reqs_queue: List[str]  # 原子需求字符串数组
    current_atomic_req_index: int  # 当前处理的原子需求索引
    atomic_req_scores_count: Dict[int, int]  # 每个原子需求的打分次数 {atomic_req_index: count}
    atomic_req_explore_history: Dict[int, List[Dict[str, str]]]  # 每个原子需求的 ReqExplore 对话历史
    req_list: List[Dict[str, Any]]  # 存储 ReqExplore 输出的结构化需求（包含 id、text 字段）
    effective_req_list: List[Dict[str, Any]]  # 最终需求清单（用于 DocGenerate）
    scores: Dict[str, int]  # {id: score} 当前原子需求的细化需求评分
    logs: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    srs_output: str
    srs_stream_printed: bool
    ablation_mode: Optional[str]  # 消融实验模式：no-clarify 或 no-explore-clarify
    req_max_scores: Dict[str, int]  # 每个需求的历史最高分 {req_id: max_score}
    req_scores_history: Dict[str, List[int]]  # 每个需求的所有评分历史 {req_id: [score1, score2, ...]}
    req_first_iteration: Dict[str, int]  # 每个需求首次出现的迭代轮次 {req_id: iteration}
    req_last_iteration: Dict[str, int]  # 每个需求最后出现的迭代轮次 {req_id: iteration}
    req_last_clarify_iteration: Dict[str, int]  # 每个需求最后一次被评分的迭代轮次 {req_id: iteration}
    banned_ids: List[str]  # 被禁用的需求ID列表（负分需求的ID）


# -----------------------------
# 统一的提示词定义
# -----------------------------
def get_req_explore_system_prompt() -> str:
    """
    返回 ReqExplore 智能体的统一系统提示词。
    并行模式和串行模式都使用此提示词。
    """
    return """
你是「软件工程需求挖掘智能体」，角色是业务需求工程师，而不是架构师或运维工程师。

【核心目标】
围绕单条【原子需求】挖掘和演化一组「业务流程需求」：
- 说明谁在什么业务情境下，期待系统做出什么可验证的业务行为；
- 不设计技术方案，不写工程细节。
- 禁止出现“用于监控成功率/优化体验/分析使用行为”等实现导向表述；
- 如需描述，仅能从业务视角表达“相关操作应可审计、可追溯”。
- 通过评分分值推断哪些表达更接近验收方预期，并据此调整。

====================
一、输入
====================

可能包含：

1. 【原子需求】（必选）
2. 【原始需求】（可选）：背景、业务目标、约束等。
3. 【上一轮需求清单】（可选）：JSON 数组
   - 形如 [{ "id": "...", "text": "..." }, ...]
4. 【上一轮评分】（可选）：JSON 数组
   - 形如 [{ "id": "...", "score": 2 }, ...]
   - 无 reason，仅有分数。

无上一轮数据时视为首轮。

====================
二、输出
====================

1. 始终仅输出一个 JSON 数组，无任何额外文本或注释：
   [
     { "id": "REQ-001", "text": "..." },
     { "id": "REQ-002", "text": "..." }
   ]

2. 该数组表示「本轮生效版的完整业务需求清单」：
   - 包含当前认为有效的所有需求；
   - 不包含已废弃的需求。

3. 每条 "text" 必须：
   - 用业务语言描述系统行为（谁、何时、因何事、系统应做什么、业务结果是什么）；
   - 是单句或短段落，语义自洽，可直接用于业务评审与验收测试设计；
   - 不拆成列表，不写代码样式内容。

====================
三、ID 规则（硬约束）
====================

1. 全局唯一：同一对话中，每个 "id" 只绑定一个需求概念，不得复用到不同内容。
2. 首轮：从 "REQ-001" 起递增（至少三位数字）。如已有历史 id，则在当前最大 id 基础上继续。
3. 对已有需求的重写或优化必须沿用原 id。
4. 新增需求必须使用「当前已用最大 id」之后的全新 id。
5. 每轮输出中，同一 id 仅出现一次。

====================
四、分值驱动行为（仅看 score）
====================

你必须严格按以下规则处理上一轮中出现的 id：

1. score = +2（强采纳）
   - 本轮必须保留该 id；
   - 仅允许极轻微措辞调整，不改变业务含义、范围、角色、状态；
   - 不拆分、不合并。

2. score = +1（采纳）
   - 本轮保留该 id；
   - 可以在不改变核心业务含义的前提下：
     - 提高叙述清晰度；
     - 补全必要的业务前提、触发条件、可见结果；
   - 允许少量拆分：
     - 原 id 作为主需求保留；
     - 与其紧密相关的子场景可拆出 1~2 条新需求（新 id）；
     - 严禁借机引入全新业务模块或与原意无关的能力。

3. score = 0（中性）
   - 本轮必须使用同一 id 对该需求进行「明显重写」：
     - 聚焦单一业务场景；
     - 去除含糊、堆叠或多流程混写；
     - 输出为可验证的业务行为，不新增新模块。
   - 不因 0 分新建无关需求。

4. score = -1（不采纳）
   - 本轮必须删除该 id，不再输出。
   - 禁止对该 id 做轻微修改后继续使用同一 id。
   - 禁止用新 id 换皮复刻该需求的关键能力方向：
     - 视为该方向不被接受，应收缩而不是坚持。
   - 如原子需求仍有未覆盖的基础意图，可基于更保守的理解，生成极少量新需求（新 id），但必须与被 -1 文本在内容上明显不同。

5. score = -2（强不采纳）
   - 本轮及后续彻底删除该 id。
   - 将该需求的业务方向视为禁用主题，严禁用新表述再次出现。

6. 未出现在评分列表中的 id
   - 视为暂不调整：保留该 id，可做轻微措辞统一，不改变含义。

====================
五、业务导向的闭环要求
====================

每条需求应尽量在自然语言中体现：

- 业务目的与场景：服务于谁，在什么情境下使用；
- 触发条件：由哪个角色，在什么前提或事件下触发操作；
- 系统应执行的业务规则：如校验信息是否完整、决定是否受理、将信息提供给哪个后续环节；
- 对用户可见的结果：如"显示确认信息及编号""展示当前处理状态""给出拒绝原因"；
- 异常场景的业务处理：信息缺失、条件不满足、无权访问时，如何反馈；
- 权限和合规：哪些角色可执行/查看，该行为是否需可追溯（简单一句即可）。

这些要素应揉进一段自然业务描述，而非逐条罗列。

====================
六、严格弱化技术细节（关键！）
====================

除非【原子需求】或【原始需求】文本中明确写出，否则：

1. 禁止在 text 中出现或重点描述以下内容：
   - 「幂等性」「埋点」「监控指标」「性能阈值（如1秒内响应）」
   - 「IP 地址」「User-Agent」「浏览器指纹」
   - 「本地缓存」「离线同步机制」「队列重试」「加密算法」
   - 具体文件格式和大小限制列表（如 JPG/PNG/10MB 等）
   - 自动化负载均衡、集群、高可用、压缩传输等工程手段
2. 若需要表达审计或追踪，只用业务化表述：
   - 如「相关操作应可追溯，满足合规要求」。
3. 若原子需求未提及「离线模式」，不得主动加入离线能力、草稿自动保存、本地暂存等设计。
4. 若原子需求未提及「自动分配/智能路由」，不得创造案件自动分配、智能分派等新流程。
5. 若原子需求未提及「实时校验」，可以描述"提交时需要完整、真实的信息"，但不写具体前端实时校验机制。

简而言之：只描述**业务承诺和可观察行为**，不描述**实现手段和内部技术细节**。

====================
七、总结要求
====================

- 你是业务流程需求挖掘工具：写清楚"系统应该如何支持业务"，而不是"系统如何实现"。
- 遵守分值规则收敛，不在被否方向死磕。
- 每轮输出：仅一个 JSON 数组，ID 合规、语义业务化、无技术细节泛滥。
"""


def get_req_clarify_system_prompt() -> str:
    """
    返回 ReqClarify 智能体的统一系统提示词。
    并行模式和串行模式都使用此提示词。
    """
    return """
你是"需求澄清智能体"。从验收方视角出发，对需求清单逐条依据"基准 SRS"进行评分。

【评分原则】
仅以"基准 SRS"为判定依据，不得引入新需求、不得改写 SRS 原文、不得扩展范围或依据行业常识脑补。

使用 5 分制：
- +2：强采纳 —— 与 SRS 完全一致，术语/范围/角色/阈值均匹配，可视为最终表述。
- +1：采纳 —— 主体含义与 SRS 一致，仅有轻微表述差异或温和细化，不改变约束与范围。
-  0：中性 —— 与 SRS 部分对应但表述模糊、混合多点或无法从 SRS 明确验证，一般需重写澄清。
- -1：不采纳 —— 存在 SRS 未记载的新能力/新模块/新约束，或与 SRS 精神有偏差，但未形成直接冲突。
- -2：强不采纳 —— 与 SRS 明确条款相矛盾，或严重歪曲 SRS 含义/边界。

【输入模式】
当输入同时包含：
- 基准 SRS：使用 ```text``` 围栏；
- 需求清单：使用 ```json``` 围栏，格式为 [{"id": "...", "text": "..."}] 数组；

进入评分模式，仅输出一个 ```json``` 围栏包裹的评分数组，例如：
```json
[{"id":"FR-01","score":1,"reason":"与SRS条款3.2含义一致，表述略扩展"}]
```

【输出要求】

- 仅输出一个 JSON 数组，且必须置于 `json` 围栏内；
- 顺序与输入需求清单完全一致，id 精确对应；
- 每项包含字段：
  - id: 原样返回；
  - score: 按上述规则取值之一；
  - reason: 中文，≤50字，仅用于审计，描述判定原因类型，如：
    - "与SRS条款X一致"
    - "轻微扩展，无冲突"
    - "无SRS对应"
    - "术语与SRS不符"
    - "阈值与SRS冲突"
    - "表述含糊，难以比对"
    - "与SRS明确条款矛盾"
- 不提出修改建议，不给出如何优化的指导，不添加解释性文字，不输出除 JSON 外的任何内容。

【非评分模式】
 若未同时提供"需求清单(json)"与"基准 SRS(text)"：

- 不进行评分；
- 输出一个结构化 JSON 示例，提示正确输入格式；
- 不输出自然语言长文说明。

整体要求：

- 严格、稳定、可审计；
- 评分仅反映"与 SRS 一致性"程度，为后续自动化处理提供可靠信号。
   """


# -----------------------------
# 节点实现
# -----------------------------
def req_parse_node(state: GraphState, llm=None) -> GraphState:
    """ReqParse：自然语言 -> 原子需求字符串数组"""
    llm = llm or get_llm_for("ReqParse")
    log("ReqParse：开始解析用户输入")
    current_iteration = state["iteration"] + 1

    user_template = """你是"软件工程需求解析智能体"，用于从自然语言输入中提取结构化的软件工程需求点。输入为自然语言描述，输出为仅包含字符串的 JSON 数组，每个元素为一个原子需求（单句、可执行、可验证）。

输出规则：
- 输出纯 JSON 数组，仅包含字符串，不带对象结构、不带键名、不带 Markdown。
- 每个元素为一个原子需求点，例如：
  [
    "用户可以通过邮箱注册账号",
    "系统支持登录与注销功能"
  ]
- 将复合句（包含"且"、"并且"、"或"、"同时"等连接词）拆分为多个独立需求。
- 删除重复或等义内容。
- 禁止输出模糊、含糊或疑问句式的描述。
- 不附加说明、注释、分类或编号。

适用场景：软件需求提炼、需求清单整理、任务拆解、开发前置分析。

【用户需求】:
{user_input}
"""

    user = user_template.format(user_input=state["user_input"])

    messages = [{"role": "user", "content": user}]
    parsed, raw_output = invoke_with_json_retry(llm, messages, max_retries=3)
    record_llm_interaction(
        state,
        agent="ReqParse",
        iteration=current_iteration,
        messages=messages,
        raw_output=raw_output,
        parsed_output=parsed,
    )
    
    # 解析结果应该是字符串数组
    normalized: List[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                item_str = item.strip()
                if item_str:
                    normalized.append(item_str)
            elif isinstance(item, dict):
                # 兼容旧格式，提取 content 字段
                content = str(item.get("content", "")).strip()
                if content:
                    normalized.append(content)
    
    state["atomic_reqs_queue"] = normalized
    state["current_atomic_req_index"] = 0
    state["atomic_req_scores_count"] = {}
    state["atomic_req_explore_history"] = {}
    state["req_list"] = []
    state["effective_req_list"] = []
    state["iteration"] = current_iteration
    log(f"ReqParse：解析完成，共 {len(normalized)} 条原子需求")
    return state


def req_explore_node(state: GraphState, llm=None) -> GraphState:
    """ReqExplore：单条原子需求 -> 细化需求列表（支持迭代优化）"""
    llm = llm or get_llm_for("ReqExplore")
    
    # 获取当前原子需求
    atomic_req_index = state.get("current_atomic_req_index", 0)
    atomic_reqs_queue = state.get("atomic_reqs_queue", [])
    
    if atomic_req_index >= len(atomic_reqs_queue):
        log("ReqExplore：所有原子需求已处理完成")
        return state
    
    # 检查当前原子需求是否已达到评分上限，如果是则切换到下一个原子需求
    max_scores = state.get("max_iterations", 3)
    scores_count = state.get("atomic_req_scores_count", {})
    current_count = scores_count.get(atomic_req_index, 0)
    
    if current_count >= max_scores:
        log(f"ReqExplore：原子需求 {atomic_req_index + 1} 已打分 {current_count} 次（达到上限 {max_scores}），切换到下一个原子需求")
        # 移动到下一个原子需求
        atomic_req_index = atomic_req_index + 1
        state["current_atomic_req_index"] = atomic_req_index
        state["scores"] = {}  # 清空当前评分
        
        # 检查是否还有更多原子需求
        if atomic_req_index >= len(atomic_reqs_queue):
            log("ReqExplore：所有原子需求已处理完成")
            return state
    
    current_atomic_req = atomic_reqs_queue[atomic_req_index]
    log(f"ReqExplore：处理原子需求 {atomic_req_index + 1}/{len(atomic_reqs_queue)}: {current_atomic_req[:50]}...")
    
    # 获取或初始化对话历史
    explore_history = state.get("atomic_req_explore_history", {})
    if atomic_req_index not in explore_history:
        explore_history[atomic_req_index] = []
    messages_history = explore_history[atomic_req_index]
    
    # 使用统一的系统提示词
    system_template = get_req_explore_system_prompt()
    
    
    
    # 如果是首次处理，创建初始消息
    if len(messages_history) == 0:
        user_template = """【原始需求】
{user_input}

【原子需求】
{atomic_req}
"""
        initial_message = {
            "role": "user",
            "content": user_template.format(
                user_input=state["user_input"],
                atomic_req=current_atomic_req
            )
        }
        messages_history.append(initial_message)
    else:
        # 如果有评分反馈，追加到历史
        scores = state.get("scores", {})
        if scores:
            scores_arr = [{"id": rid, "score": sc} for rid, sc in scores.items()]
            scores_arr_json = json.dumps(scores_arr, ensure_ascii=False)
            feedback_message = {
                "role": "user",
                "content": f"""【上一轮评分结果】
{scores_arr_json}

请严格按照系统提示中对各分值的规定执行：
- 保留并仅微调得分为 +2 / +1 的需求；
- 对得分为 0 的需求，用相同 id 重写为更清晰单一的业务行为；
- 删除得分为 -1 / -2 的需求，不再使用其 id，不再生成相同方向的需求；
- 输出本轮"完整需求清单"，仅包含仍然有效的需求及在上述基础上产生的新需求。
"""
            }
            messages_history.append(feedback_message)
    
    # 构建完整的消息列表（系统消息 + 历史消息）
    messages = [{"role": "system", "content": system_template}] + messages_history
    
    # 调用 LLM
    parsed, raw_output = invoke_with_json_retry(llm, messages, max_retries=3)
    
    # 记录交互
    record_llm_interaction(
        state,
        agent="ReqExplore",
        iteration=state["iteration"],
        messages=messages,
        raw_output=raw_output,
        parsed_output=parsed,
    )
    
    # 保存响应到历史
    assistant_message = {"role": "assistant", "content": raw_output}
    messages_history.append(assistant_message)
    explore_history[atomic_req_index] = messages_history
    state["atomic_req_explore_history"] = explore_history
    
    # 规范化输出：确保格式为 [{"id": "...", "text": "..."}]
    # 为了确保全局唯一性，在 ID 中包含原子需求索引
    # 获取当前迭代轮次（基于当前打分次数）
    scores_count = state.get("atomic_req_scores_count", {})
    current_iteration = scores_count.get(atomic_req_index, 0)
    
    # 获取评分历史状态
    req_first_iteration = state.get("req_first_iteration", {})
    req_last_iteration = state.get("req_last_iteration", {})
    
    normalized_reqs: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        # 获取 banned_ids，防止复用被禁用的ID
        banned_ids = set(state.get("banned_ids", []))
        
        req_id_counter = 1
        for item in parsed:
            if isinstance(item, dict):
                # 如果模型返回的 ID 不包含原子需求索引，我们需要添加
                original_id = str(item.get("id", "")).strip()
                
                # 确定最终的 req_id
                if original_id and not original_id.startswith(f"A{atomic_req_index}-"):
                    req_id = f"A{atomic_req_index}-{original_id}"
                elif not original_id:
                    req_id = f"A{atomic_req_index}-REQ-{req_id_counter:03d}"
                else:
                    req_id = original_id
                
                # 检查最终生成的 req_id 是否是被禁用的ID，如果是则跳过
                if req_id in banned_ids:
                    log_debug(f"ReqExplore：跳过被禁用的ID {req_id}")
                    continue
                req_text = str(item.get("text", item.get("content", ""))).strip()
                if req_text:
                    normalized_reqs.append({
                        "id": req_id, 
                        "text": req_text, 
                        "atomic_req_index": atomic_req_index,
                        "iteration": current_iteration,
                        "score_in_iteration": None  # 将在评分后更新
                    })
                    # 记录需求首次和最后出现的迭代轮次
                    if req_id not in req_first_iteration:
                        req_first_iteration[req_id] = current_iteration
                    req_last_iteration[req_id] = current_iteration
                    req_id_counter += 1
            elif isinstance(item, str):
                # 兼容纯字符串格式，自动分配 ID
                if item.strip():
                    req_id = f"A{atomic_req_index}-REQ-{req_id_counter:03d}"
                    normalized_reqs.append({
                        "id": req_id, 
                        "text": item.strip(), 
                        "atomic_req_index": atomic_req_index,
                        "iteration": current_iteration,
                        "score_in_iteration": None  # 将在评分后更新
                    })
                    # 记录需求首次和最后出现的迭代轮次
                    if req_id not in req_first_iteration:
                        req_first_iteration[req_id] = current_iteration
                    req_last_iteration[req_id] = current_iteration
                    req_id_counter += 1
    
    # 更新状态
    state["req_first_iteration"] = req_first_iteration
    state["req_last_iteration"] = req_last_iteration
    
    # 打印生成的细化需求列表
    if normalized_reqs:
        scores_count = state.get("atomic_req_scores_count", {})
        current_scores_count = scores_count.get(atomic_req_index, 0)
        log(f"[ReqExplore] 原子需求 {atomic_req_index + 1} 第 {current_scores_count + 1} 次迭代生成细化需求：")
        for req in normalized_reqs:
            req_id = req.get("id", "")
            req_text = req.get("text", "")
            # 截断长文本（超过200字符）
            if len(req_text) > 200:
                req_text_display = req_text[:200] + "..."
            else:
                req_text_display = req_text
            log(f"  - ID: {req_id}, 文本: {req_text_display}")
    
    # 将当前原子需求的细化需求合并到 req_list
    # 先移除该原子需求之前的细化需求（如果有相同 atomic_req_index 的）
    existing_req_list = state.get("req_list", [])
    updated_req_list = [
        req for req in existing_req_list
        if req.get("atomic_req_index") != atomic_req_index
    ]
    updated_req_list.extend(normalized_reqs)
    state["req_list"] = updated_req_list
    
    log(f"ReqExplore：完成细化，生成 {len(normalized_reqs)} 条需求")
    return state


def req_clarify_node(state: GraphState, llm=None) -> GraphState:
    """ReqClarify：对当前原子需求的细化需求进行评分"""
    llm = llm or get_llm_for("ReqClarify")
    
    # 获取当前原子需求的细化需求
    atomic_req_index = state.get("current_atomic_req_index", 0)
    req_list = state.get("req_list", [])
    
    # 获取当前原子需求对应的细化需求（通过 atomic_req_index 过滤）
    current_refined_reqs = [
        req for req in req_list
        if req.get("atomic_req_index") == atomic_req_index
    ]
    
    if not current_refined_reqs:
        log("ReqClarify：当前原子需求无细化需求，跳过评分，直接进入下一个原子需求")
        state["scores"] = {}
        # 标记为已达到上限，以便进入下一个原子需求
        atomic_req_index = state.get("current_atomic_req_index", 0)
        max_scores = state.get("max_iterations", 3)
        scores_count = state.get("atomic_req_scores_count", {})
        scores_count[atomic_req_index] = max_scores  # 设置为上限，触发进入下一个原子需求
        state["atomic_req_scores_count"] = scores_count
        return state
    
    log(f"ReqClarify：对原子需求 {atomic_req_index + 1} 的 {len(current_refined_reqs)} 条细化需求进行评分")
    
    # 构建需求清单（使用 id 和 text 字段）
    req_list_for_scoring = [
        {"id": req.get("id", ""), "text": req.get("text", req.get("content", ""))}
        for req in current_refined_reqs
    ]
    req_list_json = json.dumps(req_list_for_scoring, ensure_ascii=False)
    
    # 使用统一的系统提示词
    system_template = get_req_clarify_system_prompt()
    
    user_template = """【需求清单】
```json
{req_list_json}
```

【基准 SRS】
```text
{reference_srs}
```
"""
    
    user = user_template.format(
        req_list_json=req_list_json,
        reference_srs=state["reference_srs"]
    )
    
    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user}
    ]
    
    evaluations, raw_output = invoke_with_json_retry(llm, messages, max_retries=3)
    record_llm_interaction(
        state,
        agent="ReqClarify",
        iteration=state["iteration"],
        messages=messages,
        raw_output=raw_output,
        parsed_output=evaluations,
    )
    
    # 规范化评分映射
    scores_map: Dict[str, int] = {}
    for item in evaluations:
        rid = str(item.get("id"))
        sc_val = item.get("score")
        try:
            sc = int(sc_val)
        except Exception:
            continue
        if rid:
            scores_map[rid] = sc
    
    state["scores"] = scores_map
    
    # 更新当前原子需求的打分计数
    scores_count = state.get("atomic_req_scores_count", {})
    scores_count[atomic_req_index] = scores_count.get(atomic_req_index, 0) + 1
    current_iteration = scores_count[atomic_req_index] - 1  # 当前迭代轮次（从0开始）
    state["atomic_req_scores_count"] = scores_count
    
    # 更新评分历史状态
    req_max_scores = state.get("req_max_scores", {})
    req_scores_history = state.get("req_scores_history", {})
    req_last_clarify_iteration = state.get("req_last_clarify_iteration", {})
    
    # 更新 req_list 中每个需求的 score_in_iteration 和 iteration
    for req in current_refined_reqs:
        req_id = req.get("id", "")
        if req_id in scores_map:
            sc = scores_map[req_id]
            req["score_in_iteration"] = sc
            req["iteration"] = current_iteration
            
            # 更新评分历史
            if req_id not in req_scores_history:
                req_scores_history[req_id] = []
            req_scores_history[req_id].append(sc)
            
            # 更新历史最高分
            if req_id not in req_max_scores or sc > req_max_scores[req_id]:
                req_max_scores[req_id] = sc
            
            # 记录最后一次被评分的迭代轮次
            req_last_clarify_iteration[req_id] = current_iteration
    
    # 更新状态
    state["req_max_scores"] = req_max_scores
    state["req_scores_history"] = req_scores_history
    state["req_last_clarify_iteration"] = req_last_clarify_iteration
    state["req_list"] = req_list  # 更新 req_list（已更新 score_in_iteration 和 iteration）
    
    # 打印评分结果
    if scores_map:
        log(f"[ReqClarify] 原子需求 {atomic_req_index + 1} 第 {scores_count[atomic_req_index]} 次迭代评分结果：")
        for item in evaluations:
            rid = str(item.get("id", ""))
            sc_val = item.get("score")
            reason = item.get("reason", "")
            try:
                sc = int(sc_val)
                # 格式化评分为 +2, +1, 0, -1, -2
                score_str = f"+{sc}" if sc > 0 else str(sc)
                if reason:
                    # 截断长原因（超过100字符）
                    reason_display = reason[:100] + "..." if len(reason) > 100 else reason
                    log(f"  - ID: {rid}, 评分: {score_str}, 原因: {reason_display}")
                else:
                    log(f"  - ID: {rid}, 评分: {score_str}")
            except Exception:
                continue
    
    log(f"ReqClarify：评分完成，当前原子需求已打分 {scores_count[atomic_req_index]} 次")
    
    # 在当前原子需求范围内，物理移除得分为负的需求，防止后续继续带入上下文
    negative_ids = {rid for rid, sc in scores_map.items() if sc < 0}
    if negative_ids:
        new_req_list = []
        for req in state["req_list"]:
            if req.get("atomic_req_index") == atomic_req_index and req.get("id") in negative_ids:
                continue
            new_req_list.append(req)
        state["req_list"] = new_req_list
        
        # 将负分ID加入禁用列表，防止后续迭代中复用
        banned_ids = set(state.get("banned_ids", []))
        banned_ids.update(negative_ids)
        state["banned_ids"] = list(banned_ids)
        
        log(f"ReqClarify：已物理移除 {len(negative_ids)} 条负分需求（原子需求 {atomic_req_index + 1}），并加入禁用列表")
    
    return state


def doc_generate_node(state: GraphState, llm=None) -> GraphState:
    log("DocGenerate：生成最终 Markdown SRS 文档（流式输出开始）")
    stream_handler = StreamingPrinter()
    llm = llm or get_llm_for("DocGenerate", streaming=True, callbacks=[stream_handler])

    # 使用 effective_req_list，如果没有则使用 req_list
    effective_req_list = state.get("effective_req_list", state.get("req_list", []))
    
    # 预先序列化JSON字符串用于模板填充
    effective_req_list_json = json.dumps(effective_req_list, ensure_ascii=False)

    system_template = """你将把用户提供的“需求清单、用户故事、功能点与约束”等原始材料，转化为一份正式的《软件需求规格说明书（SRS）》。文档结构严格遵循以下九节格式：

1. Problem Background
2. Stakeholders
3. Functional Requirements
4. Performance Requirements
5. Design Constraints
6. External Interfaces
7. Security Requirements
8. Use Cases for the Application
   • Actor
   • Purpose
   • Event Flow
   • Special Conditions
9. Glossary of Terms

输出语言正式、无二义、可测试；不得使用“可能、大概、尽量、类似、TBD、待定”等不确定或模糊表达。不得提出澄清问题、不得做假设或推测。若信息未提供，则不生成对应内容。

每条功能性需求应包含唯一编号与标题（如 FR-001 登录功能），正文以“系统必须”“系统不得”或“系统应”开头，确保可测试。

不得在任何部分提及内容“源自需求清单”或“来自输入”等信息，只呈现正式文档内容。

语言要求：
• 禁用含糊用词，如“可能、尽量、基本、合适、快速、用户友好、及时、稳定、等、类似、TBD、待定、如果可能、应该可以”。
• 所有性能项和度量必须明确可测，包含单位与阈值。
• 采用正式书面语风格，不使用表格或字段化格式。

当用户输入需求清单或描述时，你将直接生成遵循上述结构与规范的完整 SRS，不提出问题、不补写缺失、不做推测、不引用输入来源。
"""
    
    user_template = """【最终需求清单】
{effective_req_list_json}
"""

    user = user_template.format(effective_req_list_json=effective_req_list_json)

    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user}
    ]

    record_llm_interaction(
        state,
        agent="DocGenerate",
        iteration=state["iteration"],
        messages=messages,
        raw_output=None,
    )

    print("\n====== 实时 Markdown SRS 输出（流式） ======\n", flush=True)
    resp = llm.invoke(messages)
    print("\n====== 流式输出结束 ======\n", flush=True)

    raw_content = resp.content
    full_text = (
        raw_content
        if isinstance(raw_content, str)
        else (
            "".join(str(part) for part in raw_content)
            if isinstance(raw_content, list)
            else str(raw_content) or ""
        )
    )
    if not full_text:
        full_text = stream_handler.get_text()

    state["srs_output"] = full_text
    state["srs_stream_printed"] = True
    log("DocGenerate：流式输出完成")
    return state


# -----------------------------
# 条件路由函数
# -----------------------------
def has_more_atomic_reqs(state: GraphState) -> str:
    """判断是否还有未处理的原子需求"""
    atomic_reqs_queue = state.get("atomic_reqs_queue", [])
    current_index = state.get("current_atomic_req_index", 0)
    
    if current_index >= len(atomic_reqs_queue):
        log("条件判断：所有原子需求已处理完成，进入汇总")
        return "Aggregate"
    log(f"条件判断：还有 {len(atomic_reqs_queue) - current_index} 个原子需求待处理，进入 ReqExplore")
    return "ReqExplore"


def should_go_to_clarify(state: GraphState) -> str:
    """在 ReqExplore 之后判断是否应该进入 ReqClarify"""
    atomic_req_index = state.get("current_atomic_req_index", 0)
    max_scores = state.get("max_iterations", 3)
    scores_count = state.get("atomic_req_scores_count", {})
    
    current_count = scores_count.get(atomic_req_index, 0)
    
    if current_count >= max_scores:
        log(f"条件判断：当前原子需求已打分 {current_count} 次（达到上限 {max_scores}），进入下一个原子需求")
        # 检查是否还有更多原子需求（不修改 state，状态更新在节点函数中进行）
        next_index = atomic_req_index + 1
        atomic_reqs_queue = state.get("atomic_reqs_queue", [])
        if next_index >= len(atomic_reqs_queue):
            return "Aggregate"
        return "ReqExplore"
    
    log(f"条件判断：当前原子需求已打分 {current_count} 次（上限 {max_scores}），进入 ReqClarify 进行评分")
    return "ReqClarify"


def should_continue_explore(state: GraphState) -> str:
    """在 ReqClarify 之后判断是否应该回到 ReqExplore 继续迭代"""
    atomic_req_index = state.get("current_atomic_req_index", 0)
    max_scores = state.get("max_iterations", 3)
    scores_count = state.get("atomic_req_scores_count", {})
    
    current_count = scores_count.get(atomic_req_index, 0)
    
    if current_count >= max_scores:
        log(f"条件判断：当前原子需求已打分 {current_count} 次（达到上限 {max_scores}），进入下一个原子需求")
        # 检查是否还有更多原子需求（不修改 state，状态更新在节点函数中进行）
        next_index = atomic_req_index + 1
        atomic_reqs_queue = state.get("atomic_reqs_queue", [])
        if next_index >= len(atomic_reqs_queue):
            return "Aggregate"
        return "ReqExplore"
    
    log(f"条件判断：当前原子需求已打分 {current_count} 次（上限 {max_scores}），回到 ReqExplore 继续迭代")
    return "ReqExplore"


def should_process_next_atomic_req(state: GraphState) -> str:
    """判断是否处理下一个原子需求"""
    return has_more_atomic_reqs(state)


def aggregate_node(state: GraphState) -> GraphState:
    """汇总所有处理完的原子需求，并过滤需求
    
    过滤规则：
    - 相同ID的需求只保留最高分对应的版本（如果最高分 >= 0）
    - 如果最高分 < 0 但存在未评分的新增项，保留未评分的新增项
    """
    log("汇总：收集所有处理完的原子需求")
    req_list = state.get("req_list", [])
    
    # 获取评分和迭代信息
    req_max_scores = state.get("req_max_scores", {})
    req_last_iteration = state.get("req_last_iteration", {})
    req_last_clarify_iteration = state.get("req_last_clarify_iteration", {})
    max_iterations = state.get("max_iterations", 5)
    
    # 最后一次迭代的索引（从0开始，所以是 max_iterations - 1）
    last_iteration_index = max_iterations - 1
    
    # 按ID分组所有需求
    reqs_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for req in req_list:
        req_id = req.get("id", "")
        if not req_id:
            continue
        if req_id not in reqs_by_id:
            reqs_by_id[req_id] = []
        reqs_by_id[req_id].append(req)
    
    # 对于每个ID，找到最高分对应的版本（如果最高分 >= 0）或未评分的新增项
    effective_req_list = []
    for req_id, req_versions in reqs_by_id.items():
        max_score = req_max_scores.get(req_id, None)
        
        # 如果最高分 >= 1，保留最高分对应的版本
        if max_score is not None and max_score >= 1:
            # 找到最高分对应的版本（如果有多个，选择最新的）
            best_req = None
            best_iteration = -1
            for req in req_versions:
                score_in_iteration = req.get("score_in_iteration")
                req_iteration = req.get("iteration", -1)
                if score_in_iteration is not None and score_in_iteration == max_score:
                    # 选择迭代轮次最大的（最新的）
                    if req_iteration > best_iteration:
                        best_req = req
                        best_iteration = req_iteration
            
            # 如果没找到确切匹配的版本（可能因为评分历史记录问题），选择第一个有评分的版本
            if best_req is None:
                for req in req_versions:
                    if req.get("score_in_iteration") is not None:
                        best_req = req
                        break
            
            # 如果还是没找到，选择最后一个版本（最新的）
            if best_req is None:
                best_req = req_versions[-1]
            
            if best_req:
                effective_req_list.append(best_req)
        else:
            # 如果最高分 < 0 或不存在，检查是否有未评分的新增项
            # 未评分的新增项：在最后一次迭代中出现，但最后一次被评分的迭代轮次 < 最后出现的迭代轮次
            unrated_new_req = None
            for req in req_versions:
                req_iteration = req.get("iteration", -1)
                score_in_iteration = req.get("score_in_iteration")
                # 如果该版本在最后一次迭代中，且未被评分
                if (req_iteration == last_iteration_index and 
                    score_in_iteration is None and
                    req_last_clarify_iteration.get(req_id, -1) < req_last_iteration.get(req_id, -1)):
                    unrated_new_req = req
                    break
            
            if unrated_new_req:
                effective_req_list.append(unrated_new_req)
    
    state["effective_req_list"] = effective_req_list
    log(f"汇总：共收集 {len(req_list)} 条需求，按ID去重后保留 {len(effective_req_list)} 条需求")
    log(f"汇总：评分>=1的需求 {sum(1 for score in req_max_scores.values() if score is not None and score >= 1)} 条")
    return state


def build_graph(ablation_mode: Optional[str] = None):
    graph = StateGraph(GraphState)
    
    # 添加所有节点
    graph.add_node("ReqParse", req_parse_node)
    graph.add_node("ReqExplore", req_explore_node)
    graph.add_node("ReqClarify", req_clarify_node)
    graph.add_node("Aggregate", aggregate_node)
    graph.add_node("DocGenerate", doc_generate_node)
    
    graph.set_entry_point("ReqParse")
    
    if ablation_mode == "no-explore-clarify":
        # 模式：移除 ReqExplore + ReqClarify
        # 流程：ReqParse -> Aggregate -> DocGenerate -> END
        log("构建图结构：no-explore-clarify 模式（跳过 ReqExplore 和 ReqClarify）")
        graph.add_edge("ReqParse", "Aggregate")
        graph.add_edge("Aggregate", "DocGenerate")
    elif ablation_mode == "no-clarify":
        # 模式：移除 ReqClarify，只使用 ReqExplore
        # 流程：ReqParse -> ReqExplore -> (循环) -> Aggregate -> DocGenerate -> END
        log("构建图结构：no-clarify 模式（跳过 ReqClarify，使用串行 ReqExplore）")
        graph.add_conditional_edges(
            "ReqParse",
            has_more_atomic_reqs,
            {"ReqExplore": "ReqExplore", "Aggregate": "Aggregate"}
        )
        graph.add_conditional_edges(
            "ReqExplore",
            should_process_next_atomic_req,
            {"ReqExplore": "ReqExplore", "Aggregate": "Aggregate"}
        )
        graph.add_edge("Aggregate", "DocGenerate")
    else:
        # 默认模式：完整流程（串行处理）
        # 流程：ReqParse -> ReqExplore -> ReqClarify -> (循环) -> Aggregate -> DocGenerate -> END
        log("构建图结构：默认模式（完整流程，串行处理所有原子需求）")
        graph.add_conditional_edges(
            "ReqParse",
            has_more_atomic_reqs,
            {"ReqExplore": "ReqExplore", "Aggregate": "Aggregate"}
        )
        graph.add_conditional_edges(
            "ReqExplore",
            should_go_to_clarify,
            {"ReqClarify": "ReqClarify", "ReqExplore": "ReqExplore", "Aggregate": "Aggregate"}
        )
        graph.add_conditional_edges(
            "ReqClarify",
            should_continue_explore,
            {"ReqExplore": "ReqExplore", "Aggregate": "Aggregate"}
        )
        graph.add_edge("Aggregate", "DocGenerate")
    
    graph.add_edge("DocGenerate", END)
    return graph.compile()


class DemoInput(BaseModel):
    user_input: str = Field(..., description="自然语言需求")
    reference_srs: str = Field(..., description="基准SRS（文本）")
    max_iterations: int = 5
    ablation_mode: Optional[str] = Field(
        default=None, description="消融实验模式：no-clarify 或 no-explore-clarify"
    )


def run_demo(demo: DemoInput, silent: bool = False) -> GraphState:
    """
    运行演示流程，返回最终状态。

    Args:
        demo: 演示输入
        silent: 如果为 True，不打印输出（用于批量处理）

    Returns:
        最终状态字典
    """
    app = build_graph(ablation_mode=demo.ablation_mode)
    if not silent:
        log("启动 LangGraph 演示流程")
    init: GraphState = {
        "user_input": demo.user_input,
        "reference_srs": demo.reference_srs,
        "atomic_reqs_queue": [],
        "current_atomic_req_index": 0,
        "atomic_req_scores_count": {},
        "atomic_req_explore_history": {},
        "req_list": [],
        "effective_req_list": [],
        "scores": {},
        "logs": [],
        "iteration": 0,
        "max_iterations": demo.max_iterations,
        "srs_output": "",
        "srs_stream_printed": False,
        "ablation_mode": demo.ablation_mode,
        "req_max_scores": {},
        "req_scores_history": {},
        "req_first_iteration": {},
        "req_last_iteration": {},
        "req_last_clarify_iteration": {},
        "banned_ids": [],
    }
    config = {"recursion_limit": 1000}
    if not silent:
        log(f"设置递归限制为 1000")
    final_state = app.invoke(init, config)
    if not silent:
        log("流程结束，输出结果")
        if final_state["srs_stream_printed"]:
            print("\n====== Markdown SRS 输出已通过流式打印展示 ======\n", flush=True)
        else:
            print("\n====== 最终 Markdown SRS 输出 ======\n", flush=True)
            print(final_state["srs_output"], flush=True)

    return final_state


# 默认示例
DEFAULT_USER_INPUT = (
    "我们需要一个用户管理系统："
    "用户可注册/登录/退出；管理员可查看并按时间导出用户活动记录；"
    "系统需在高峰期保持快速响应；需要访问控制与审计。"
)
DEFAULT_REFERENCE_SRS = (
    "- FR：注册、登录、退出；管理员查看与按时间过滤导出活动；\n"
    "- NFR：响应时间 ≤ 2s，并发 ≥ 1000；\n"
    "- CON：遵从公司安全策略和访问控制规范；"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph 多智能体 SRS 生成演示脚本")
    # 1) 文件输入
    parser.add_argument(
        "-u", "--user-input", type=str, help="用户需求文本文件路径（user_input）"
    )
    parser.add_argument(
        "-r", "--reference-srs", type=str, help="参考 SRS 文本文件路径（reference_srs）"
    )
    # 2) 直接文本（优先级高于文件）
    parser.add_argument(
        "--user-text", type=str, help="用户需求文本内容，直接传入字符串"
    )
    parser.add_argument(
        "--reference-text", type=str, help="参考 SRS 文本内容，直接传入字符串"
    )
    # 3) 迭代轮数
    parser.add_argument(
        "-m", "--max-iterations", type=int, default=5, help="最大迭代轮数，默认 5"
    )
    # 4) 消融实验模式
    parser.add_argument(
        "--ablation-mode",
        type=str,
        choices=["no-clarify", "no-explore-clarify"],
        default=None,
        help="消融实验模式：no-clarify（移除需求澄清智能体）或 no-explore-clarify（移除需求挖掘+澄清智能体）",
    )

    args = parser.parse_args()

    # 参数校验与选择：优先使用直接文本；其次文件；都缺失则使用内置示例
    has_text_pair = bool(args.user_text) and bool(args.reference_text)
    has_file_pair = bool(args.user_input) and bool(args.reference_srs)

    if (args.user_text and not args.reference_text) or (
        args.reference_text and not args.user_text
    ):
        parser.error("直接文本模式必须同时提供 --user-text 与 --reference-text。")
    if (args.user_input and not args.reference_srs) or (
        args.reference_srs and not args.user_input
    ):
        parser.error("文件模式必须同时提供 --user-input 与 --reference-srs。")

    if has_text_pair:
        user_input_text = args.user_text
        reference_srs_text = args.reference_text
        log("使用命令行直接文本作为输入")
    elif has_file_pair:
        user_input_text = read_text_file(args.user_input, "user_input")
        reference_srs_text = read_text_file(args.reference_srs, "reference_srs")
    else:
        user_input_text = DEFAULT_USER_INPUT
        reference_srs_text = DEFAULT_REFERENCE_SRS
        log("未提供文本/文件参数，使用内置示例场景")

    demo = DemoInput(
        user_input=user_input_text,
        reference_srs=reference_srs_text,
        max_iterations=args.max_iterations,
        ablation_mode=args.ablation_mode,
    )
    run_demo(demo)


if __name__ == "__main__":
    main()
