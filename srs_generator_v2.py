# -*- coding: utf-8 -*-
"""
Multi-Agent SRS Generation with LangGraph
-----------------------------------------
流程：ReqParse -> ReqExplore <-> ReqClarify（全局迭代）-> Aggregate -> DocGenerate -> END

- ReqParse：自然语言 -> 原子需求字符串数组
- ReqExplore：一次性处理所有原子需求，输出全局业务需求清单（REQ-001格式，支持迭代优化）
- ReqClarify：对全局需求清单全量打分（+2 强采纳，+1 采纳，0 中性，-1 不采纳，-2 强不采纳）
- Aggregate：汇总并过滤需求（仅保留历史最高分 >= +1 的版本）
- DocGenerate：输出 Markdown（IEEE Std 830-1998 基本格式）

全局迭代机制：
- 使用 global_iteration 控制迭代轮次
- 负分需求（-1/-2）在 ReqClarify 后物理移除并加入 banned_ids，防止后续复用
- 通过 req_max_scores 记录每个需求的历史最高分，用于 Aggregate 过滤
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
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
      ReqExplore: 0.7 —— 受控挖掘，允许适度业务扩展
      ReqClarify: 0.2 —— 判定与对齐，需一致性
      DocGenerate: 0.1 —— 文档成形，稳定输出
    可通过环境变量覆盖：
      OPENAI_TEMP_REQPARSE / OPENAI_TEMP_REQEXPLORE / OPENAI_TEMP_REQCLARIFY / OPENAI_TEMP_DOCGENERATE
    """
    defaults = {
        "ReqParse": 0.2,
        "ReqExplore": 0.5,
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
    llm: ChatOpenAI, messages: List[Dict[str, str]], max_retries: int = 5
) -> Tuple[Any, str]:
    """
    调用 LLM 并自动重试 JSON 解析失败和 API 调用异常的情况

    Args:
        llm: ChatOpenAI 实例
        messages: 要发送的消息列表（重试时保持完全不变）
        max_retries: 最大重试次数（默认 5）

    Returns:
        (parsed_json, raw_output): 解析后的 JSON 和原始输出

    Raises:
        Exception: 如果所有重试都失败，包含所有尝试的错误信息
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
                log(f"重试成功（第 {attempt + 1} 次尝试）")

            return parsed, raw_output
        except ValueError as e:
            # JSON 解析错误
            error_msg = f"第 {attempt + 1} 次尝试失败（JSON 解析错误）: {str(e)}"
            errors.append(error_msg)

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                log(
                    f"JSON 解析失败，{error_msg}，等待 {wait_time} 秒后重试（剩余 {max_retries - attempt - 1} 次）"
                )
                time.sleep(wait_time)
            else:
                log(f"JSON 解析失败，{error_msg}，已达到最大重试次数")
        except Exception as e:
            # API 调用错误（网络、HTTP 状态码等）
            error_type = type(e).__name__
            error_msg = f"第 {attempt + 1} 次尝试失败（API 调用错误: {error_type}）: {str(e)}"
            errors.append(error_msg)

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                log(
                    f"API 调用失败，{error_msg}，等待 {wait_time} 秒后重试（剩余 {max_retries - attempt - 1} 次）"
                )
                time.sleep(wait_time)
            else:
                log(f"API 调用失败，{error_msg}，已达到最大重试次数")

    # 所有重试都失败
    all_errors = "\n".join(errors)
    raise Exception(
        f"LLM 调用失败，已重试 {max_retries} 次均失败。\n"
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
    global_iteration: int  # 全局迭代轮次
    req_list: List[Dict[str, Any]]  # 存储 ReqExplore 输出的结构化需求（包含 id、text 字段）
    effective_req_list: List[Dict[str, Any]]  # 最终需求清单（用于 DocGenerate）
    scores: Dict[str, int]  # {id: score} 全局需求评分
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
- 禁止出现"用于监控成功率/优化体验/分析使用行为"等实现导向表述；
- 如需描述，仅能从业务视角表达"相关操作应可审计、可追溯"。
- 通过评分分值推断哪些表达更接近验收方预期，并据此调整。
- 系统化挖掘所有合理隐含的业务需求，鼓励在业务范围内进行深度挖掘，不遗漏关键业务场景。

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
   - 允许适度拆分：
     - 原 id 作为主需求保留；
     - 与其紧密相关的子场景可拆出 1~2 条新需求（新 id），仅当该需求确实隐含多个明确的业务场景时；
     - 可以适度补全该需求隐含的必要业务要素，但不鼓励系统化挖掘大量支持性业务需求（深度挖掘应在 0 分需求时进行）；
     - 严禁借机引入全新业务模块或与原意无关的能力。

3. score = 0（中性）
   - 本轮必须使用同一 id 对该需求进行「明显重写」：
     - 聚焦单一业务场景；
     - 去除含糊、堆叠或多流程混写；
     - 输出为可验证的业务行为，不新增新模块。
   - **这是深度挖掘和扩展的主要场景**：因为原需求表述模糊、混合多点或无法从 SRS 明确验证，需要澄清。
   - 鼓励挖掘业务变体场景（正常/异常/边界情况），使用新 id 表达这些变体。
   - 鼓励系统化挖掘该需求相关的上下游业务场景、异常处理、权限控制等支持性业务需求，使用新 id 表达。
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
五、业务需求扩展策略
====================

基于评分结果，采用以下扩展策略：

1. 高分需求（+2/+1）：
   - **保守调整策略**：因为评分含义是"与 SRS 一致"或"仅有轻微表述差异"，不应大量扩展。
   - +2 分：仅允许极轻微措辞调整，不拆分、不合并。
   - +1 分：允许轻微措辞调整和补全必要的业务要素（前提、触发条件、可见结果），最多允许 1~2 条紧密相关的子场景拆分。
   - 主要目标是补全必要业务要素，确保业务闭环的完整性，但不鼓励系统化挖掘大量支持性业务需求。

2. 中性需求（0）：
   - **深度挖掘和扩展的主要场景**：因为评分含义是"表述模糊、混合多点或无法从 SRS 明确验证，需重写澄清"。
   - 明显重写该需求（使用同一 id），聚焦单一业务场景。
   - 鼓励拆分为多个明确的业务场景，每个场景聚焦单一业务行为，使用不同 id 表达。
   - 鼓励识别业务变体（正常流程、异常流程、边界情况），使用不同 id 表达。
   - 鼓励系统化挖掘该需求相关的上下游业务场景、异常处理、权限控制等支持性业务需求，使用新 id 表达。
   - 确保拆分后的需求在业务上可验证、可测试。

3. 业务完整性要求：
   - 触发条件：明确谁、在什么业务情境下、基于什么前提条件触发；
   - 处理流程：系统应执行的业务规则、决策点、状态转换；
   - 结果反馈：对用户/业务方可见的结果、确认信息、状态展示；
   - 异常处理：信息缺失、条件不满足、权限不足等业务异常的处理；
   - 权限控制：哪些角色可执行/查看、业务合规要求；
   - 业务关联：与其他业务需求的关联、依赖关系。

====================
六、业务导向的闭环要求
====================

每条需求应尽量在自然语言中体现完整的业务闭环，包括以下要素：

1. **业务触发与输入**：
   - 明确谁（角色）、在什么业务情境下、基于什么前提条件触发该业务操作；
   - 输入的业务信息、数据、状态等前置条件。

2. **业务处理流程**：
   - 系统应执行的业务规则、决策点、状态转换；
   - 业务逻辑判断（如校验信息是否完整、决定是否受理、将信息提供给哪个后续环节）。

3. **业务结果与反馈**：
   - 对用户/业务方可见的结果、确认信息、状态展示；
   - 如"显示确认信息及编号""展示当前处理状态""给出拒绝原因"。

4. **异常业务场景**：
   - 信息缺失、条件不满足、权限不足等业务异常的处理；
   - 业务异常情况下的反馈和处理流程。

5. **业务权限与合规**：
   - 哪些角色可执行/查看该业务操作；
   - 业务合规要求、可追溯性（从业务视角描述，如"相关操作应可追溯，满足合规要求"）。

6. **业务数据完整性**：
   - 业务数据的一致性要求（用业务语言描述，如"确保业务数据准确、完整"）；
   - 业务操作的幂等性（用业务语言描述，如"重复提交相同业务请求时，系统应识别并避免重复处理"）。

7. **业务关联性**：
   - 与其他业务需求的关联、依赖关系；
   - 前置业务条件、后续业务影响。

这些要素应揉进一段自然业务描述，而非逐条罗列。确保业务闭环的完整性，不遗漏关键业务场景。

====================
七、严格弱化技术细节（关键！）
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
八、业务需求挖掘指导
====================

在保持业务导向的前提下，采用以下挖掘策略：

1. **从单一需求到业务闭环**：
   - 鼓励从单一需求扩展到完整的业务场景闭环；
   - 识别并补充该业务场景所需的所有支持性业务需求。

2. **业务变体挖掘**：
   - 允许挖掘合理的业务变体（正常流程、异常流程、边界情况）；
   - 每个变体应聚焦单一业务场景，使用独立的 id。

3. **业务依赖关系**：
   - 鼓励识别业务依赖关系（前置条件、后续影响）；
   - 明确业务需求之间的关联和依赖，确保业务逻辑的完整性。

4. **业务语言描述**：
   - 所有需求必须用业务语言描述，避免技术实现细节；
   - 从业务视角表达系统应支持的业务行为，而非技术实现手段。

5. **系统化挖掘**：
   - 对于高分需求，系统化挖掘其隐含的业务场景和支持性需求；
   - 确保不遗漏关键业务场景，保持业务闭环的完整性。

====================
九、总结要求
====================

- 你是业务流程需求挖掘工具：写清楚"系统应该如何支持业务"，而不是"系统如何实现"。
- 系统化挖掘业务需求，不遗漏关键业务场景，确保业务闭环完整性。
- 遵守分值规则收敛，不在被否方向死磕。
- 在业务范围内进行深度扩展，保持业务导向，避免技术细节泛滥。
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
[{"id":"FR-01","score":"+1"}]
```

【输出要求】

- 仅输出一个 JSON 数组，且必须置于 `json` 围栏内；
- 顺序与输入需求清单完全一致，id 精确对应；
- 每项仅包含两个字段：
  - id: 原样返回；
  - score: 字符串类型，按上述规则取值之一（"+2"、"+1"、"0"、"-1"、"-2"）；
- **不需要 reason 字段**，仅输出 id 和 score 即可，降噪和节省 token；
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
    state["global_iteration"] = 0
    state["req_list"] = []
    state["effective_req_list"] = []
    state["scores"] = {}
    state["iteration"] = current_iteration
    log(f"ReqParse：解析完成，共 {len(normalized)} 条原子需求")
    return state


def req_explore_node(state: GraphState, llm=None) -> GraphState:
    """ReqExplore（全局版）：一次性处理所有原子需求，输出全局业务需求清单"""
    llm = llm or get_llm_for("ReqExplore")
    
    global_iteration = state.get("global_iteration", 0)
    atomic_reqs_queue = state.get("atomic_reqs_queue", [])
    user_input = state.get("user_input", "")
    prev_req_list = state.get("req_list", [])
    prev_scores = state.get("scores", {})
    
    log(f"ReqExplore（全局版）：第 {global_iteration + 1} 轮迭代")
    
    # 使用统一的系统提示词
    system_template = get_req_explore_system_prompt()
    
    # 构建用户消息
    user_parts = []
    
    # 1. 原始需求全文
    user_parts.append(f"""【原始需求】
{user_input}""")
    
    # 2. 全量原子需求数组
    if atomic_reqs_queue:
        atomic_reqs_text = "\n".join(f"- {req}" for req in atomic_reqs_queue)
        user_parts.append(f"""【原子需求】
{atomic_reqs_text}""")
    
    # 3. 上一轮需求清单（如有）
    if prev_req_list:
        prev_req_list_json = json.dumps(
            [{"id": req.get("id", ""), "text": req.get("text", "")} for req in prev_req_list],
            ensure_ascii=False
        )
        user_parts.append(f"""【上一轮需求清单】
```json
{prev_req_list_json}
```""")
    
    # 4. 上一轮评分（如有，仅分值）
    if prev_scores:
        scores_arr = [{"id": rid, "score": sc} for rid, sc in prev_scores.items()]
        scores_arr_json = json.dumps(scores_arr, ensure_ascii=False)
        user_parts.append(f"""【上一轮评分】
```json
{scores_arr_json}
```

请严格按照系统提示中对各分值的规定执行：
- 保留并仅微调得分为 +2 / +1 的需求；
- 对得分为 0 的需求，用相同 id 重写为更清晰单一的业务行为；
- 删除得分为 -1 / -2 的需求，不再使用其 id，不再生成相同方向的需求；
- 输出本轮"完整需求清单"，仅包含仍然有效的需求及在上述基础上产生的新需求。""")
    
    user_content = "\n\n".join(user_parts)
    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user_content}
    ]
    
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
    
    # 规范化输出：确保格式为 [{"id": "...", "text": "..."}]
    global_iteration = state.get("global_iteration", 0)
    req_first_iteration = state.get("req_first_iteration", {})
    req_last_iteration = state.get("req_last_iteration", {})
    banned_ids = set(state.get("banned_ids", []))
    
    # 计算当前最大 ID
    max_id_num = 0
    if prev_req_list:
        for req in prev_req_list:
            req_id = req.get("id", "")
            # 尝试提取 ID 中的数字部分（如 REQ-005 -> 5）
            match = re.search(r"REQ-(\d+)", req_id)
            if match:
                try:
                    id_num = int(match.group(1))
                    if id_num > max_id_num:
                        max_id_num = id_num
                except Exception:
                    pass
    
    normalized_reqs: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        req_id_counter = max_id_num + 1  # 从最大 ID 后续排
        
        for item in parsed:
            if isinstance(item, dict):
                original_id = str(item.get("id", "")).strip()
                req_text = str(item.get("text", item.get("content", ""))).strip()
                
                if not req_text:
                    continue
                
                # ID 处理：沿用高分 ID，新需求在当前最大 ID 后续排
                if original_id:
                    # 检查是否是已存在的 ID（高分需求应沿用原 ID）
                    req_id = original_id
                    # 如果 ID 被禁用，跳过
                    if req_id in banned_ids:
                        log_debug(f"ReqExplore：跳过被禁用的ID {req_id}")
                        continue
                else:
                    # 新需求，使用新 ID
                    req_id = f"REQ-{req_id_counter:03d}"
                    req_id_counter += 1
                
                # 检查 ID 是否被禁用
                if req_id in banned_ids:
                    log_debug(f"ReqExplore：跳过被禁用的ID {req_id}")
                    continue
                
                normalized_reqs.append({
                    "id": req_id,
                    "text": req_text,
                    "iteration": global_iteration,
                    "score_in_iteration": None  # 将在评分后更新
                })
                
                # 记录需求首次和最后出现的迭代轮次
                if req_id not in req_first_iteration:
                    req_first_iteration[req_id] = global_iteration
                req_last_iteration[req_id] = global_iteration
                
            elif isinstance(item, str):
                # 兼容纯字符串格式，自动分配 ID
                if item.strip():
                    req_id = f"REQ-{req_id_counter:03d}"
                    req_id_counter += 1
                    
                    if req_id in banned_ids:
                        log_debug(f"ReqExplore：跳过被禁用的ID {req_id}")
                        continue
                    
                    normalized_reqs.append({
                        "id": req_id,
                        "text": item.strip(),
                        "iteration": global_iteration,
                        "score_in_iteration": None  # 将在评分后更新
                    })
                    
                    # 记录需求首次和最后出现的迭代轮次
                    if req_id not in req_first_iteration:
                        req_first_iteration[req_id] = global_iteration
                    req_last_iteration[req_id] = global_iteration
    
    # 更新状态
    state["req_first_iteration"] = req_first_iteration
    state["req_last_iteration"] = req_last_iteration
    state["req_list"] = normalized_reqs  # 直接替换为新的全局清单
    
    # 打印生成的细化需求列表
    if normalized_reqs:
        log(f"[ReqExplore] 第 {global_iteration + 1} 轮迭代生成全局需求清单（{len(normalized_reqs)} 条）：")
        for req in normalized_reqs:
            req_id = req.get("id", "")
            req_text = req.get("text", "")
            # 截断长文本（超过200字符）
            if len(req_text) > 200:
                req_text_display = req_text[:200] + "..."
            else:
                req_text_display = req_text
            log(f"  - ID: {req_id}, 文本: {req_text_display}")
    
    log(f"ReqExplore：完成全局需求清单生成，共 {len(normalized_reqs)} 条需求")
    return state


def req_clarify_node(state: GraphState, llm=None) -> GraphState:
    """ReqClarify（全局版）：对全局需求清单全量打分"""
    llm = llm or get_llm_for("ReqClarify")
    
    req_list = state.get("req_list", [])
    global_iteration = state.get("global_iteration", 0)
    
    if not req_list:
        log("ReqClarify：无需求清单，跳过评分")
        state["scores"] = {}
        return state
    
    log(f"ReqClarify（全局版）：对 {len(req_list)} 条全局需求进行评分")
    
    # 构建需求清单（使用 id 和 text 字段）
    req_list_for_scoring = [
        {"id": req.get("id", ""), "text": req.get("text", req.get("content", ""))}
        for req in req_list
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
    
    # 规范化评分映射（仅提取 id 和 score，忽略 reason）
    scores_map: Dict[str, int] = {}
    for item in evaluations:
        rid = str(item.get("id", ""))
        sc_val = item.get("score")
        try:
            sc = int(sc_val)
        except Exception:
            continue
        if rid:
            scores_map[rid] = sc
    
    state["scores"] = scores_map
    
    # 更新评分历史状态
    req_max_scores = state.get("req_max_scores", {})
    req_scores_history = state.get("req_scores_history", {})
    req_last_clarify_iteration = state.get("req_last_clarify_iteration", {})
    
    # 更新 req_list 中每个需求的 score_in_iteration 和 iteration
    for req in req_list:
        req_id = req.get("id", "")
        if req_id in scores_map:
            sc = scores_map[req_id]
            req["score_in_iteration"] = sc
            req["iteration"] = global_iteration
            
            # 更新评分历史
            if req_id not in req_scores_history:
                req_scores_history[req_id] = []
            req_scores_history[req_id].append(sc)
            
            # 更新历史最高分
            if req_id not in req_max_scores or sc > req_max_scores[req_id]:
                req_max_scores[req_id] = sc
            
            # 记录最后一次被评分的迭代轮次
            req_last_clarify_iteration[req_id] = global_iteration
    
    # 更新状态
    state["req_max_scores"] = req_max_scores
    state["req_scores_history"] = req_scores_history
    state["req_last_clarify_iteration"] = req_last_clarify_iteration
    state["req_list"] = req_list  # 更新 req_list（已更新 score_in_iteration 和 iteration）
    
    # 打印评分结果
    if scores_map:
        log(f"[ReqClarify] 第 {global_iteration + 1} 轮迭代评分结果：")
        for item in evaluations:
            rid = str(item.get("id", ""))
            sc_val = item.get("score")
            try:
                sc = int(sc_val)
                # 格式化评分为 +2, +1, 0, -1, -2
                score_str = f"+{sc}" if sc > 0 else str(sc)
                log(f"  - ID: {rid}, 评分: {score_str}")
            except Exception:
                continue
    
    log(f"ReqClarify：评分完成，共评分 {len(scores_map)} 条需求")
    
    # 物理移除得分为负的需求，防止后续继续带入上下文
    negative_ids = {rid for rid, sc in scores_map.items() if sc < 0}
    if negative_ids:
        new_req_list = [req for req in state["req_list"] if req.get("id") not in negative_ids]
        state["req_list"] = new_req_list
        
        # 将负分ID加入禁用列表，防止后续迭代中复用
        banned_ids = set(state.get("banned_ids", []))
        banned_ids.update(negative_ids)
        state["banned_ids"] = list(banned_ids)
        
        log(f"ReqClarify：已物理移除 {len(negative_ids)} 条负分需求，并加入禁用列表")
    
    # 更新全局迭代轮次
    state["global_iteration"] = global_iteration + 1
    
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
    
    # 流式输出重试机制
    max_retries = 5
    resp = None
    errors: List[str] = []
    
    for attempt in range(max_retries):
        try:
            resp = llm.invoke(messages)
            if attempt > 0:
                log(f"流式输出重试成功（第 {attempt + 1} 次尝试）")
            break
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"第 {attempt + 1} 次尝试失败（API 调用错误: {error_type}）: {str(e)}"
            errors.append(error_msg)
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                log(
                    f"流式输出 API 调用失败，{error_msg}，等待 {wait_time} 秒后重试（剩余 {max_retries - attempt - 1} 次）"
                )
                time.sleep(wait_time)
                # 重新创建 stream_handler 和 llm 以重置状态
                stream_handler = StreamingPrinter()
                llm = get_llm_for("DocGenerate", streaming=True, callbacks=[stream_handler])
            else:
                log(f"流式输出 API 调用失败，{error_msg}，已达到最大重试次数")
                all_errors = "\n".join(errors)
                raise Exception(
                    f"流式输出 LLM 调用失败，已重试 {max_retries} 次均失败。\n"
                    f"所有尝试的错误信息：\n{all_errors}"
                )
    
    if resp is None:
        raise Exception("流式输出调用失败：未获得响应")
    
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
def should_continue_global(state: GraphState) -> str:
    """判断是否继续全局迭代"""
    global_iteration = state.get("global_iteration", 0)
    max_iterations = state.get("max_iterations", 5)
    
    if global_iteration < max_iterations:
        log(f"条件判断：全局迭代 {global_iteration + 1}/{max_iterations}，继续迭代")
        return "ReqExplore"
    else:
        log(f"条件判断：全局迭代已达到上限 {max_iterations}，进入汇总")
        return "Aggregate"


def aggregate_node(state: GraphState) -> GraphState:
    """汇总全局需求，并过滤需求
    
    过滤规则：
    - 仅保留历史最高分 >= +1 的版本
    - 0 分视为"过程态，需要澄清"，不进最终清单
    - 负分在 Clarify 后直接剪掉，不会进入 Aggregate
    """
    log("汇总：收集全局需求清单")
    req_list = state.get("req_list", [])
    
    # 获取评分信息
    req_max_scores = state.get("req_max_scores", {})
    
    # 按ID分组所有需求
    reqs_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for req in req_list:
        req_id = req.get("id", "")
        if not req_id:
            continue
        if req_id not in reqs_by_id:
            reqs_by_id[req_id] = []
        reqs_by_id[req_id].append(req)
    
    # 对于每个ID，仅保留历史最高分 >= +1 的版本
    effective_req_list = []
    for req_id, req_versions in reqs_by_id.items():
        max_score = req_max_scores.get(req_id, None)
        
        # 仅保留历史最高分 >= +1 的版本
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
        # 0 分和负分都不进入最终清单（0 分视为过程态，负分已在 Clarify 中移除）
    
    state["effective_req_list"] = effective_req_list
    log(f"汇总：共收集 {len(req_list)} 条需求，按ID去重后保留 {len(effective_req_list)} 条需求（仅保留历史最高分 >= +1）")
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
        log("构建图结构：no-clarify 模式（跳过 ReqClarify，使用全局 ReqExplore）")
        graph.add_edge("ReqParse", "ReqExplore")
        graph.add_conditional_edges(
            "ReqExplore",
            should_continue_global,
            {"ReqExplore": "ReqExplore", "Aggregate": "Aggregate"}
        )
        graph.add_edge("Aggregate", "DocGenerate")
    else:
        # 默认模式：完整流程（全局一锅炖）
        # 流程：ReqParse -> ReqExplore -> ReqClarify -> (循环 K 次) -> Aggregate -> DocGenerate -> END
        log("构建图结构：默认模式（完整流程，全局一锅炖）")
        graph.add_edge("ReqParse", "ReqExplore")
        graph.add_edge("ReqExplore", "ReqClarify")
        graph.add_conditional_edges(
            "ReqClarify",
            should_continue_global,
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
        "global_iteration": 0,
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
