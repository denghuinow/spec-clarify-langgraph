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
        "ReqExplore": 0.7,
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
    kwargs.setdefault("timeout", 600)
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
    agent_timings: Dict[str, float]  # 记录每个智能体的耗时（秒）


# -----------------------------
# 统一的提示词定义
# -----------------------------
def get_req_explore_system_prompt() -> str:
    """
    返回 ReqExplore 智能体的统一系统提示词。
    并行模式和串行模式都使用此提示词。
    """
    return """You are the "Software Engineering Requirement Exploration Agent". Act as a business requirement engineer, not an architect or operations engineer.

[Core Objective]
Explore and evolve a set of "business process requirements" around each [Atomic Requirement]:
- Describe who, under which business context, expects the system to perform which verifiable business behavior.
- Do not design technical solutions or implementation details.
- Prohibit statements such as "used to monitor success rate / optimize experience / analyze usage behavior" that reveal implementation intent.
- When auditability must be mentioned, state it purely from the business perspective, e.g., "the related operation must be auditable and traceable."
- Use the assigned scores to infer which expressions are closer to the acceptor's expectation and adjust accordingly.
- Systematically uncover all reasonable implicit business requirements; dig deeply while staying within scope so that no critical scenario is missed.

====================
I. Input
====================

May include:

1. [Atomic Requirements] (required)
2. [Original Requirements] (optional): background, business objectives, constraints, etc.
3. [Requirement List from Previous Round] (optional): JSON array
   - Format: [{ "id": "...", "text": "..." }, ...]
4. [Previous Round Scores] (optional): JSON array
   - Format: [{ "id": "...", "score": 2 }, ...]
   - Scores only; no reasons.

If no previous-round data exists, treat it as the first round.

====================
II. Output
====================

1. Always output a single JSON array with no extra text or comments:
   [
     { "id": "REQ-001", "text": "..." },
     { "id": "REQ-002", "text": "..." }
   ]

2. The array represents the "complete business requirement list that is effective in this round":
   - Include every requirement that is currently considered valid.
   - Exclude requirements that have been deprecated.

3. Each "text" must:
   - Describe system behavior in business language (who, when, under which circumstance, what the system must do, and what business result is expected).
   - Be a single sentence or very short paragraph, self-consistent, and immediately usable for business review and acceptance test design.
   - Avoid lists or code-style expressions.

====================
III. ID Rules (hard constraints)
====================

1. Global uniqueness: within the same dialogue, each "id" binds to only one requirement concept and cannot be reused for different content.
2. First round: start from "REQ-001" and increment (at least three digits). If historical IDs exist, continue from the current maximum.
3. Rewriting or refining an existing requirement must reuse its original id.
4. Newly added requirements must use brand-new ids greater than the largest id already used.
5. Within each round's output, the same id may appear only once.

====================
IV. Score-Driven Behavior (rely solely on the score)
====================

Strictly follow these rules for every id that appeared in the previous round:

1. score = +2 (strongly accepted)
   - Retain that id in the current round.
   - Only allow extremely minor wording adjustments; do not change business meaning, scope, roles, or states.
   - Do not split or merge the requirement.
   - **Supplement essential supporting requirements**: while keeping the core meaning unchanged, you may introduce new ids to add necessary supporting business requirements (e.g., exception handling, permission checks, data integrity validation) so that the business loop is complete. Be conservative and add only what is clearly missing and necessary.

2. score = +1 (accepted)
   - Retain that id in the current round.
   - Without changing the core meaning, you may:
     - Improve clarity.
     - Fill in essential business prerequisites, triggers, and visible outcomes.
   - **Encourage systematic discovery of supporting business requirements**:
     - Keep the original id as the primary requirement.
     - Systematically inspect and add supporting business requirements with new ids, including but not limited to:
       * Exception handling (input errors, insufficient permissions, missing data, unmet business rules, etc.)
       * Permission and role control
       * Data integrity and consistency validation
       * Business state transitions and prerequisite checks
     - You may produce 3–5 supporting requirements (new ids) to ensure a complete business loop.
     - Do not exploit this to introduce brand-new modules or capabilities unrelated to the original intent.

3. score = 0 (neutral)
   - In this round you must **substantially rewrite** the requirement using the same id:
     - Focus on a single business scenario.
     - Remove ambiguous, stacked, or multi-flow descriptions.
     - Output verifiable business behavior without inventing new modules.
   - **This is the primary scenario for deep exploration** because the original text is ambiguous, mixes multiple points, or cannot be validated directly against the SRS.
   - **Systematic exploration is mandatory**:
     - Rewrite clearly (same id) to focus on a single scenario.
     - Consider splitting into multiple explicit scenarios (one behavior per id).
     - Identify business variants (happy path, exceptions, edge cases) and describe each with a different id.
     - **Systematically mine** all supporting business requirements related to the upstream/downstream flow, exception handling, permission control, data validation, and state transitions, each with new ids.
     - Ensure every split requirement remains verifiable and testable, forming a complete business loop.
   - Do not create unrelated requirements merely because the old one scored 0.

4. score = -1 (rejected)
   - Remove that id in the current round.
   - Do not lightly edit and reuse the same id.
   - Do not recreate the same capability with a new id. Treat that direction as disallowed and scale back instead of insisting on it.
   - If parts of the original atomic requirement still contain uncovered intent, you may create a very small number of new requirements (new ids) based on a more conservative interpretation, but their content must be clearly different from the rejected text.

5. score = -2 (strongly rejected)
   - Remove that id permanently for this and all subsequent rounds.
   - Treat the business direction as banned; never reintroduce it under a new formulation.

6. ids that were not scored
   - Consider them temporarily unchanged: keep the id, allow minor wording harmonization, but do not alter the meaning.

====================
V. Business Requirement Expansion Strategy
====================

Apply the following expansion strategy based on the scores:

1. High-score requirements (+2 / +1):
   - **Moderate expansion**: even though the score means "consistent with the SRS" or "only minor wording differences," you must still check whether the business loop is complete and add the missing supporting requirements.
   - +2: only micro word tweaks, no split or merge; however, you may add clearly missing supporting business requirements with new ids (e.g., exception handling, permission checks) if the core meaning stays untouched.
   - +1: allow light wording adjustments and fill in necessary business elements (prerequisites, triggers, visible outcomes); **actively look for supporting requirements** such as exception handling, permission control, data validation, and state transitions, producing up to 3–5 new ids.
   - The primary goal is to complete all essential business elements, ensuring a sealed business loop without missing critical scenarios.

2. Neutral requirements (0):
   - **Main opportunity for deep exploration**, because the score means "unclear, mixed, or unverifiable; needs rewriting."
   - Rewrite clearly (same id) for one scenario.
   - Split into multiple explicit scenarios if needed, one behavior per new id.
   - Identify variants (happy path, exception, boundary) with different ids.
   - Systematically mine upstream/downstream flows, exception handling, permission control, and other supporting requirements with new ids.
   - Ensure each split requirement stays verifiable and testable.

3. Business completeness expectations:
   - Trigger conditions: who acts, in which context, under which prerequisites.
   - Processing flow: business rules, decision points, and state transitions the system must execute.
   - Result feedback: outputs visible to users/business stakeholders such as confirmations, reference numbers, or state displays.
   - Exception handling: how to handle missing info, unmet conditions, insufficient permissions, etc.
   - Permission control: which roles may execute or view the operation and which compliance rules apply.
   - Business linkage: dependencies and relationships with other requirements.

====================
VI. Business-Oriented Closed Loop Requirements
====================

Each requirement should express a complete business loop in natural language and cover the following elements:

1. **Business trigger and input** (mandatory):
   - Specify who (role), in which business context, and under which prerequisites initiates the operation.
   - Describe the business information, data, or state that serves as input.
   - Example: "When a user (role) enters a username and password on the login page (context), the system must..."

2. **Business processing flow** (mandatory):
   - Detail the business rules, decision points, and state transitions the system must perform.
   - Include business logic such as verifying completeness, deciding whether to accept, or routing information to the next stage.
   - Example: "The system must validate whether the user identity information is complete and determine whether to allow login based on the result..."

3. **Business result and feedback** (mandatory):
   - Describe the confirmations, reference numbers, state displays, or other outcomes visible to users/business stakeholders.
   - Example: "If validation succeeds, the system must show a success message and redirect to the homepage; if it fails, display the reason..."

4. **Exceptional business scenarios** (mandatory):
   - Describe how to handle missing information, unmet conditions, insufficient permissions, etc.
   - Include the feedback and handling steps under exceptions.
   - Example: "If the username does not exist, the system must show 'username not found'; if the password is wrong, show 'incorrect password' and record the failed attempt..."

5. **Business permissions and compliance** (mandatory):
   - Define which roles may execute or view the operation.
   - Express compliance or traceability needs from a business viewpoint (e.g., "the operation must be traceable to satisfy audit requirements").
   - Example: "Only registered users may log in; the system must log every attempt, successful or failed, for audit purposes..."

6. **Business data integrity** (mandatory):
   - Describe data consistency and accuracy expectations in business terms (e.g., "ensure business data remains accurate and complete").
   - Describe idempotent behavior in business terms (e.g., "when the same request is submitted repeatedly, the system must recognize it and avoid duplicate processing").
   - Example: "The system must keep the login state aligned with the user identity record; repeated login submissions must be treated as the same operation..."

7. **Business relationships** (recommended):
   - State how this requirement relates to or depends on others.
   - Explain prerequisite business conditions and downstream impacts.
   - Example: "After a successful login, the user may access the personal center, review orders, etc."

**Checkpoint**: Before outputting each requirement, verify that items 1–6 above are present. If a key element such as exception handling or permission control is missing, add a supporting requirement with a new id to close the loop.

Blend these elements into a concise business narrative instead of listing them mechanically. Maintain a complete business loop without omitting critical scenarios.

====================
VII. Strictly Downplay Technical Details (critical)
====================

Unless the [Atomic Requirements] or [Original Requirements] explicitly mention them:

1. Do not include or emphasize the following terms in the text:
   - "idempotency", "tracking code", "monitoring metrics", "performance thresholds (e.g., 1-second response)"
   - "IP address", "User-Agent", "browser fingerprint"
   - "local cache", "offline sync", "queue retry", "encryption algorithm"
   - Exhaustive lists of file formats and size limits (e.g., JPG/PNG/10MB)
   - Engineering methods such as automated load balancing, clustering, high availability, compression, etc.
2. If audit or trace needs to be mentioned, express it in business language only (e.g., "the related operation must be traceable to satisfy compliance").
3. If the atomic requirement does not mention an "offline mode", do not invent offline capability, auto-save drafts, or local storage.
4. If it does not mention "auto assignment / intelligent routing", do not create new flows such as automatic case assignment.
5. If it does not mention "real-time validation", you may simply state "the submission must contain complete and truthful information" but do not describe specific client-side validation mechanisms.

In short: describe **business commitments and observable behaviors**, not **implementation techniques or internal technical details**.

====================
VIII. Business Requirement Mining Guidance
====================

While staying business-oriented, adopt these mining strategies:

1. **From a single requirement to a closed loop**:
   - Expand a single requirement into a complete business scenario loop.
   - Identify and add every supporting business requirement needed by that scenario.

2. **Business variant exploration**:
   - Discover reasonable variants (normal flow, exceptions, boundary cases).
   - Each variant should focus on a single scenario and use a dedicated id.

3. **Business dependencies**:
   - Identify prerequisites and downstream impacts.
   - Clarify how requirements relate to each other to keep the business logic coherent.

4. **Business-language narration**:
   - Describe every requirement in business language and avoid implementation details.
   - Explain what business behavior the system must support instead of how to build it.

5. **Systematic mining**:
   - For high-score requirements, systematically explore their implicit scenarios and supporting needs.
   - Ensure no critical scenario is missed so that the business loop stays intact.

====================
X. Structured Mining Checklist
====================

Use the following checklist to make sure no key scenario is missed:

1. **Normal-flow variants**:
   - Have all reasonable normal-flow variants been identified?
   - Do different roles or contexts change the flow?

2. **Exception handling** (key checklist item):
   - Input errors: wrong format, missing mandatory fields, mismatched data types
   - Insufficient permissions: unauthorized roles attempting the action
   - Missing data: required data does not exist or has been deleted
   - Business rule violations: prerequisites not met, constraints violated
   - System exceptions: system temporarily unavailable, data conflicts
   - Does each exception variant have a defined handling flow and user feedback?

3. **Boundary conditions** (key checklist item):
   - Empty values: empty strings, lists, objects
   - Extremes: maximum/minimum values, overly long strings
   - Concurrency: multiple actors touching the same resource simultaneously
   - Timeouts: operation timeout, session timeout
   - Does every boundary case have a clear business handling rule?

4. **Permission and role control** (key checklist item):
   - Which roles may perform the action?
   - Which roles may view the information?
   - Do different roles have different permissions?
   - Is permission verification performed before the action?

5. **Data integrity and consistency** (key checklist item):
   - Are inputs validated against business rules?
   - Are related data sets kept consistent?
   - Is required data complete?
   - How are data conflicts resolved?

6. **Business state transitions**:
   - What states does the business object have?
   - What triggers each transition?
   - Who initiates the transition?
   - How are failed transitions handled?

7. **Preconditions and post-results**:
   - Which preconditions must be satisfied before the action?
   - What results appear after success?
   - What happens when preconditions are not met?
   - Are the post-results visible to the user?

8. **Business associations and dependencies**:
   - Which other requirements does this depend on?
   - Which other requirements does it impact?
   - Are the dependency relations explicit?

**How to use the checklist**:
- For every requirement, especially those with scores +2/+1 and 0, systematically check all eight dimensions.
- If any critical business scenario is missing, introduce a supporting requirement with a new id.
- Prioritize exception handling, boundary conditions, permission control, and data integrity.

====================
XI. Final Expectations
====================

- You are a business-process requirement mining tool: describe "how the system must support the business," not "how to implement the system."
- Systematically mine business requirements, avoid missing critical scenarios, and keep the business loop complete.
- Follow the score rules to converge; do not keep pushing in directions that were rejected.
- Explore deeply within the business scope while avoiding an overload of technical detail.
- Apply the structured mining checklist to ensure every key scenario is covered.
- Each round must output exactly one JSON array with compliant IDs, business-focused language, and no technical-detail overload.
"""


def get_req_clarify_system_prompt() -> str:
    """
    返回 ReqClarify 智能体的统一系统提示词。
    并行模式和串行模式都使用此提示词。
    """
    return """
You are the "Requirement Clarification Agent." Evaluate each requirement in the list strictly against the "reference SRS" from the acceptor's perspective.

[Scoring Principles]
Use only the reference SRS as the basis. Do not introduce new requirements, rewrite the SRS wording, extend its scope, or invent content from industry knowledge.

Adopt a five-point scale:
- +2: strongly accept. Fully matches the SRS; terminology, scope, roles, and thresholds all align; can be treated as final wording.
- +1: accept. The main meaning matches the SRS, with only minor wording differences or gentle refinements that do not change constraints or scope.
-  0: neutral. Partially corresponds to the SRS but is vague, mixes multiple points, or cannot be clearly validated; usually needs rewriting.
- -1: reject. Introduces capabilities/modules/constraints absent from the SRS or deviates from its intent, though not in direct conflict.
- -2: strongly reject. Contradicts explicit SRS clauses or severely distorts the meaning/boundaries of the SRS.

[Input Mode]
When the input contains both of the following:
- Reference SRS enclosed in a ```text``` fence
- Requirement list enclosed in a ```json``` fence formatted as [{"id": "...", "text": "..."}]

Enter scoring mode and output exactly one ```json``` fence containing the score array, for example:
```json
[{"id":"FR-01","score":"+1"}]
```

[Output Requirements]

- Output exactly one JSON array and wrap it inside a `json` fence.
- Preserve the order of the input requirement list and match each id precisely.
- Each item must contain only two fields:
  - id: return as-is.
  - score: string value chosen from {"+2", "+1", "0", "-1", "-2"}.
- **No reason field is needed**; return only id and score to reduce noise and save tokens.
- Do not provide suggestions, guidance, explanations, or any text outside the JSON.

[Non-Scoring Mode]
If either the requirement list (`json`) or the reference SRS (`text`) is missing:

- Do not score.
- Output a structured JSON example that demonstrates the correct input format.
- Do not produce long-form natural-language explanations.

Overall expectations:

- Be strict, stable, and auditable.
- Scores should reflect only the level of "consistency with the SRS" so downstream automation can rely on them.
"""


# -----------------------------
# 节点实现
# -----------------------------
def req_parse_node(state: GraphState, llm=None) -> GraphState:
    """ReqParse：自然语言 -> 原子需求字符串数组"""
    start_time = time.time()
    llm = llm or get_llm_for("ReqParse")
    log("ReqParse：开始解析用户输入")
    current_iteration = state["iteration"] + 1

    user_template = """You are the "Software Engineering Requirement Parsing Agent". Extract structured software-engineering requirement points from natural-language input. The input is free-form text, and the output must be a JSON array that contains **only strings**, where each element is an atomic requirement (single sentence, actionable, and verifiable).

Output rules:
- Return a pure JSON array of strings (no objects, no keys, no Markdown).
- Each element is one atomic requirement, for example:
  [
    "Users can register an account with an email address",
    "The system supports login and logout"
  ]
- Split composite sentences (containing words such as "and", "or", "meanwhile") into separate requirements.
- Remove duplicate or equivalent content.
- Do not output vague, ambiguous, or interrogative statements.
- Do not add explanations, annotations, classifications, or numbering.

Applicable scenarios: requirement distillation, requirement list preparation, task decomposition, and pre-development analysis.

[User Requirements]:
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
    # 记录耗时
    elapsed = time.time() - start_time
    state.setdefault("agent_timings", {})
    state["agent_timings"]["ReqParse"] = state["agent_timings"].get("ReqParse", 0.0) + elapsed
    log(f"ReqParse：解析完成，共 {len(normalized)} 条原子需求，耗时 {elapsed:.2f} 秒")
    return state


def req_explore_node(state: GraphState, llm=None) -> GraphState:
    """ReqExplore（全局版）：一次性处理所有原子需求，输出全局业务需求清单"""
    start_time = time.time()
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
    user_parts.append(f"""[Original Requirements]
{user_input}""")
    
    # 2. 全量原子需求数组
    if atomic_reqs_queue:
        atomic_reqs_text = "\n".join(f"- {req}" for req in atomic_reqs_queue)
        user_parts.append(f"""[Atomic Requirements]
{atomic_reqs_text}""")
    
    # 3. 上一轮需求清单（如有）
    if prev_req_list:
        prev_req_list_json = json.dumps(
            [{"id": req.get("id", ""), "text": req.get("text", "")} for req in prev_req_list],
            ensure_ascii=False
        )
        user_parts.append(f"""[Requirement List from Previous Round]
```json
{prev_req_list_json}
```""")
    
    # 4. 上一轮评分（如有，仅分值）
    if prev_scores:
        scores_arr = [{"id": rid, "score": sc} for rid, sc in prev_scores.items()]
        scores_arr_json = json.dumps(scores_arr, ensure_ascii=False)
        
        # 分析评分分布，提供针对性挖掘指导
        high_score_ids = [rid for rid, sc in prev_scores.items() if sc >= 1]
        zero_score_ids = [rid for rid, sc in prev_scores.items() if sc == 0]
        negative_score_ids = [rid for rid, sc in prev_scores.items() if sc < 0]
        
        guidance_parts = []
        guidance_parts.append("Follow the score-specific rules from the system prompt:")

        if high_score_ids:
            guidance_parts.append(f"- **Requirements scored +2 / +1 (total {len(high_score_ids)} items)**:")
            guidance_parts.append("  * Keep these ids and make only minimal wording adjustments.")
            guidance_parts.append("  * **Systematically check and add the necessary supporting business requirements**:")
            guidance_parts.append("    - Use new ids to cover exception handling (input errors, insufficient permissions, missing data, unmet business rules, etc.).")
            guidance_parts.append("    - Use new ids for permission and role-control requirements.")
            guidance_parts.append("    - Use new ids for data integrity and consistency validation.")
            guidance_parts.append("    - Use new ids for business state transitions and prerequisite checks.")
            guidance_parts.append("    - Ensure each requirement forms a complete business loop (trigger, processing, result, exception, permission).")
            guidance_parts.append("    - Reference the \"Structured Mining Checklist\" from the system prompt and inspect all eight dimensions.")
            guidance_parts.append("    - For +1 items you may add at most 3–5 supporting requirements; be conservative when augmenting +2 items.")

        if zero_score_ids:
            guidance_parts.append(f"- **Requirements scored 0 (total {len(zero_score_ids)} items)**:")
            guidance_parts.append("  * Rewrite each one with the same id to describe a single, clear business behavior.")
            guidance_parts.append("  * **Deeply explore every business variant and supporting requirement**:")
            guidance_parts.append("    - Split into multiple explicit business scenarios, each with a different id.")
            guidance_parts.append("    - Identify variants (normal flow, exception flow, boundary cases) and use different ids for each.")
            guidance_parts.append("    - Systematically mine upstream/downstream scenarios, exception handling, permission control, data validation, and state transitions to cover all supporting requirements.")
            guidance_parts.append("    - Use the \"Structured Mining Checklist\" from the system prompt to ensure no key scenario is missed.")

        if negative_score_ids:
            guidance_parts.append(f"- **Requirements scored -1 / -2 (total {len(negative_score_ids)} items)**:")
            guidance_parts.append("  * Delete these requirements, stop using their ids, and do not recreate the same direction.")

        guidance_parts.append('- Output this round\'s "complete requirement list" containing only the requirements that remain valid plus any new ones created per the above rules.')
        guidance_parts.append("")
        guidance_parts.append("**Important reminders**:")
        guidance_parts.append('- Apply the "Structured Mining Checklist" (Section X) from the system prompt and inspect all eight dimensions.')
        guidance_parts.append("- Prioritize exception handling, boundary conditions, permission control, and data integrity.")
        guidance_parts.append("- Ensure every requirement contains the six essential elements of the business loop (trigger, processing, result, exception, permission, data integrity).")

        user_parts.append(f"""[Previous Round Scores]
```json
{scores_arr_json}
```

{chr(10).join(guidance_parts)}""")
    
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
    
    # 记录耗时
    elapsed = time.time() - start_time
    state.setdefault("agent_timings", {})
    state["agent_timings"]["ReqExplore"] = state["agent_timings"].get("ReqExplore", 0.0) + elapsed
    log(f"ReqExplore：完成全局需求清单生成，共 {len(normalized_reqs)} 条需求，耗时 {elapsed:.2f} 秒")
    return state


def req_clarify_node(state: GraphState, llm=None) -> GraphState:
    """ReqClarify（全局版）：对全局需求清单全量打分"""
    start_time = time.time()
    llm = llm or get_llm_for("ReqClarify")
    
    req_list = state.get("req_list", [])
    global_iteration = state.get("global_iteration", 0)
    
    if not req_list:
        log("ReqClarify：无需求清单，跳过评分")
        state["scores"] = {}
        # 记录耗时（即使跳过也记录）
        elapsed = time.time() - start_time
        state.setdefault("agent_timings", {})
        state["agent_timings"]["ReqClarify"] = state["agent_timings"].get("ReqClarify", 0.0) + elapsed
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
    
    user_template = """[Requirement List]
```json
{req_list_json}
```

[Reference SRS]
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
    
    # 记录耗时
    elapsed = time.time() - start_time
    state.setdefault("agent_timings", {})
    state["agent_timings"]["ReqClarify"] = state["agent_timings"].get("ReqClarify", 0.0) + elapsed
    log(f"ReqClarify：评分完成，耗时 {elapsed:.2f} 秒")
    
    return state


def doc_generate_node(state: GraphState, llm=None) -> GraphState:
    start_time = time.time()
    log("DocGenerate：生成最终 Markdown SRS 文档（流式输出开始）")
    stream_handler = StreamingPrinter()
    llm = llm or get_llm_for("DocGenerate", streaming=True, callbacks=[stream_handler])

    # 使用 effective_req_list，如果没有则使用 req_list
    effective_req_list = state.get("effective_req_list", state.get("req_list", []))
    
    # 预先序列化JSON字符串用于模板填充
    effective_req_list_json = json.dumps(effective_req_list, ensure_ascii=False)

    system_template = """You will transform the user's raw materials such as "requirement lists, user stories, features, and constraints" into a formal Software Requirements Specification (SRS). The document must strictly follow the nine-section structure below:

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

Use formal, unambiguous, testable language. Do **not** use uncertain or vague expressions such as "maybe, probably, try to, similar, TBD, to be determined". Do not ask clarification questions, make assumptions, or speculate. If information is missing, simply omit that subsection.

Every functional requirement must include a unique identifier and title (e.g., FR-001 Login). Each statement must begin with "The system must", "The system shall", or "The system shall not" to keep it testable.

Never mention that any content "comes from the requirement list/input"; only present the finished document.

Language expectations:
• Ban vague wording such as "maybe, try to, basically, appropriate, fast, user-friendly, timely, stable, etc., similar, TBD, to be determined, if possible, should be able to".
• All performance items and metrics must be measurable with explicit units and thresholds.
• Use formal prose; do not use tables or field-style layouts.

When the user supplies a requirement list or description, immediately generate the complete SRS that follows the above structure. Do not ask questions, do not fill in missing data, do not speculate, and do not reference the input source.
"""
    
    user_template = """[Final Requirement List]
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
    # 记录耗时
    elapsed = time.time() - start_time
    state.setdefault("agent_timings", {})
    state["agent_timings"]["DocGenerate"] = state["agent_timings"].get("DocGenerate", 0.0) + elapsed
    log(f"DocGenerate：流式输出完成，耗时 {elapsed:.2f} 秒")
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
        "agent_timings": {},
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
