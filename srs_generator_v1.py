# -*- coding: utf-8 -*-
"""
Multi-Agent SRS Generation with LangGraph
-----------------------------------------
流程：ReqParse -> ReqExplore <-> ReqClarify -> DocGenerate -> END

- ReqParse：自然语言 -> 初始需求清单（JSON 数组，FR-01/NFR-01/CON-01 格式）
- ReqExplore：仅基于分数优化（只处理未冻结且未移除项），输出新版本清单（JSON 数组，仅含 id/content）
- ReqClarify：逐条评分（排除冻结项），评分范围：+2 强采纳，+1 采纳，0 中性，-1 不采纳，-2 强不采纳
- DocGenerate：输出 Markdown（IEEE Std 830-1998 基本格式）

冻结与移除机制：
- 冻结机制：每轮评分后，最高正向分（>= +1）的所有需求进入冻结列表（frozen_ids），后续不再修改
- 移除机制：得分为 -2（强不采纳）的需求被标记为移除（removed_ids），后续不再处理
- 冻结项和移除项在 ReqExplore 和 ReqClarify 中均被排除，确保已确定的需求不再变动
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict

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
def log(message: str) -> None:
    """Print log messages immediately for real-time feedback."""
    print(f"[日志] {message}", flush=True)


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
      ReqExplore: 0.6 —— 受控挖掘，允许适度外推
      ReqClarify: 0.2 —— 判定与对齐，需一致性
      DocGenerate: 0.1 —— 文档成形，稳定输出
    可通过环境变量覆盖：
      OPENAI_TEMP_REQPARSE / OPENAI_TEMP_REQEXPLORE / OPENAI_TEMP_REQCLARIFY / OPENAI_TEMP_DOCGENERATE
    """
    defaults = {
        "ReqParse": 0.2,
        "ReqExplore": 0.6,
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

    pretty_messages = json.dumps(messages, ensure_ascii=False, indent=2)
    log(f"{agent}（第 {iteration} 轮）输入：\n{pretty_messages}")
    if parsed_output is not None:
        pretty_parsed = json.dumps(parsed_output, ensure_ascii=False, indent=2)
        log(f"{agent}（第 {iteration} 轮）输出（解析后）：\n{pretty_parsed}")
    else:
        log(f"{agent}（第 {iteration} 轮）输出（原始）：\n{raw_text}")


# -----------------------------
# 状态结构
# -----------------------------
class GraphState(TypedDict):
    user_input: str
    reference_srs: str
    req_list: List[Dict[str, Any]]
    scores: Dict[str, int]  # {id: score}
    logs: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    srs_output: str
    srs_stream_printed: bool

    frozen_ids: List[str]
    frozen_reqs: Dict[str, str]

    removed_ids: List[str]

    ablation_mode: Optional[str]  # 消融实验模式：no-clarify 或 no-explore-clarify


# -----------------------------
# 节点实现
# -----------------------------
def req_parse_node(state: GraphState, llm=None) -> GraphState:
    """ReqParse：自然语言 -> 初始需求清单（JSON 数组，仅含 id 与 content）"""
    llm = llm or get_llm_for("ReqParse")
    log("ReqParse：开始解析用户输入")
    current_iteration = state["iteration"] + 1

    user_template = (
        'You are the "Requirement Parsing Agent (ReqParse)", responsible for converting natural language requirements into an **atomic** initial list.\n'
        "\n"
        "[Parsing Method (done internally, do not appear in output)]\n"
        "1) Identify triples: <Actor/Role, Action, Object>, and map to business objects (e.g., user/admin/system/external service).\n"
        '2) Atomic splitting: Each requirement describes only **a single verifiable behavior or constraint**; when encountering conjunctions like "and/also/or/simultaneously", split into multiple items if necessary.\n'
        "3) Deduplication and synonym merging: Merge equivalent expressions, unify terminology and units; avoid repetition.\n"
        "4) Classification (heuristic):\n"
        "   • FR-* (Functional): Describes observable behaviors or interface interactions of input→processing→output.\n"
        "   • NFR-* (Non-functional): Performance/capacity (throughput/concurrency/response latency/peak), reliability/availability, maintainability, portability,\n"
        "     security (authentication/audit/minimum privilege/encryption/compliance), observability (logs/metrics/tracing).\n"
        "   • CON-* (Constraint): Legal/compliance/organizational policies/platform boundaries/external dependencies/data sovereignty/deployment and network constraints, etc., which are **strong constraints** not primarily focused on functional experience.\n"
        "5) Terminology anchoring:\n"
        '   • Preserve proper nouns and units from user input; prohibit inventing new terms; if nouns need to be supplemented, use "Prerequisite: ..." embedded at the end of content.\n'
        '6) Quality threshold: Avoid vague words like "possibly/try to/appropriate/TBD/?"; testable, traceable; prohibit question-style/clarification-style output.\n'
        "\n"
        "[Numbering Convention]\n"
        "- Functional requirements: FR-01, FR-02, FR-03, ... (two-digit increment, starting from 01, consecutive and unique)\n"
        "- Non-functional requirements: NFR-01, NFR-02, ... (two-digit increment)\n"
        "- Constraints: CON-01, CON-02, ... (two-digit increment)\n"
        "Numbers must not skip or repeat; the three sequences are independent.\n"
        "\n"
        "[Output Format (JSON array only, elements contain only id and content)]\n"
        "```json\n"
        '[{{"id":"FR-01","content":"……"}}]\n'
        "```\n"
        "No fields other than id and content may appear; no explanations, tables, or comments may be output.\n"
        "\n"
        "---\n"
        "\n"
        "User Requirements:\n"
        "{user_input}\n"
        "\n"
        "Please output only a structured JSON array (outer layer uses ```json fence, array elements contain only id and content)."
    )

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
    # 轻量合法化：保证每条都只有 id/content 且为字符串
    normalized: List[Dict[str, str]] = []
    for it in parsed:
        rid = str(it.get("id"))
        content = str(it.get("content", "")).strip()
        if not rid or not content:
            continue
        normalized.append({"id": rid, "content": content})
    state["req_list"] = normalized
    state["iteration"] = current_iteration
    log(f"ReqParse：解析完成，共 {len(normalized)} 条需求")
    return state


def req_explore_node(state: GraphState, llm=None) -> GraphState:
    llm = llm or get_llm_for("ReqExplore")
    log(f"ReqExplore：第 {state['iteration']} 轮，根据评分强化挖掘与优化（仅处理未冻结且未移除项）")
    state.setdefault("frozen_ids", [])
    state.setdefault("frozen_reqs", {})
    state.setdefault("removed_ids", [])
    frozen_ids_set = set(state["frozen_ids"])
    removed_ids_set = set(state["removed_ids"])

    # 仅“未冻结 & 未移除”参与本轮挖掘
    unfrozen_list = [
        it for it in state["req_list"]
        if it["id"] not in frozen_ids_set and it["id"] not in removed_ids_set
    ]

    scores_arr_unfrozen = [
        {"id": rid, "score": sc}
        for rid, sc in state.get("scores", {}).items()
        if rid not in frozen_ids_set and rid not in removed_ids_set
    ]

    if not unfrozen_list:
        log("ReqExplore：无未冻结且未移除的需求，跳过模型调用，需求清单保持不变")
        return state

    frozen_payload = [
        {"id": fid, "content": state["frozen_reqs"].get(fid, "")}
        for fid in state["frozen_ids"]
    ]
    removed_payload = [{"id": rid} for rid in state["removed_ids"]]

    frozen_payload_json = json.dumps(frozen_payload, ensure_ascii=False, indent=2) if frozen_payload else "无"
    removed_payload_json = json.dumps(removed_payload, ensure_ascii=False, indent=2) if removed_payload else "无"
    unfrozen_list_json = json.dumps(unfrozen_list, ensure_ascii=False)
    scores_arr_unfrozen_json = json.dumps(scores_arr_unfrozen, ensure_ascii=False)

    user_template = (
        'You are the "Requirement Exploration Agent (ReqExplore)", a senior requirements engineer and closed-loop design expert.\n'
        "You must perform **aggressive but controlled expansion**, only on **unfrozen and non-removed** items.\n"
        "Frozen items and removed items are strictly read-only and must NOT appear in your output.\n"
        "\n"
        "[Your Goals]\n"
        "1) For each unfrozen requirement, use its score and content to:\n"
        "   - Improve clarity, testability, and structure.\n"
        "   - Systematically derive ALL reasonably implied supporting requirements\n"
        "     (functional, non-functional, constraints), as long as they:\n"
        "       • Stay within the SAME business/domain scope;\n"
        "       • Are strongly grounded in existing items or Reference SRS patterns;\n"
        "       • Make the end-to-end behavior verifiable and operable.\n"
        "2) Compared to a conservative mode, you are encouraged to generate MORE items\n"
        "   when they are logically necessary, typical, or high-confidence implications.\n"
        "\n"
        "[Score-Based Exploration Strategy]\n"
        "+2 (Strong Adoption):\n"
        "   - Keep semantics strictly; refine wording/structure.\n"
        "   - MAY add missing **fine-grained checks** tightly coupled to this item\n"
        "     (e.g., explicit error handling, logging, audit, boundary cases)\n"
        "     when obviously implied; avoid changing its scope.\n"
        "+1 (Adoption):\n"
        "   - Keep the core intent.\n"
        "   - AGGRESSIVELY enumerate supporting requirements needed to test/operate it:\n"
        "       • detailed input/output rules;\n"
        "       • role/permission & audit;\n"
        "       • exception handling & rollback;\n"
        "       • data integrity & idempotency;\n"
        "       • observability (logs/metrics/traces/alerts).\n"
        "   - You MAY generate multiple new items around this requirement to cover the loop.\n"
        "0 (Neutral):\n"
        "   - Treat as a seed: clarify assumptions via verifiable expressions.\n"
        "   - Split vague or overloaded text into multiple precise items.\n"
        "   - Derive variants/scenarios (normal / abnormal / edge cases) when reasonable.\n"
        "-1 (Non-adoption):\n"
        "   - Rewrite or replace into a set of precise items that match scope & constraints.\n"
        "   - If one vague item actually implies several concrete responsibilities,\n"
        "     expand it into multiple new requirements.\n"
        "-2 (Strong Non-adoption):\n"
        "   - This id is handled outside you: DO NOT output or resurrect it,\n"
        "     and DO NOT produce near-synonym replacements for it.\n"
        "\n"
        "[Aggressive Exploration Boundaries]\n"
        "- You MAY create many new FR/NFR/CON/SUG items when:\n"
        "   • They are direct upstream/downstream/operational/support requirements of existing items; or\n"
        "   • They reflect standard, high-confidence practices (security, audit, monitoring, reliability)\n"
        "     required to make current requirements pass real-world acceptance.\n"
        "- You MUST NOT:\n"
        "   • Introduce a completely new business domain/module unrelated to current list;\n"
        "   • Invent new roles or external systems out of thin air;\n"
        "   • Reintroduce any id from Removed List, or its near-synonyms.\n"
        "\n"
        "[Numbering Rules]\n"
        "- Only output **unfrozen + new** items.\n"
        "- Do NOT output frozen ids and removed ids.\n"
        "- For existing unfrozen ids: keep their ids, update contents as needed.\n"
        "- For new items:\n"
        "   • Use id prefixes FR-/NFR-/CON-/SUG- consistent with their nature;\n"
        "   • Ensure no conflict with any existing id;\n"
        "   • Continue sequences at the tail (you infer next numbers from context).\n"
        "\n"
        "[Closed-Loop Checklist for Each Area]\n"
        "- Trigger & input conditions are explicit.\n"
        "- Processing steps and decision rules are clear.\n"
        "- Output and acceptance criteria are testable.\n"
        "- Error/exception handling & rollback paths exist.\n"
        "- Access control & audit trail are covered.\n"
        "- Data consistency & idempotency are addressed when relevant.\n"
        "- Observability: logs/metrics/traces/alerts for critical flows.\n"
        "- Prerequisites are explicit via \"Prerequisite: ...\" in content if needed.\n"
        "- Remove vague words (fast/many/robust/etc.) by adding thresholds.\n"
        "\n"
        "[Hard Output Constraints]\n"
        "- You MUST return ONLY one JSON array wrapped by ```json ...```.\n"
        "- Each element: {{\"id\":\"...\",\"content\":\"...\"}}.\n"
        "- No extra keys, no comments, no natural-language explanation.\n"
        "- Do NOT include any frozen id or removed id.\n"
        "\n"
        "[(Read-only) Frozen List]:\n"
        "{frozen_payload_json}\n"
        "\n"
        "[(Read-only) Removed List (MUST NOT be mentioned/reintroduced)]:\n"
        "{removed_payload_json}\n"
        "\n"
        "---\n"
        "\n"
        "Unfrozen & Non-removed List (JSON):\n"
        "```json\n"
        "{unfrozen_list_json}\n"
        "```\n"
        "\n"
        "Scores (for unfrozen & non-removed items, JSON):\n"
        "```json\n"
        "{scores_arr_unfrozen_json}\n"
        "```\n"
        "\n"
        "Now output the optimized + aggressively expanded list as a single JSON array (```json fenced)."
    )

    user = user_template.format(
        frozen_payload_json=frozen_payload_json,
        removed_payload_json=removed_payload_json,
        unfrozen_list_json=unfrozen_list_json,
        scores_arr_unfrozen_json=scores_arr_unfrozen_json,
    )

    messages = [{"role": "user", "content": user}]
    unfrozen_new_list, raw_output = invoke_with_json_retry(llm, messages, max_retries=3)
    record_llm_interaction(
        state,
        agent="ReqExplore",
        iteration=state["iteration"],
        messages=messages,
        raw_output=raw_output,
        parsed_output=unfrozen_new_list,
    )

    # 合并逻辑保持：冻结不动，移除不回流，新旧 unfrozen 替换 + 追加真正新增项
    new_map: Dict[str, str] = {}
    for it in unfrozen_new_list:
        rid = str(it.get("id"))
        if not rid or rid in frozen_ids_set or rid in removed_ids_set:
            continue  # 模型越界输出，丢弃
        content = str(it.get("content", "")).strip()
        if not content:
            continue
        new_map[rid] = content  # 重复 id 以最后一次为准

    prev_ids_order = [str(it["id"]) for it in state["req_list"]]

    merged: List[Dict[str, str]] = []
    for rid in prev_ids_order:
        if rid in removed_ids_set:
            continue
        if rid in frozen_ids_set:
            # 冻结项内容以 frozen_reqs 为准，兜底用旧值
            content = state["frozen_reqs"].get(
                rid,
                next((x["content"] for x in state["req_list"] if x["id"] == rid), "")
            )
            if content:
                merged.append({"id": rid, "content": content})
        else:
            # 未冻结：若模型给出新内容则替换，否则沿用旧值
            content = new_map.get(
                rid,
                next((x["content"] for x in state["req_list"] if x["id"] == rid), "")
            )
            if content and rid not in removed_ids_set:
                merged.append({"id": rid, "content": content})

    # 附加真正新增的条目（不在原列表且未被标记移除）
    for rid, content in new_map.items():
        if rid not in prev_ids_order and rid not in removed_ids_set:
            merged.append({"id": rid, "content": content})

    state["req_list"] = merged
    log(
        f"ReqExplore：强化挖掘完成，共 {len(state['req_list'])} 条；"
        f"冻结 {len(state['frozen_ids'])} 条；已移除 {len(state['removed_ids'])} 条"
    )
    return state


def req_clarify_node(state: GraphState, llm=None) -> GraphState:
    """ReqClarify：对未冻结需求逐条评分（排除冻结项）
    
    评分规则：
    - +2：强采纳，进入冻结列表
    - +1：采纳，进入冻结列表
    - 0：中性，需要重写澄清
    - -1：不采纳，需要重写或替换
    - -2：强不采纳，标记为移除
    
    冻结策略：每轮评分后，最高正向分（>= +1）的所有需求进入冻结列表。
    """
    llm = llm or get_llm_for("ReqClarify")
    log(f"ReqClarify：第 {state['iteration']} 轮，对需求逐条评分（排除冻结项）")
    state.setdefault("frozen_ids", [])
    state.setdefault("frozen_reqs", {})
    state.setdefault("removed_ids", [])
    frozen_ids_set = set(state["frozen_ids"])

    unfrozen_list = [
        item for item in state["req_list"] if item["id"] not in frozen_ids_set
    ]
    if not unfrozen_list:
        log("ReqClarify：所有需求均已冻结，本轮跳过评分")
        state["scores"] = {}
        state["iteration"] += 1
        return state

    # 预先序列化JSON字符串用于模板填充
    unfrozen_list_json = json.dumps(unfrozen_list, ensure_ascii=False)

    user_template = (
        'You are the "Requirement Clarification Agent (ReqClarify)", scoring "unfrozen" requirements item by item from the **requester/acceptor** perspective;'
        'using "Reference SRS" as the sole standard, anchoring terminology/units/thresholds/role names, prohibiting introducing new requirements or rewriting the original text.\n'
        "\n"
        "[Scoring Criteria]\n"
        "+2 (Strong Adoption): Consistent with or better than reference SRS; semantically complete, testable, clear boundaries, no ambiguous words.\n"
        "+1 (Adoption): Basically consistent, only missing a few verifiable details (exceptions/permissions/observability, etc.).\n"
        "0 (Neutral): Insufficient information or too many uncertain assumptions, need to condition or supplement verification criteria before re-evaluation.\n"
        "-1 (Non-adoption): Obvious gaps or deviations in role/scope/trigger/result/failure strategy/consistency, etc., should be rewritten or replaced.\n"
        "-2 (Strong Non-adoption): Severely conflicts with goals/constraints/compliance or cross-domain; **should be directly removed, and prohibited from re-mentioning or near-synonym rewriting in subsequent iterations**.\n"
        "\n"
        "[Consistency Constraints]\n"
        "• Must score each unfrozen requirement in the input item by item; order and length must match the input; ids must correspond one-to-one.\n"
        '• Output only one array wrapped in ```json fence: [{{"id":"FR-01","score":1,"reason":"..."}}].\n'
        "• reason should be concise and auditable (≤50 characters).\n"
        "\n"
        "---\n"
        "\n"
        "Requirements to Score (unfrozen) List:\n"
        "```json\n"
        "{unfrozen_list_json}\n"
        "```\n"
        "\n"
        "Reference SRS:\n"
        "```text\n"
        "{reference_srs}\n"
        "```\n"
        "\n"
        "Please output scoring JSON array (wrapped with ```json```):"
    )

    user = user_template.format(
        unfrozen_list_json=unfrozen_list_json, reference_srs=state["reference_srs"]
    )

    messages = [{"role": "user", "content": user}]
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

    # -2 强不采纳 -> 标记移除
    to_remove_now = [rid for rid, sc in scores_map.items() if sc == -2]
    if to_remove_now:
        known = set(state["removed_ids"])
        add_cnt = 0
        for rid in to_remove_now:
            if rid not in known:
                state["removed_ids"].append(rid)
                add_cnt += 1
        log(
            f"ReqClarify：标记移除 {add_cnt} 项（-2），累计移除 {len(state['removed_ids'])} 项"
        )

    # 冻结策略：取本轮最高正向分（>=1）的所有 id 进入冻结
    if scores_map:
        max_score = max(scores_map.values())
        if max_score >= 1:
            top_ids = [rid for rid, sc in scores_map.items() if sc == max_score]
            req_map = {str(it["id"]): str(it["content"]) for it in state["req_list"]}
            added = 0
            for rid in top_ids:
                if rid not in state["frozen_ids"]:
                    state["frozen_ids"].append(rid)
                    state["frozen_reqs"][rid] = req_map.get(rid, "")
                    added += 1
            log(
                f"ReqClarify：本轮最高正向分 = {max_score}，新增冻结 {added} 项；累计冻结 {len(state['frozen_ids'])} 项"
            )
        else:
            log("ReqClarify：本轮最高分 < 1（无正向高分），不新增冻结项")
    else:
        log("ReqClarify：本轮无评分结果，未新增冻结项")

    log("ReqClarify：评分记录完成（仅未冻结项）")
    state["iteration"] += 1
    return state


def doc_generate_node(state: GraphState, llm=None) -> GraphState:
    log("DocGenerate：生成最终 Markdown SRS 文档（流式输出开始）")
    stream_handler = StreamingPrinter()
    llm = llm or get_llm_for("DocGenerate", streaming=True, callbacks=[stream_handler])

    effective_req_list = [
        it
        for it in state["req_list"]
        if it["id"] not in set(state.get("removed_ids", []))
    ]

    # 预先序列化JSON字符串用于模板填充
    effective_req_list_json = json.dumps(effective_req_list, ensure_ascii=False)

    user_template = (
        'You are the "Document Generation Agent (DocGenerate)". Please convert the optimized requirement list into'
        "a Software Requirements Specification (SRS) following IEEE Std 830-1998, with Markdown as the output medium.\n"
        '• Language should be formal, unambiguous, and testable; avoid uncertain expressions like "possibly/probably/TBD".\n'
        "Final Requirement List (JSON array):\n"
        "```json\n"
        "{effective_req_list_json}\n"
        "```\n"
        "\n"
        "Please output a Markdown version SRS following the basic format of IEEE Std 830-1998:"
    )

    user = user_template.format(effective_req_list_json=effective_req_list_json)

    messages = [{"role": "user", "content": user}]

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
# 条件路由：是否继续迭代
# -----------------------------
def should_continue(state: GraphState) -> str:
    """达到最大迭代轮数后结束，否则继续。"""
    # no-clarify 模式：ReqExplore 只执行一次，直接进入 DocGenerate
    if state.get("ablation_mode") == "no-clarify":
        log("条件判断：no-clarify 模式，ReqExplore 执行一次后直接进入文档生成")
        return "DocGenerate"
    if state["iteration"] > state["max_iterations"]:
        log("条件判断：达到最大迭代次数，进入文档生成")
        return "DocGenerate"
    log("条件判断：继续迭代优化需求清单")
    return "ReqExplore"


def build_graph(ablation_mode: Optional[str] = None):
    graph = StateGraph(GraphState)
    graph.add_node("ReqParse", req_parse_node)
    graph.add_node("DocGenerate", doc_generate_node)

    graph.set_entry_point("ReqParse")

    if ablation_mode == "no-explore-clarify":
        # 模式：移除 ReqExplore + ReqClarify
        # 流程：ReqParse -> DocGenerate -> END
        log("构建图结构：no-explore-clarify 模式（跳过 ReqExplore 和 ReqClarify）")
        graph.add_edge("ReqParse", "DocGenerate")
    elif ablation_mode == "no-clarify":
        # 模式：移除 ReqClarify
        # 流程：ReqParse -> ReqExplore -> DocGenerate -> END
        log("构建图结构：no-clarify 模式（跳过 ReqClarify，ReqExplore 执行一次）")
        graph.add_node("ReqExplore", req_explore_node)
        graph.add_edge("ReqParse", "ReqExplore")
        graph.add_conditional_edges(
            "ReqExplore", should_continue, {"DocGenerate": "DocGenerate"}
        )
    else:
        # 默认模式：完整流程
        # 流程：ReqParse -> ReqExplore <-> ReqClarify -> DocGenerate -> END
        log("构建图结构：默认模式（完整流程）")
        graph.add_node("ReqExplore", req_explore_node)
        graph.add_node("ReqClarify", req_clarify_node)
        graph.add_edge("ReqParse", "ReqExplore")
        graph.add_edge("ReqExplore", "ReqClarify")
        graph.add_conditional_edges(
            "ReqClarify",
            should_continue,
            {"ReqExplore": "ReqExplore", "DocGenerate": "DocGenerate"},
        )

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
        "req_list": [],
        "scores": {},
        "logs": [],
        "iteration": 0,
        "max_iterations": demo.max_iterations,
        "srs_output": "",
        "srs_stream_printed": False,
        "frozen_ids": [],
        "frozen_reqs": {},
        "removed_ids": [],
        "ablation_mode": demo.ablation_mode,
    }
    final_state = app.invoke(init)
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
