# -*- coding: utf-8 -*-
"""
Multi-Agent SRS Generation with LangGraph
-----------------------------------------
四智能体：ReqParse -> ReqExplore <-> ReqClarify -> DocGenerate
- ReqClarify：逐条评分（+2 强采纳，+1 采纳，0 中性，-1 不采纳，-2 强不采纳）
- ReqExplore：仅基于分数优化，输出完整新版本清单（JSON 数组，仅含 id/content）
- DocGenerate：输出 Markdown（IEEE 29148-2018 结构）

本版本新增/修订能力：
1) “最高分需求项冻结暂存”，不再重复评价；
2) 冻结项在 ReqExplore 阶段**不再交由迭代优化**（只读参考，不进入待优化输入）；
3) 对 LLM 输出执行“冻结项强制回填 + 顺序稳定化”，确保冻结项内容与存在性不变。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI  # 若使用其他模型，可替换为相应 LangChain 接口
from langchain_core.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field

# 加载 .env 文件
from dotenv import load_dotenv

load_dotenv()


def read_text_file(file_path: str, label: str) -> str:
    """
    Read UTF-8 text from the given file path, raising a helpful error if missing.
    """
    resolved_path = os.path.expanduser(file_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"{label} 文件不存在: {file_path}")
    log(f"读取 {label} 文件: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as f:
        return f.read()


def log(message: str) -> None:
    """Print log messages immediately for real-time feedback."""
    print(f"[日志] {message}", flush=True)


def record_llm_interaction(
    state: "GraphState",
    *,
    agent: str,
    iteration: int,
    messages: List[Dict[str, str]],
    raw_output: Any,
    parsed_output: Any = None,
) -> None:
    """Append a structured log entry capturing每次模型调用的输入与输出，并立即打印。"""
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

    # 及时打印模型交互内容
    pretty_messages = json.dumps(messages, ensure_ascii=False, indent=2)
    log(f"{agent}（第 {iteration} 轮）输入：\n{pretty_messages}")
    if parsed_output is not None:
        pretty_parsed = json.dumps(parsed_output, ensure_ascii=False, indent=2)
        log(f"{agent}（第 {iteration} 轮）输出（解析后）：\n{pretty_parsed}")
    else:
        log(f"{agent}（第 {iteration} 轮）输出（原始）：\n{raw_text}")


class StreamingPrinter(BaseCallbackHandler):
    """Stream tokens to stdout while buffering the full text."""

    def __init__(self) -> None:
        self._buffer: List[str] = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:  # type: ignore[override]
        print(token, end="", flush=True)
        self._buffer.append(token)

    def get_text(self) -> str:
        return "".join(self._buffer)


# -----------------------------
# LLM 工具（可替换为任意 LangChain 聊天模型）
# -----------------------------
def get_llm(
    model: str = None,
    temperature: float = 0.2,
    streaming: bool = False,
    callbacks: Optional[List[Any]] = None,
):
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
    return ChatOpenAI(**kwargs)


def extract_first_json(text: str) -> Any:
    """
    从模型输出中提取第一个 JSON 数组或对象。
    兼容出现多余文本或代码围栏的情况。
    """
    # 先尝试从 ```json ... ``` 中提取
    fence = re.findall(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S)
    candidates = fence if fence else re.findall(r"(\{.*?\}|\[.*?\])", text, flags=re.S)
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    raise ValueError("未能从模型输出中解析出有效 JSON。原始输出:\n" + text)


# -----------------------------
# 系统状态
# -----------------------------
class GraphState(TypedDict):
    user_input: str
    reference_srs: str
    req_list: List[Dict[str, Any]]
    scores: Dict[str, int]           # {id: score}
    logs: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    srs_output: str
    srs_stream_printed: bool
    clarification_memory: Dict[str, str]  # {id: clarification_request}
    pending_clarifications: List[Dict[str, Any]]  # 存储待澄清的需求

    # ===== 冻结机制 =====
    frozen_ids: List[str]            # 被“最高分”选中的需求 id（不再评分/不再交给 ReqExplore）
    frozen_reqs: Dict[str, str]      # id -> content（冻结版本内容，ReqExplore 不得修改）


# -----------------------------
# 节点实现
# -----------------------------
def req_parse_node(state: GraphState, llm=None) -> GraphState:
    """ReqParse：自然语言 -> 初始需求清单（JSON 数组，仅含 id 与 content）"""
    llm = llm or get_llm()
    log("ReqParse：开始解析用户输入")
    current_iteration = state["iteration"] + 1
    system = (
        '你是"需求解析智能体（ReqParse）"，专门协助开发人员将用户提供的自然语言需求转化为结构化的初始需求清单。\n\n'
        
        '## 核心职责\n'
        '1. 逐段解析用户提供的自然语言需求，识别其中的功能点、输入输出、触发条件、边界情况\n'
        '2. 将识别出的需求分类为：\n'
        '   - 功能需求(FR)：系统应提供的具体功能\n'
        '   - 非功能需求(NFR)：性能、安全、可用性等质量要求\n'
        '   - 约束(CON)：技术、业务、法律等限制条件\n'
        '3. 保持原始需求的术语风格和表达习惯，确保格式与用户原始输入保持一致\n'
        '4. 遇到模糊或不明确的表述时，记录为待澄清项，不自行假设\n\n'
        
        '## 输出规范\n'
        '1. 输出格式严格为：```json\n[{"id": "FR-01", "content": "..."}]\n``` \n'
        '2. 每项仅包含 id 与 content 两个字段，类别通过 id 前缀体现\n'
        '3. content 字段保持原始需求的标题层级、编号、术语风格\n'
        '4. 不要编造缺失信息，保持原始需求的完整性\n\n'
        
        '## 质量要求\n'
        ' - 准确性：忠实地反映用户原始需求\n'
        ' - 完整性：不遗漏重要的功能点或约束\n'
        ' - 一致性：保持原始格式和术语风格\n'
        ' - 清晰性：确保每条需求表述明确无歧义\n\n'
        
        '请严格按照上述规范解析用户需求。'
    )
    user = (
        f"用户需求：\n{state['user_input']}\n\n"
        "请仅输出结构化 JSON 数组，外层用 ```json``` 包裹。"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    resp = llm.invoke(messages)
    parsed = extract_first_json(resp.content)
    record_llm_interaction(
        state,
        agent="ReqParse",
        iteration=current_iteration,
        messages=messages,
        raw_output=resp.content,
        parsed_output=parsed,
    )
    state["req_list"] = parsed
    state["iteration"] = current_iteration
    log(f"ReqParse：解析完成，共 {len(parsed)} 条需求")
    return state


def req_explore_node(state: GraphState, llm=None) -> GraphState:
    """ReqExplore：根据逐条评分（+2~−2）优化，输出新的完整需求清单（不再处理冻结项）"""
    llm = llm or get_llm()
    log(f"ReqExplore：第 {state['iteration']} 轮，根据评分优化需求清单（冻结项不参与）")
    
    # 初始化集合
    state.setdefault('clarification_memory', {})
    state.setdefault('pending_clarifications', [])
    state.setdefault('frozen_ids', [])
    state.setdefault('frozen_reqs', {})

    frozen_ids_set = set(state['frozen_ids'])

    # 仅保留“未冻结”的评分传给提示
    scores_arr_unfrozen = [
        {"id": rid, "score": sc}
        for rid, sc in state.get("scores", {}).items()
        if rid not in frozen_ids_set
    ]

    # “待优化清单”：仅未冻结项
    unfrozen_list = [it for it in state['req_list'] if it['id'] not in frozen_ids_set]

    # 若无未冻结项，直接跳过模型调用
    if not unfrozen_list:
        log("ReqExplore：无未冻结需求，跳过模型调用，需求清单保持不变")
        return state
    
    # 根据评分处理澄清记忆（仅考虑未冻结项）
    for score_item in scores_arr_unfrozen:
        req_id = score_item["id"]
        score = score_item["score"]
        if score <= -1 and req_id in state['clarification_memory']:
            del state['clarification_memory'][req_id]
    
    # 从未冻结条目里提取(待澄清: ...)
    current_clarifications = []
    for req in unfrozen_list:
        content = req['content']
        for match in re.findall(r'\(待澄清: (.*?)\)', content):
            current_clarifications.append({
                "id": req['id'],
                "content": match,
                "original_content": content
            })
    for clarification in current_clarifications:
        state['clarification_memory'][clarification['id']] = clarification['content']
        state['pending_clarifications'].append(clarification)

    # 冻结只读清单（用于上下文一致性，但不进入待优化输入）
    frozen_payload = [
        {"id": fid, "content": state["frozen_reqs"].get(fid, "")}
        for fid in state["frozen_ids"]
    ]

    system = (
        '你是"需求挖掘智能体（ReqExplore）"，负责在保持格式/编号稳定的前提下，对“待优化清单（仅未冻结项）”进行闭环补全与优化。\n\n'
        '## 只读冻结清单\n'
        f'{json.dumps(frozen_payload, ensure_ascii=False, indent=2) if frozen_payload else "无"}\n'
        ' - 冻结清单**只读**，你不得输出或修改这些 id；最终结果会由系统自动合并。\n\n'
        '## 输出范围\n'
        ' - 仅针对“待优化清单”中的 id（未冻结）进行修改/细化；\n'
        ' - 如需新增条目，可新增连续编号的新 id；\n'
        ' - **禁止**输出冻结 id；**禁止**变更任何冻结项内容或编号。\n\n'
        '## 输出规范（直接输出完整需求）\n'
        ' - 保持与输入相同的标题/编号/术语风格；\n'
        ' - 输出闭环需求：触发条件 → 执行逻辑 → 结果输出 → 异常兜底；\n'
        ' - 输出为 ```json\n[{"id": "FR-01", "content": "..."}]\n```；仅包含待优化 id 和新增 id；\n'
        ' - 禁止出现(建议补全: ...)/(待澄清: ...)等标记。\n'
    )
    user = (
        "待优化清单（仅未冻结项，JSON 数组）：\n"
        f"```json\n{json.dumps(unfrozen_list, ensure_ascii=False)}\n```\n\n"
        "逐条评分结果（仅未冻结项）：\n"
        f"```json\n{json.dumps(scores_arr_unfrozen, ensure_ascii=False)}\n```\n\n"
        "请输出新的**仅包含未冻结及新增 id**的完整需求清单（使用 ```json``` 包裹，不含解释文本）："
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    resp = llm.invoke(messages)
    unfrozen_new_list = extract_first_json(resp.content)
    record_llm_interaction(
        state,
        agent="ReqExplore",
        iteration=state["iteration"],
        messages=messages,
        raw_output=resp.content,
        parsed_output=unfrozen_new_list,
    )

    # ===== 合并：冻结项强制回填 + 顺序稳定化 =====
    # 1) 便于查找
    new_map: Dict[str, str] = {str(it["id"]): str(it["content"]) for it in unfrozen_new_list}
    prev_ids_order = [str(it["id"]) for it in state["req_list"]]

    merged: List[Dict[str, str]] = []
    for rid in prev_ids_order:
        if rid in frozen_ids_set:
            # 冻结项：严格使用冻结版本
            content = state["frozen_reqs"].get(rid, next((x["content"] for x in state["req_list"] if x["id"] == rid), ""))
        else:
            # 未冻结项：若 LLM 提供了新版，用新版；否则沿用旧版
            content = new_map.get(rid, next((x["content"] for x in state["req_list"] if x["id"] == rid), ""))
        merged.append({"id": rid, "content": content})

    # 2) 追加 LLM 新增的 id（保持模型扩展）
    for rid, content in new_map.items():
        if rid not in prev_ids_order:
            merged.append({"id": rid, "content": content})

    state["req_list"] = merged

    # 3) 清理 pending_clarifications（仅根据未冻结项评分）
    if scores_arr_unfrozen:
        processed_ids = {item["id"] for item in scores_arr_unfrozen}
        state['pending_clarifications'] = [
            item for item in state['pending_clarifications']
            if item['id'] not in processed_ids or item['id'] in state['clarification_memory']
        ]
    
    log(f"ReqExplore：完成合并，共 {len(state['req_list'])} 条；冻结 {len(state['frozen_ids'])} 条（未参与优化）")
    return state


def req_clarify_node(state: GraphState, llm=None) -> GraphState:
    """ReqClarify：对每条需求逐项评分，并记录简要理由（不传递给下游）
    - 仅对“未冻结”的需求进行评分；
    - 本轮评分结束后，将“最高分”的需求加入冻结集合，后续不再参与评分与优化；
    """
    llm = llm or get_llm()
    log(f"ReqClarify：第 {state['iteration']} 轮，对需求逐条评分（排除冻结项）")

    state.setdefault('frozen_ids', [])
    state.setdefault('frozen_reqs', {})
    frozen_ids_set = set(state['frozen_ids'])

    # 仅选择未冻结条目参与评价
    unfrozen_list = [item for item in state["req_list"] if item["id"] not in frozen_ids_set]
    if not unfrozen_list:
        log("ReqClarify：所有需求均已冻结，本轮跳过评分")
        state["scores"] = {}
        state["iteration"] += 1
        return state

    system = (
        '你是"需求澄清智能体（ReqClarify）"，模拟提出需求的用户，负责对生成的需求清单进行评分。\n\n'
        '评分标准：+2 强采纳，+1 采纳，0 中性，-1 不采纳，-2 强不采纳。\n'
        '输出 ```json\n[{"id": "FR-01", "score": 1, "reason": "..."}]\n```。reason 可省略但需简洁。'
    )
    user = (
        "需评分（未冻结）需求清单：\n"
        f"```json\n{json.dumps(unfrozen_list, ensure_ascii=False)}\n```\n\n"
        "基准 SRS：\n"
        "```text\n"
        f"{state['reference_srs']}\n"
        "```\n\n"
        "请输出评分 JSON 数组（使用 ```json``` 包裹）："
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    resp = llm.invoke(messages)
    evaluations = extract_first_json(resp.content)
    record_llm_interaction(
        state,
        agent="ReqClarify",
        iteration=state["iteration"],
        messages=messages,
        raw_output=resp.content,
        parsed_output=evaluations,
    )

    # 仅保留分数供 ReqExplore 使用
    scores_map: Dict[str, int] = {}
    for item in evaluations:
        rid = str(item.get("id"))
        sc = int(item.get("score"))
        scores_map[rid] = sc
    state["scores"] = scores_map

    # 将“本轮最高分”的需求冻结（加入 frozen_ids / frozen_reqs）
    if scores_map:
        max_score = max(scores_map.values())
        top_ids = [rid for rid, sc in scores_map.items() if sc == max_score]
        req_map = {str(it["id"]): str(it["content"]) for it in state["req_list"]}
        added = 0
        for rid in top_ids:
            if rid not in state["frozen_ids"]:
                state["frozen_ids"].append(rid)
                state["frozen_reqs"][rid] = req_map.get(rid, "")
                added += 1
        log(f"ReqClarify：本轮最高分 = {max_score}，新增冻结 {added} 项；累计冻结 {len(state['frozen_ids'])} 项")
    else:
        log("ReqClarify：本轮无评分结果，未新增冻结项")

    log("ReqClarify：评分记录完成（仅未冻结项）")
    state["iteration"] += 1
    return state


def doc_generate_node(state: GraphState, llm=None) -> GraphState:
    """DocGenerate：将最终需求清单转为 Markdown（IEEE 29148-2018 结构）"""
    log("DocGenerate：生成最终 Markdown SRS 文档（流式输出开始）")
    stream_handler = StreamingPrinter()
    llm = llm or get_llm(temperature=0.1, streaming=True, callbacks=[stream_handler])
    system = (
        '你是"文档生成智能体（DocGenerate）"，将优化后的需求清单转化为符合 IEEE 29148-2018 的 SRS。\n'
        '必须保留原始 id；按 FR/NFR/CON 分组；语言正式、无二义；\n'
        '确保每条需求具备：触发条件→执行逻辑→结果输出→异常兜底。'
    )
    user = (
        "最终需求清单（JSON 数组）：\n"
        f"```json\n{json.dumps(state['req_list'], ensure_ascii=False)}\n```\n\n"
        "请输出 Markdown 版本 SRS："
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    
    # 记录输入日志
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
    if isinstance(raw_content, str):
        full_text = raw_content
    elif isinstance(raw_content, list):
        full_text = "".join(str(part) for part in raw_content)
    else:
        full_text = str(raw_content) if raw_content is not None else ""
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
    """
    达到最大迭代轮数后结束，否则继续。
    """
    if state["iteration"] > state["max_iterations"]:
        log("条件判断：达到最大迭代次数，进入文档生成")
        return "DocGenerate"
    log("条件判断：继续迭代优化需求清单")
    return "ReqExplore"


# -----------------------------
# 构建 LangGraph
# -----------------------------
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("ReqParse", req_parse_node)
    graph.add_node("ReqExplore", req_explore_node)
    graph.add_node("ReqClarify", req_clarify_node)
    graph.add_node("DocGenerate", doc_generate_node)

    graph.set_entry_point("ReqParse")
    graph.add_edge("ReqParse", "ReqExplore")
    graph.add_edge("ReqExplore", "ReqClarify")
    graph.add_conditional_edges("ReqClarify", should_continue, {
        "ReqExplore": "ReqExplore",
        "DocGenerate": "DocGenerate"
    })
    graph.add_edge("DocGenerate", END)
    return graph.compile()


# -----------------------------
# 演示运行
# -----------------------------
class DemoInput(BaseModel):
    user_input: str = Field(..., description="自然语言需求")
    reference_srs: str = Field(..., description="基准SRS（文本）")
    max_iterations: int = 5


def run_demo(demo: DemoInput):
    app = build_graph()
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
        "clarification_memory": {},
        "pending_clarifications": [],
        # 冻结机制初始态
        "frozen_ids": [],
        "frozen_reqs": {},
    }
    final_state = app.invoke(init)
    log("流程结束，输出结果")
    if final_state["srs_stream_printed"]:
        print("\n====== Markdown SRS 输出已通过流式打印展示 ======\n", flush=True)
    else:
        print("\n====== 最终 Markdown SRS 输出 ======\n", flush=True)
        print(final_state["srs_output"], flush=True)


# -----------------------------
# 示例入口
# -----------------------------
DEFAULT_USER_INPUT = (
    "我们需要一个用户管理系统："
    "用户可注册/登录/退出；管理员可查看并按时间导出用户活动记录；"
    "系统需在高峰期保持快速响应；需要访问控制与审计。"
)
DEFAULT_REFERENCE_SRS = (
    "# 参考SRS（摘要）\n"
    "- FR：注册、登录、退出；管理员查看与按时间过滤导出活动；\n"
    "- NFR：响应时间 ≤ 2s，并发 ≥ 1000；\n"
    "- CON：遵从公司安全策略和访问控制规范；"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph 多智能体 SRS 生成演示脚本"
    )
    parser.add_argument(
        "-u",
        "--user-input",
        type=str,
        help="用户需求文本文件路径（user_input）",
    )
    parser.add_argument(
        "-r",
        "--reference-srs",
        type=str,
        help="参考 SRS 文本文件路径（reference_srs）",
    )
    parser.add_argument(
        "-m",
        "--max-iterations",
        type=int,
        default=2,
        help="最大迭代轮数，默认 5",
    )

    args = parser.parse_args()

    if bool(args.user_input) ^ bool(args.reference_srs):
        parser.error("必须同时提供 --user-input 与 --reference-srs。")

    if args.user_input and args.reference_srs:
        user_input_text = read_text_file(args.user_input, "user_input")
        reference_srs_text = read_text_file(args.reference_srs, "reference_srs")
    else:
        user_input_text = DEFAULT_USER_INPUT
        reference_srs_text = DEFAULT_REFERENCE_SRS

    demo = DemoInput(
        user_input=user_input_text,
        reference_srs=reference_srs_text,
        max_iterations=args.max_iterations,
    )
    if not args.user_input:
        log("未提供参数，使用内置示例场景")
    run_demo(demo)


if __name__ == "__main__":
    main()
