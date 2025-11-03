# -*- coding: utf-8 -*-
"""
Multi-Agent SRS Generation with LangGraph
-----------------------------------------
四智能体：ReqParse -> ReqExplore <-> ReqClarify -> DocGenerate
- ReqClarify：逐条评分（+2 强采纳，+1 采纳，0 中性，-1 不采纳，-2 强不采纳）
- ReqExplore：仅基于分数优化，输出完整新版本清单（JSON 数组，仅含 id/content）
- DocGenerate：输出 Markdown（IEEE 29148-2018 结构）
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


# -----------------------------
# 节点实现
# -----------------------------
def req_parse_node(state: GraphState, llm=None) -> GraphState:
    """ReqParse：自然语言 -> 初始需求清单（JSON 数组，仅含 id 与 content）"""
    llm = llm or get_llm()
    log("ReqParse：开始解析用户输入")
    current_iteration = state["iteration"] + 1
    system = (
        "你是“需求解析智能体（ReqParse）”。"
        "任务：从用户自然语言中抽取结构化需求清单，"
        "分类为功能需求(FR)、非功能需求(NFR)、约束(CON)。"
        "不要编造缺失信息；输出严格为如下格式：```json\n[{\"id\": \"FR-01\", \"content\": \"...\"}]\n``` "
        "即 JSON 数组，每项仅包含 id 与 content（类别通过 id 前缀体现）。"
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
    """ReqExplore：根据逐条评分（+2~−2）优化，输出“新的完整需求清单”（无 action 字段）"""
    llm = llm or get_llm()
    log(f"ReqExplore：第 {state['iteration']} 轮，根据评分优化需求清单")
    # 将评分 dict 映射为数组，便于展示
    scores_arr = [{"id": rid, "score": sc} for rid, sc in state.get("scores", {}).items()]

    system = (
        "你是“需求挖掘智能体（ReqExplore）”。"
        "任务：根据上一轮需求清单与逐条评分，生成新的完整需求清单。"
        "评分含义：\n"
        "+2 强采纳：完全正确，应保留；\n"
        "+1 采纳：基本正确，优化表述；\n"
        " 0 中性：不确定，可保持或略作泛化；\n"
        "−1 不采纳：存在明显问题，应重新表述；\n"
        "−2 强不采纳：错误或无关，应删除或用替代需求；\n"
        "仅使用分数，不使用任何文字澄清意见。"
        "输出严格为 ```json\n[{\"id\": \"FR-01\", \"content\": \"...\"}]\n``` "
        "形式的 JSON 数组。若删除请移除该条；若新增请分配新 id（连续编号）。保持已有编号稳定。"
    )
    user = (
        "上一轮需求清单（JSON 数组）：\n"
        f"```json\n{json.dumps(state['req_list'], ensure_ascii=False)}\n```\n\n"
        "逐条评分结果（JSON 数组，来自 ReqClarify）：\n"
        f"```json\n{json.dumps(scores_arr, ensure_ascii=False)}\n```\n\n"
        "请输出新的完整需求清单（仅 JSON 数组，使用 ```json``` 包裹，不含任何解释文本）："
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    resp = llm.invoke(messages)
    new_list = extract_first_json(resp.content)
    record_llm_interaction(
        state,
        agent="ReqExplore",
        iteration=state["iteration"],
        messages=messages,
        raw_output=resp.content,
        parsed_output=new_list,
    )
    state["req_list"] = new_list
    log(f"ReqExplore：生成新的需求清单，共 {len(new_list)} 条")
    return state


def req_clarify_node(state: GraphState, llm=None) -> GraphState:
    """ReqClarify：对每条需求逐项评分，并记录简要理由（不传递给下游）"""
    llm = llm or get_llm()
    log(f"ReqClarify：第 {state['iteration']} 轮，对需求逐条评分")

    system = (
        "你是“需求澄清智能体（ReqClarify）”。"
        "根据生成的需求清单与基准 SRS，对每条需求逐项评分。"
        "评分标准：+2 强采纳（完全符合），+1 采纳（基本符合），"
        "0 中性（无法判断或相关性弱），−1 不采纳（部分错误或缺失），−2 强不采纳（严重偏离）。"
        "请输出 ```json\n[{\"id\": \"FR-01\", \"score\": 1, \"reason\": \"...\"}]\n``` "
        "形式的 JSON 数组，其中 reason 可省略但需简洁。"
    )
    user = (
        "需求清单：\n"
        f"```json\n{json.dumps(state['req_list'], ensure_ascii=False)}\n```\n\n"
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

    log("ReqClarify：评分记录完成（已通过代码过滤理由供下游使用）")
    state["iteration"] += 1
    return state


def doc_generate_node(state: GraphState, llm=None) -> GraphState:
    """DocGenerate：将最终需求清单转为 Markdown（IEEE 29148-2018 结构）"""
    log("DocGenerate：生成最终 Markdown SRS 文档（流式输出开始）")
    stream_handler = StreamingPrinter()
    llm = llm or get_llm(temperature=0.1, streaming=True, callbacks=[stream_handler])
    system = (
        "你是“文档生成智能体（DocGenerate）”。"
        "任务：将最终需求清单生成符合 IEEE 29148-2018 的 SRS 文档（Markdown 格式）。"
        "最终需求清单的结构为 ```json\n[{\"id\": \"FR-01\", \"content\": \"...\"}]\n```，其中 id 前缀（FR/NFR/CON）标识类别。"
        "必须包含章节：1 引言（目的/范围），2 总体描述，3 详细需求说明（3.1 功能需求 / 3.2 非功能需求 / 3.3 约束），附录。"
        "要求：根据 id 前缀分组列出需求，沿用原始 id，语言正式、无二义。"
        "只输出 Markdown 文档。"
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
    record_llm_interaction(
        state,
        agent="DocGenerate",
        iteration=state["iteration"],
        messages=messages,
        raw_output=full_text,
    )
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
