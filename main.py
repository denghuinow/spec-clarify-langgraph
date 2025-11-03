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
    clarification_memory: Dict[str, str]  # {id: clarification_request}
    pending_clarifications: List[Dict[str, Any]]  # 存储待澄清的需求


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
    """ReqExplore：根据逐条评分（+2~−2）优化，输出"新的完整需求清单"（无 action 字段）"""
    llm = llm or get_llm()
    log(f"ReqExplore：第 {state['iteration']} 轮，根据评分优化需求清单")
    
    # 初始化记忆和待澄清列表
    if 'clarification_memory' not in state:
        state['clarification_memory'] = {}
    if 'pending_clarifications' not in state:
        state['pending_clarifications'] = []
    
    # 将评分 dict 映射为数组，便于展示
    scores_arr = [{"id": rid, "score": sc} for rid, sc in state.get("scores", {}).items()]

    # 如果有评分，根据评分处理澄清
    if scores_arr:
        # 处理澄清结果，更新澄清记忆
        for score_item in scores_arr:
            req_id = score_item["id"]
            score = score_item["score"]
            
            # 如果评分较低（-1 或 -2），可能需要重新处理该需求
            if score <= -1 and req_id in state['clarification_memory']:
                # 从澄清记忆中移除已处理的需求
                del state['clarification_memory'][req_id]
    
    # 提取当前需求清单中的澄清请求并转换为假设性补全
    current_clarifications = []
    for req in state['req_list']:
        content = req['content']
        # 提取(待澄清: ...)内容并转换为假设性补全
        clarification_matches = re.findall(r'\(待澄清: (.*?)\)', content)
        for match in clarification_matches:
            current_clarifications.append({
                "id": req['id'],
                "content": match,
                "original_content": content
            })
    
    # 更新澄清记忆
    for clarification in current_clarifications:
        state['clarification_memory'][clarification['id']] = clarification['content']
        state['pending_clarifications'].append(clarification)
    
    system = (
        '你是"需求挖掘智能体（ReqExplore）"，同时充当资深软件需求分析师兼系统逻辑闭环专家，专门协助开发人员将草稿级功能需求转化为结构完整、逻辑严密、可执行落地的正式文档。\n\n'
        
        '## 核心职责\n'
        '1. 逐条解析原始需求，识别功能点、触发条件、输入输出、状态流转、依赖关系与边界情况\n'
        '2. 在不改变原意前提下，主动补全缺失细节，直接输出完整的需求描述：\n'
        '   - 权限控制规则（谁？在什么条件下？）\n'
        '   - 异常处理路径（失败/超时/冲突/无权限反馈机制）\n'
        '   - 数据一致性机制（同步策略、冲突解决、幂等性）\n'
        '   - 状态机定义（状态变更、回退/撤销机制）\n'
        '   - 用户反馈机制（成功/失败提示、前端交互）\n'
        '3. 确保每条需求形成完整闭环，包含：触发条件 → 执行逻辑 → 结果输出 → 异常兜底\n'
        '4. 覆盖正向流程 + 逆向流程（删除/撤回/驳回/重试）\n'
        '5. 确保数据来源 → 数据处理 → 数据消费 → 数据归档/同步的完整链路\n'
        '6. 遇到模糊、矛盾、缺失、歧义内容，应直接提出明确的完整需求，而非提出问题或建议补全\n\n'
        
        '## 记忆机制\n'
        '1. 你有澄清记忆，记录了之前识别出的待澄清项：\n'
        f'{json.dumps(state["clarification_memory"], ensure_ascii=False, indent=2) if state["clarification_memory"] else "无"}\n'
        '2. 对于澄清记忆中的项目，应基于上下文直接输出明确的完整需求\n'
        '3. 评分含义：\n'
        '   - +2 强采纳：保持原需求并精炼，假设性补全被验证为正确\n'
        '   - +1 采纳：保留并细化，假设性补全基本正确\n'
        '   - 0 中性：保持或泛化，可能需要调整假设性补全\n'
        '   - -1 不采纳：需重写或替换，假设性补全需要调整\n'
        '   - -2 强不采纳：删除或替代，假设性补全需要重新考虑\n\n'
        
        '## 输出规范（直接输出完整需求）\n'
        '1. 格式一致性原则（最高优先级）：\n'
        '   - 最终输出的标题层级、编号体系、段落缩进、列表符号、术语风格必须与原始需求内容100%一致\n'
        '   - 禁止重构文档结构、更改编号或标题层级、添加原始文档未使用的格式\n'
        '2. 输出格式：\n'
        '   - 直接输出完整的、明确的需求描述，不使用任何标记如(建议补全: ...)或(待澄清: ...)\n'
        '   - 将所有必要的细节直接整合到需求内容中\n'
        '3. 绝对禁止使用任何补全标记或问题标记\n\n'
        
        '## 自检清单（输出前必查）\n'
        ' - [ ] 所有操作是否有明确权限控制？\n'
        ' - [ ] 所有外部依赖是否有失败处理？\n'
        ' - [ ] 所有"支持XXX"是否定义输入、输出、异常？\n'
        ' - [ ] 所有自动行为是否定义触发条件、频率、失败策略？\n'
        ' - [ ] 所有列表是否定义默认排序、分页、检索字段？\n'
        ' - [ ] 所有状态变更是否有对应事件或回调？\n'
        ' - [ ] 格式是否与原始需求内容完全一致？\n'
        ' - [ ] 是否避免了使用任何补全或问题标记？\n\n'
        
        '## 最终目标\n'
        '交付一份满足以下标准的功能规格文档：\n'
        ' - 逻辑无漏洞\n'
        ' - 权限无越界\n'
        ' - 数据无断点\n'
        ' - 交互无盲区\n'
        ' - 格式零偏差（与原始内容完全一致）\n'
        ' - 所有需求都是明确、完整、可执行的\n\n'
        
        '请严格按照上述规范，结合评分结果和澄清记忆生成新的完整需求清单。'
        '直接输出完整的、明确的需求描述，不要使用任何标记格式。'
        '严禁输出澄清条目的任何解释性文本或表格，只能返回 ```json\n[{"id": "FR-01", "content": "完整的需求描述"}]\n``` 形式的 JSON 数组。'
        '若删除条目请移除该 id；若新增条目请分配连续的新 id，并保持既有编号稳定。'
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
    
    # 更新需求列表
    state["req_list"] = new_list
    
    # 清理已处理的澄清项（基于评分结果）
    if scores_arr:
        processed_ids = {item["id"] for item in scores_arr}
        state['pending_clarifications'] = [
            item for item in state['pending_clarifications'] 
            if item['id'] not in processed_ids or item['id'] in state['clarification_memory']
        ]
    
    log(f"ReqExplore：生成新的需求清单，共 {len(new_list)} 条")
    return state


def req_clarify_node(state: GraphState, llm=None) -> GraphState:
    """ReqClarify：对每条需求逐项评分，并记录简要理由（不传递给下游）"""
    llm = llm or get_llm()
    log(f"ReqClarify：第 {state['iteration']} 轮，对需求逐条评分")

    system = (
        '你是"需求澄清智能体（ReqClarify）"，模拟提出需求的用户，负责对生成的需求清单进行评分。\n\n'
        
        '## 核心职责\n'
        '1. 以用户视角评估需求清单中的每项需求\n'
        '2. 对比基准SRS文档，判断需求是否符合预期\n'
        '3. 基于业务需要和基准SRS内容对需求进行评分\n'
        '4. 重点关注需求是否满足业务目标和功能要求\n\n'
        
        '## 评分标准\n'
        ' - +2 强采纳：需求完全符合基准SRS要求，超出预期，非常满意\n'
        ' - +1 采纳：需求基本符合基准SRS要求，略有改进，满意\n'
        ' - 0 中性：需求与基准SRS要求基本一致，无明显改进或问题\n'
        ' - -1 不采纳：需求与基准SRS要求有偏差，需要调整\n'
        ' - -2 强不采纳：需求严重偏离基准SRS要求，完全不符合预期\n\n'
        
        '## 评估依据\n'
        ' - [ ] 需求是否符合基准SRS中的功能要求？\n'
        ' - [ ] 需求是否满足业务目标？\n'
        ' - [ ] 需求是否与基准SRS中的非功能要求一致？\n'
        ' - [ ] 需求是否体现了基准SRS中的约束条件？\n'
        ' - [ ] 需求是否与基准SRS中的总体目标相符？\n\n'
        
        '## 输出规范\n'
        '请输出 ```json\n[{"id": "FR-01", "score": 1, "reason": "..."}]\n``` 形式的 JSON 数组，'
        '其中 reason 可省略但需简洁，重点说明与基准SRS的对比结果和评分依据。'
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
        '你是"文档生成智能体（DocGenerate）"，专门将经过逻辑闭环完善的需求清单转化为符合 IEEE 29148-2018 标准的正式SRS文档。\n\n'
        
        '## 核心职责\n'
        '1. 将最终需求清单（包含(建议补全)和(待澄清)标识的完整需求）转化为结构化的SRS文档\n'
        '2. 保持需求的原始编号和ID，确保可追溯性\n'
        '3. 按照IEEE 29148-2018标准组织文档结构，确保逻辑清晰\n'
        '4. 将(建议补全)和(待澄清)内容整合为正式的需求表述\n\n'
        
        '## 文档结构要求\n'
        '必须包含以下章节：\n'
        '1. 引言（目的、范围、定义、参考资料、文档概述）\n'
        '2. 总体描述（产品概述、产品功能、用户特点、约束、假设和依赖）\n'
        '3. 详细需求说明：\n'
        '   3.1 功能需求（FR-XX系列）\n'
        '   3.2 非功能需求（NFR-XX系列）\n'
        '   3.3 约束（CON-XX系列）\n'
        '4. 附录（如需要）\n\n'
        
        '## 输出规范\n'
        '1. 最终需求清单的结构为 ```json\n[{"id": "FR-01", "content": "..."}]\n```，其中 id 前缀（FR/NFR/CON）标识类别\n'
        '2. 根据 id 前缀分组列出需求，沿用原始 id，语言正式、无二义\n'
        '3. 将(建议补全: 内容...)和(待澄清: 问题...)整合为完整、正式的需求描述\n'
        '4. 确保每条需求都具备完整的触发条件→执行逻辑→结果输出→异常兜底的闭环描述\n'
        '5. 只输出 Markdown 文档，格式清晰易读\n\n'
        
        '## 质量要求\n'
        ' - 结构性：严格遵循IEEE 29148-2018标准结构\n'
        ' - 完整性：包含所有经过优化的需求项\n'
        ' - 一致性：保持原始ID和编号体系\n'
        ' - 正式性：语言正式、逻辑清晰、无歧义\n\n'
        
        '请严格按照上述规范生成SRS文档。'
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
    
    # 更新日志记录，避免重复
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
