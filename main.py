# -*- coding: utf-8 -*-
"""
Multi-Agent SRS Generation with LangGraph
-----------------------------------------
流程：ReqParse -> ReqExplore <-> ReqClarify -> DocGenerate -> SimEvaluate -> END

- ReqClarify：逐条评分（+2 强采纳，+1 采纳，0 中性，-1 不采纳，-2 强不采纳）
- ReqExplore：仅基于分数优化（只处理未冻结项），输出新版本清单（JSON 数组，仅含 id/content）
- DocGenerate：输出 Markdown（IEEE Std 830-1998 基本格式）
- SimEvaluate：文本嵌入（不分片）评估两类相似度并打印：
    1) 生成 SRS（srs_output） vs 基准 SRS（reference_srs）
    2) 用户输入（user_input） vs 基准 SRS（reference_srs）
   同时打印三侧向量维度（gen/ref/user），维度不一致则跳过对应相似度计算。

新增/修订要点：
1) 评分语义拆分：explore_action（给 ReqExplore）与 clarify_action（给 ReqClarify）。
2) 冻结：每轮最高分条目加入冻结；冻结项不再交给 ReqExplore，合并时强制回填。
3) SimEvaluate：除原有 gen vs ref 相似度外，新增 user vs ref 相似度；打印向量维度。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field

# 加载 .env 文件
from dotenv import load_dotenv

load_dotenv()

# =========================
# 统一评分等级定义（集中配置）
# 拆分 explore_action / clarify_action
# =========================
SCORE_POLICY: Dict[int, Dict[str, str]] = {
    2: {
        "label": "强采纳",
        "explore_action": "保留并微调表达；确认隐含设定为已达成；仅做措辞精炼与条理化，不改变语义。",
        "clarify_action": "判定与基准 SRS 完全一致或更优；给出高度肯定的简要理由；标注为可直接纳入与优先冻结的候选。"
    },
    1: {
        "label": "采纳",
        "explore_action": "保留并细化；补充输入/输出、权限、异常、数据一致性、状态机与用户反馈等缺失细节。",
        "clarify_action": "基本符合基准 SRS；指出少量需补充的点（边界/异常/可观测性），鼓励完善后纳入。"
    },
    0: {
        "label": "中性",
        "explore_action": "保持或泛化；去除不确定假设，改为条件化表述；提升可检验性与边界定义。",
        "clarify_action": "与基准 SRS 等效或信息不足；提示风险点与验证口径，建议保持或泛化后再评估。"
    },
    -1: {
        "label": "不采纳",
        "explore_action": "重写或替换；对齐业务目标和用户期望；明确触发条件、失败策略与回退路径。",
        "clarify_action": "与基准 SRS 有偏差；指出具体缺口并建议重写方向或替代方案。"
    },
    -2: {
        "label": "强不采纳",
        "explore_action": "删除或用新条目替代；如保留原 id 则重新定义其 content 以满足业务目标和用户期望。",
        "clarify_action": "严重偏离或违背约束；明确拒绝并建议删除或以新条目完全替代。"
    },
}

def score_policy_text(role: str) -> str:
    """
    role ∈ {'explore', 'clarify'}，返回对应角色可读的评分语义说明文本。
    """
    assert role in {"explore", "clarify"}
    header = "评分等级与含义（统一定义，面向 {}）：".format("ReqExplore" if role == "explore" else "ReqClarify")
    lines = [header]
    for k in sorted(SCORE_POLICY.keys(), reverse=True):
        v = SCORE_POLICY[k]
        action = v["explore_action"] if role == "explore" else v["clarify_action"]
        lines.append(f"{k}（{v['label']}）：{action}")
    return "\n".join(lines)


def read_text_file(file_path: str, label: str) -> str:
    """Read UTF-8 text from the given file path, raising a helpful error if missing."""
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
    """记录每次模型调用的输入与输出，并立即打印。"""
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
# LLM / Embeddings
# -----------------------------
def get_llm(
    model: str = None,
    temperature: float = 0.2,
    streaming: bool = False,
    callbacks: Optional[List[Any]] = None,
):
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs: Dict[str, Any] = {"model": model, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    if streaming:
        kwargs["streaming"] = True
    if callbacks:
        kwargs["callbacks"] = callbacks
    return ChatOpenAI(**kwargs)


def get_embeddings_model(model: str = None):
    """默认 text-embedding-3-large；兼容 OPENAI_BASE_URL。"""
    model = model or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs: Dict[str, Any] = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)


def extract_first_json(text: str) -> Any:
    """从模型输出中提取第一个 JSON 数组或对象（容错围栏/噪声）。"""
    fence = re.findall(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S)
    candidates = fence if fence else re.findall(r"(\{.*?\}|\[.*?\])", text, flags=re.S)
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    raise ValueError("未能从模型输出中解析出有效 JSON。原始输出:\n" + text)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度。"""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


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

    # 冻结机制
    frozen_ids: List[str]
    frozen_reqs: Dict[str, str]

    # 文档相似度与维度（gen vs ref）
    embedding_similarity: Optional[float]
    embedding_distance: Optional[float]
    embedding_dim_gen: Optional[int]
    embedding_dim_ref: Optional[int]

    # 用户输入 vs 基准 SRS 的相似度与维度
    embedding_similarity_user_ref: Optional[float]
    embedding_distance_user_ref: Optional[float]
    embedding_dim_user: Optional[int]


# -----------------------------
# 节点实现
# -----------------------------
def req_parse_node(state: GraphState, llm=None) -> GraphState:
    """ReqParse：自然语言 -> 初始需求清单（JSON 数组，仅含 id 与 content）"""
    llm = llm or get_llm()
    log("ReqParse：开始解析用户输入")
    current_iteration = state["iteration"] + 1
    system = (
        '你是"需求解析智能体（ReqParse）"，负责将自然语言需求转为结构化初始清单。\n\n'
        '【需求编号规范】\n'
        '- 功能需求（Functional Requirements）：FR-01, FR-02, FR-03, ...（从01开始，连续递增，使用两位数字）\n'
        '- 非功能需求（Non-Functional Requirements）：NFR-01, NFR-02, NFR-03, ...（从01开始，连续递增）\n'
        '- 约束（Constraints）：CON-01, CON-02, CON-03, ...（从01开始，连续递增）\n'
        '- 编号格式：前缀（FR/NFR/CON）- 两位数字（01-99），编号必须连续且唯一\n\n'
        '输出规范：```json\n[{"id": "FR-01", "content": "..."}]\n```（仅 id 与 content）。\n'
        '不得引入占位或问题标记；保持原术语与风格一致；严格按照上述编号规范分配 id。'
    )
    user = f"用户需求：\n{state['user_input']}\n\n请仅输出结构化 JSON 数组，外层用 ```json``` 包裹。"
    messages = [{"role": "system", "content": system},{"role": "user", "content": user}]
    resp = llm.invoke(messages)
    parsed = extract_first_json(resp.content)
    record_llm_interaction(state, agent="ReqParse", iteration=current_iteration,
                           messages=messages, raw_output=resp.content, parsed_output=parsed)
    state["req_list"] = parsed
    state["iteration"] = current_iteration
    log(f"ReqParse：解析完成，共 {len(parsed)} 条需求")
    return state


def req_explore_node(state: GraphState, llm=None) -> GraphState:
    """ReqExplore：根据评分优化未冻结条目，输出新的完整需求清单（不处理冻结项）"""
    llm = llm or get_llm()
    log(f"ReqExplore：第 {state['iteration']} 轮，根据评分优化需求清单（冻结项不参与）")
    state.setdefault('frozen_ids', [])
    state.setdefault('frozen_reqs', {})
    frozen_ids_set = set(state['frozen_ids'])

    scores_arr_unfrozen = [{"id": rid, "score": sc}
                           for rid, sc in state.get("scores", {}).items()
                           if rid not in frozen_ids_set]
    unfrozen_list = [it for it in state['req_list'] if it['id'] not in frozen_ids_set]
    if not unfrozen_list:
        log("ReqExplore：无未冻结需求，跳过模型调用，需求清单保持不变")
        return state

    policy_text_explore = score_policy_text("explore")
    frozen_payload = [{"id": fid, "content": state["frozen_reqs"].get(fid, "")}
                      for fid in state["frozen_ids"]]

    system = (
        '你是"需求挖掘智能体（ReqExplore）"，仅对未冻结 id 做闭环补全与优化。\n\n'
        f"{policy_text_explore}\n\n"
        "【需求编号规范（必须严格遵守）】\n"
        "- 功能需求：FR-01, FR-02, FR-03, ...（两位数字，从01开始）\n"
        "- 非功能需求：NFR-01, NFR-02, NFR-03, ...（两位数字，从01开始）\n"
        "- 约束：CON-01, CON-02, CON-03, ...（两位数字，从01开始）\n"
        "- 新增 id 必须遵循上述格式，且与现有编号保持连续（不得重复或跳跃）\n"
        "- 若删除原 id，新增 id 应填补空缺；若替换，可保留原 id 或使用新连续编号\n\n"
        "按评分采取动作（保留并微调/细化、保持或泛化、重写或替代、删除）。可新增连续编号的新 id 承载替代稿。\n"
        "只读冻结清单如下（严禁输出或修改这些 id；系统稍后自动合并）：\n"
        f"{json.dumps(frozen_payload, ensure_ascii=False, indent=2) if frozen_payload else '无'}\n\n"
        "输出为 ```json\n[{\"id\": \"FR-01\", \"content\": \"...\"}]\n```，仅包含未冻结及新增 id；"
        "不得出现占位或问题标记；保持编号/风格一致；每条包含触发→逻辑→输出→异常兜底。"
    )
    user = (
        "待优化清单（仅未冻结项）：\n"
        f"```json\n{json.dumps(unfrozen_list, ensure_ascii=False)}\n```\n\n"
        "逐条评分（仅未冻结项）：\n"
        f"```json\n{json.dumps(scores_arr_unfrozen, ensure_ascii=False)}\n```\n\n"
        "请输出新的仅含未冻结及新增 id 的完整需求清单（包裹为 ```json```）："
    )
    messages = [{"role": "system", "content": system},{"role": "user", "content": user}]
    resp = llm.invoke(messages)
    unfrozen_new_list = extract_first_json(resp.content)
    record_llm_interaction(state, agent="ReqExplore", iteration=state["iteration"],
                           messages=messages, raw_output=resp.content, parsed_output=unfrozen_new_list)

    # 合并：冻结项强制回填 + 顺序稳定化
    new_map: Dict[str, str] = {str(it["id"]): str(it["content"]) for it in unfrozen_new_list}
    prev_ids_order = [str(it["id"]) for it in state["req_list"]]

    merged: List[Dict[str, str]] = []
    for rid in prev_ids_order:
        if rid in frozen_ids_set:
            content = state["frozen_reqs"].get(rid,
                      next((x["content"] for x in state["req_list"] if x["id"] == rid), ""))
        else:
            content = new_map.get(rid,
                      next((x["content"] for x in state["req_list"] if x["id"] == rid), ""))
        merged.append({"id": rid, "content": content})

    for rid, content in new_map.items():
        if rid not in prev_ids_order:
            merged.append({"id": rid, "content": content})

    state["req_list"] = merged
    log(f"ReqExplore：完成合并，共 {len(state['req_list'])} 条；冻结 {len(state['frozen_ids'])} 条（未参与优化）")
    return state


def req_clarify_node(state: GraphState, llm=None) -> GraphState:
    """ReqClarify：对每条需求逐项评分（仅未冻结），并冻结本轮最高分"""
    llm = llm or get_llm()
    log(f"ReqClarify：第 {state['iteration']} 轮，对需求逐条评分（排除冻结项）")
    state.setdefault('frozen_ids', [])
    state.setdefault('frozen_reqs', {})
    frozen_ids_set = set(state['frozen_ids'])

    unfrozen_list = [item for item in state["req_list"] if item["id"] not in frozen_ids_set]
    if not unfrozen_list:
        log("ReqClarify：所有需求均已冻结，本轮跳过评分")
        state["scores"] = {}
        state["iteration"] += 1
        return state

    policy_text_clarify = score_policy_text("clarify")
    system = (
        '你是"需求澄清智能体（ReqClarify）"，以需求方立场对需求清单逐项评分；'
        '基准 SRS 是需求方诉求与底线，评分以其为依据。\n\n'
        f"{policy_text_clarify}\n\n"
        "评分标准：+2、+1、0、-1、-2；输出 ```json\n[{\"id\":\"FR-01\",\"score\":1,\"reason\":\"...\"}]\n```。"
        "reason 可省略但应基于基准 SRS。"
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
    messages = [{"role": "system", "content": system},{"role": "user", "content": user}]
    resp = llm.invoke(messages)
    evaluations = extract_first_json(resp.content)
    record_llm_interaction(state, agent="ReqClarify", iteration=state["iteration"],
                           messages=messages, raw_output=resp.content, parsed_output=evaluations)

    scores_map: Dict[str, int] = {}
    for item in evaluations:
        rid = str(item.get("id"))
        sc = int(item.get("score"))
        scores_map[rid] = sc
    state["scores"] = scores_map

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
    """DocGenerate：将最终需求清单转为 Markdown（IEEE Std 830-1998 基本格式）"""
    log("DocGenerate：生成最终 Markdown SRS 文档（流式输出开始）")
    stream_handler = StreamingPrinter()
    llm = llm or get_llm(temperature=0.1, streaming=True, callbacks=[stream_handler])

    # === 关键修改：输出结构切换为 IEEE Std 830-1998 基本格式 ===
    system = (
        '你是"文档生成智能体（DocGenerate）"。请将优化后的需求清单转化为'
        '遵循 IEEE Std 830-1998 的软件需求规格说明书（SRS），输出介质为 Markdown。\n\n'
        "【结构与编号（须包含以下章节，必要时可精简无内容的小节但不得更名）：】\n"
        "1. Introduction\n"
        "   1.1 Purpose\n"
        "   1.2 Document Conventions\n"
        "   1.3 Intended Audience and Reading Suggestions\n"
        "   1.4 Project Scope\n"
        "   1.5 References\n"
        "2. Overall Description\n"
        "   2.1 Product Perspective\n"
        "   2.2 Product Functions（高层功能概览，非逐条 FR）\n"
        "   2.3 User Characteristics\n"
        "   2.4 Constraints（高层约束，如法规、平台、组织策略）\n"
        "   2.5 Assumptions and Dependencies\n"
        "3. Specific Requirements\n"
        "   3.1 External Interface Requirements（User / Hardware / Software / Communications）\n"
        "   3.2 Functional Requirements（逐条列出，必须保留原始 id，如 FR-01；"
        "每条包含：触发条件→执行逻辑→结果输出→异常兜底；应具可验证性和可追踪性）\n"
        "   3.3 Performance Requirements（将 NFR-* 中与性能相关的条目归入，保留原始 id）\n"
        "   3.4 Design Constraints（将 CON-* 或约束性 NFR 归入，保留原始 id）\n"
        "   3.5 Software System Attributes（Reliability、Availability、Security、Maintainability、Portability 等；"
        "映射 NFR-*，保留原始 id）\n"
        "   3.6 Other Requirements（无法归入以上小节但仍必要的条目；保留原始 id）\n"
        "4. Appendices（可选）\n"
        "5. Index（可选）\n\n"
        "【需求编号规范】\n"
        "• 需求 id 格式：FR-01, FR-02, ...（功能需求，两位数字）；NFR-01, NFR-02, ...（非功能需求）；CON-01, CON-02, ...（约束）\n"
        "• 必须保留并原样展示清单中的需求 id，不得修改或重新编号\n\n"
        "【强制要求】\n"
        "• 将 FR-* 归入 3.2；将 NFR-* 与 CON-* 根据语义分别放入 3.3/3.4/3.5/3.6；若无法判断则置于 3.6，并注明理由。\n"
        "• 语言正式、无二义、可测试；避免使用'可能/大概/TBD'等不确定措辞。\n"
        "• 可使用表格或列表提高可读性，但章节与编号必须符合 830-1998 的基本结构。"
    )

    user = (
        "最终需求清单（JSON 数组）：\n"
        f"```json\n{json.dumps(state['req_list'], ensure_ascii=False)}\n```\n\n"
        "请输出遵循 IEEE Std 830-1998 基本格式的 Markdown 版本 SRS："
    )
    messages = [{"role": "system", "content": system},{"role": "user", "content": user}]

    record_llm_interaction(state, agent="DocGenerate", iteration=state["iteration"],
                           messages=messages, raw_output=None)

    print("\n====== 实时 Markdown SRS 输出（流式） ======\n", flush=True)
    resp = llm.invoke(messages)
    print("\n====== 流式输出结束 ======\n", flush=True)

    raw_content = resp.content
    full_text = raw_content if isinstance(raw_content, str) else \
                ("".join(str(part) for part in raw_content) if isinstance(raw_content, list) else str(raw_content) or "")
    if not full_text:
        full_text = stream_handler.get_text()

    state["srs_output"] = full_text
    state["srs_stream_printed"] = True
    log("DocGenerate：流式输出完成")
    return state


def sim_evaluate_node(state: GraphState) -> GraphState:
    """SimEvaluate：嵌入（不分片）计算并打印两类相似度与向量维度：
        A) 生成文档（srs_output） vs 基准 SRS（reference_srs）
        B) 用户输入（user_input） vs 基准 SRS（reference_srs）
    """
    log("SimEvaluate：开始计算文本嵌入相似度（不分片）")
    if not state.get("srs_output"):
        log("SimEvaluate：未检测到生成文档内容（srs_output 为空），仍将计算 user_input vs reference_srs。")

    try:
        emb = get_embeddings_model()

        # --- (A) 生成文档 vs 基准 SRS ---
        if state.get("srs_output"):
            vec_gen = emb.embed_query(state["srs_output"])
            vec_ref = emb.embed_query(state["reference_srs"])
            dim_gen = len(vec_gen)
            dim_ref = len(vec_ref)

            state["embedding_dim_gen"] = dim_gen
            state["embedding_dim_ref"] = dim_ref

            print(f"\n====== 嵌入信息（SimEvaluate: Generated vs Reference） ======\n"
                  f"Embedding Dim (Generated) : {dim_gen}\n"
                  f"Embedding Dim (Reference) : {dim_ref}\n", flush=True)
            log(f"SimEvaluate：gen/ref 维度 gen={dim_gen}, ref={dim_ref}")

            if dim_gen != dim_ref:
                log("SimEvaluate：警告——gen/ref 维度不一致，跳过 gen vs ref 相似度计算")
                state["embedding_similarity"] = None
                state["embedding_distance"] = None
            else:
                sim_gr = cosine_similarity(vec_gen, vec_ref)
                dist_gr = 1.0 - sim_gr
                state["embedding_similarity"] = sim_gr
                state["embedding_distance"] = dist_gr
                print(f"====== 嵌入相似度（Generated vs Reference） ======\n"
                      f"Cosine Similarity : {sim_gr:.6f}\n"
                      f"Distance (1 - cos): {dist_gr:.6f}\n", flush=True)
                log(f"SimEvaluate：gen vs ref 相似度={sim_gr:.6f}，距离={dist_gr:.6f}")
        else:
            state["embedding_dim_gen"] = None
            state["embedding_dim_ref"] = None
            state["embedding_similarity"] = None
            state["embedding_distance"] = None

        # --- (B) 用户输入 vs 基准 SRS ---
        vec_user = emb.embed_query(state["user_input"])
        vec_ref2 = emb.embed_query(state["reference_srs"])
        dim_user = len(vec_user)
        dim_ref2 = len(vec_ref2)

        state["embedding_dim_user"] = dim_user
        # 对 ref 维度，若已有（A）则无需覆盖；此处仅用于日志展示
        print(f"\n====== 嵌入信息（SimEvaluate: UserInput vs Reference） ======\n"
              f"Embedding Dim (UserInput)  : {dim_user}\n"
              f"Embedding Dim (Reference)  : {dim_ref2}\n", flush=True)
        log(f"SimEvaluate：user/ref 维度 user={dim_user}, ref={dim_ref2}")

        if dim_user != dim_ref2:
            log("SimEvaluate：警告——user/ref 维度不一致，跳过 user vs ref 相似度计算")
            state["embedding_similarity_user_ref"] = None
            state["embedding_distance_user_ref"] = None
        else:
            sim_ur = cosine_similarity(vec_user, vec_ref2)
            dist_ur = 1.0 - sim_ur
            state["embedding_similarity_user_ref"] = sim_ur
            state["embedding_distance_user_ref"] = dist_ur
            print(f"====== 嵌入相似度（UserInput vs Reference） ======\n"
                  f"Cosine Similarity : {sim_ur:.6f}\n"
                  f"Distance (1 - cos): {dist_ur:.6f}\n", flush=True)
            log(f"SimEvaluate：user vs ref 相似度={sim_ur:.6f}，距离={dist_ur:.6f}")

    except Exception as e:
        # 统一失败兜底
        state["embedding_dim_gen"] = state.get("embedding_dim_gen", None)
        state["embedding_dim_ref"] = state.get("embedding_dim_ref", None)
        state["embedding_dim_user"] = state.get("embedding_dim_user", None)
        state["embedding_similarity"] = None
        state["embedding_distance"] = None
        state["embedding_similarity_user_ref"] = None
        state["embedding_distance_user_ref"] = None
        log(f"SimEvaluate：计算嵌入相似度失败：{e}")

    return state


# -----------------------------
# 条件路由：是否继续迭代
# -----------------------------
def should_continue(state: GraphState) -> str:
    """达到最大迭代轮数后结束，否则继续。"""
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
    graph.add_node("SimEvaluate", sim_evaluate_node)

    graph.set_entry_point("ReqParse")
    graph.add_edge("ReqParse", "ReqExplore")
    graph.add_edge("ReqExplore", "ReqClarify")
    graph.add_conditional_edges("ReqClarify", should_continue, {
        "ReqExplore": "ReqExplore",
        "DocGenerate": "DocGenerate"
    })
    graph.add_edge("DocGenerate", "SimEvaluate")
    graph.add_edge("SimEvaluate", END)
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
        # 冻结机制
        "frozen_ids": [],
        "frozen_reqs": {},
        # gen vs ref
        "embedding_similarity": None,
        "embedding_distance": None,
        "embedding_dim_gen": None,
        "embedding_dim_ref": None,
        # user vs ref
        "embedding_similarity_user_ref": None,
        "embedding_distance_user_ref": None,
        "embedding_dim_user": None,
    }
    final_state = app.invoke(init)
    log("流程结束，输出结果")
    if final_state["srs_stream_printed"]:
        print("\n====== Markdown SRS 输出已通过流式打印展示 ======\n", flush=True)
    else:
        print("\n====== 最终 Markdown SRS 输出 ======\n", flush=True)
        print(final_state["srs_output"], flush=True)

    # 摘要回显（SimEvaluate 已打印详细信息）
    print("\n====== 相似度摘要 ======", flush=True)
    print("Dims => gen: {}, ref: {}, user: {}".format(
        final_state.get("embedding_dim_gen"),
        final_state.get("embedding_dim_ref"),
        final_state.get("embedding_dim_user")
    ), flush=True)
    sim_gr = final_state.get("embedding_similarity")
    dst_gr = final_state.get("embedding_distance")
    sim_ur = final_state.get("embedding_similarity_user_ref")
    dst_ur = final_state.get("embedding_distance_user_ref")
    if sim_gr is not None and dst_gr is not None:
        print(f"Generated vs Reference => Cosine: {sim_gr:.6f}, Distance: {dst_gr:.6f}", flush=True)
    else:
        print("Generated vs Reference => 未计算（维度不一致或错误）", flush=True)
    if sim_ur is not None and dst_ur is not None:
        print(f"UserInput vs Reference => Cosine: {sim_ur:.6f}, Distance: {dst_ur:.6f}", flush=True)
    else:
        print("UserInput vs Reference => 未计算（维度不一致或错误）", flush=True)


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
    parser = argparse.ArgumentParser(description="LangGraph 多智能体 SRS 生成演示脚本")
    parser.add_argument("-u", "--user-input", type=str, help="用户需求文本文件路径（user_input）")
    parser.add_argument("-r", "--reference-srs", type=str, help="参考 SRS 文本文件路径（reference_srs）")
    parser.add_argument("-m", "--max-iterations", type=int, default=2, help="最大迭代轮数，默认 5")
    args = parser.parse_args()

    if bool(args.user_input) ^ bool(args.reference_srs):
        parser.error("必须同时提供 --user-input 与 --reference-srs。")

    if args.user_input and args.reference_srs:
        user_input_text = read_text_file(args.user_input, "user_input")
        reference_srs_text = read_text_file(args.reference_srs, "reference_srs")
    else:
        user_input_text = DEFAULT_USER_INPUT
        reference_srs_text = DEFAULT_REFERENCE_SRS

    demo = DemoInput(user_input=user_input_text, reference_srs=reference_srs_text, max_iterations=args.max_iterations)
    if not args.user_input:
        log("未提供参数，使用内置示例场景")
    run_demo(demo)


if __name__ == "__main__":
    main()
