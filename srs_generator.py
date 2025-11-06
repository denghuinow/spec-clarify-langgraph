# -*- coding: utf-8 -*-
"""
Multi-Agent SRS Generation with LangGraph
-----------------------------------------
流程：ReqParse -> ReqExplore <-> ReqClarify -> DocGenerate -> SimEvaluate -> END

- ReqClarify：逐条评分（+2 强采纳，+1 采纳，0 中性，-1 不采纳，-2 强不采纳）
- ReqExplore：仅基于分数优化（只处理未冻结项），输出新版本清单（JSON 数组，仅含 id/content）
- DocGenerate：输出 Markdown（IEEE Std 830-1998 基本格式）
- SimEvaluate：文本嵌入评估两类相似度并打印：
    1) 生成 SRS（srs_output） vs 基准 SRS（reference_srs）
    2) 用户输入（user_input） vs 基准 SRS（reference_srs）
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

# 从独立模块导入文本相似度计算函数
from text_similarity import compute_direct_distance


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
    llm = get_llm(model=model, temperature=temp, streaming=streaming, callbacks=callbacks)
    log(f"{agent} LLM 初始化：temperature={temp}")
    return llm


def get_embeddings_model(model: Optional[str] = None):
    model = model or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs: Dict[str, Any] = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)


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


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


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
    scores: Dict[str, int]           # {id: score}
    logs: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    srs_output: str
    srs_stream_printed: bool

    frozen_ids: List[str]
    frozen_reqs: Dict[str, str]

    removed_ids: List[str]

    embedding_similarity: Optional[float]
    embedding_similarity_user_ref: Optional[float]

    ablation_mode: Optional[str]     # 消融实验模式：no-clarify 或 no-explore-clarify


# -----------------------------
# 节点实现
# -----------------------------
def req_parse_node(state: GraphState, llm=None) -> GraphState:
    """ReqParse：自然语言 -> 初始需求清单（JSON 数组，仅含 id 与 content）"""
    llm = llm or get_llm_for("ReqParse")
    log("ReqParse：开始解析用户输入")
    current_iteration = state["iteration"] + 1

    user_template = (
        '你是"需求解析智能体（ReqParse）"，负责将自然语言需求转为**原子化**的初始清单。\n'
        '\n'
        '【解析方法（在内部完成，不要出现在输出中）】\n'
        '1) 识别三元组：<角色/Actor, 动作/Action, 客体/Object>，并映射到业务对象（如 用户/管理员/系统/外部服务）。\n'
        '2) 原子化拆分：一条需求仅描述**单一可验证行为或约束**；遇到"且/并且/以及/或/同时"等连接词，必要时拆分为多条。\n'
        '3) 去重与同义合并：合并等义表述，统一术语与量纲；避免重复。\n'
        '4) 分类判别（启发式）：\n'
        '   • FR-*（功能）：描述输入→处理→输出的可观察行为或接口交互。\n'
        '   • NFR-*（非功能）：性能/容量（吞吐/并发/响应时延/峰值）、可靠性/可用性、可维护性、可移植性、\n'
        '     安全（鉴权/审计/最小权限/加密/合规）、可观测性（日志/指标/追踪）。\n'
        '   • CON-*（约束）：法律/合规/组织策略/平台边界/外部依赖/数据主权/部署与网络限制等不以功能体验为主的**强约束**。\n'
        '5) 术语锚定：\n'
        '   • 保留用户输入中的专有名词与单位；禁止发明新术语；如需补足名词，使用"前提：…"嵌入到 content 末尾。\n'
        '6) 质量门槛：避免"可能/尽量/适当/TBD/？"等模糊词；可测试、可追踪；禁止问题式/澄清式输出。\n'
        '\n'
        '【编号规范】\n'
        '- 功能需求：FR-01, FR-02, FR-03, ...（两位数字递增，起始01，连续且唯一）\n'
        '- 非功能需求：NFR-01, NFR-02, ...（两位数字递增）\n'
        '- 约束：CON-01, CON-02, ...（两位数字递增）\n'
        '编号不得跳号或重复；三类序列相互独立。\n'
        '\n'
        '【输出格式（仅 JSON 数组，元素字段仅含 id 与 content）】\n'
        '```json\n'
        '[{"id":"FR-01","content":"……"}]\n'
        '```\n'
        '不得出现除 id、content 之外的字段；不得输出解释、表格或注释。\n'
        '\n'
        '---\n'
        '\n'
        '用户需求：\n'
        '{user_input}\n'
        '\n'
        '请仅输出结构化 JSON 数组（外层使用 ```json 围栏，数组元素仅含 id 与 content）。'
    )

    user = user_template.format(user_input=state['user_input'])

    messages = [{"role": "user", "content": user}]
    resp = llm.invoke(messages)
    parsed = extract_first_json(resp.content)
    record_llm_interaction(
        state, agent="ReqParse", iteration=current_iteration,
        messages=messages, raw_output=resp.content, parsed_output=parsed
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
    log(f"ReqExplore：第 {state['iteration']} 轮，根据评分优化需求清单（冻结项不参与）")
    state.setdefault('frozen_ids', [])
    state.setdefault('frozen_reqs', {})
    state.setdefault('removed_ids', [])
    frozen_ids_set = set(state['frozen_ids'])
    removed_ids_set = set(state['removed_ids'])

    # 仅“未冻结&未移除”参与
    unfrozen_list = [
        it for it in state['req_list']
        if it['id'] not in frozen_ids_set and it['id'] not in removed_ids_set
    ]

    scores_arr_unfrozen = [{"id": rid, "score": sc}
                           for rid, sc in state.get("scores", {}).items()
                           if rid not in frozen_ids_set and rid not in removed_ids_set]

    if not unfrozen_list:
        log("ReqExplore：无未冻结且未移除的需求，跳过模型调用，需求清单保持不变")
        return state

    frozen_payload = [{"id": fid, "content": state["frozen_reqs"].get(fid, "")}
                      for fid in state["frozen_ids"]]
    removed_payload = [{"id": rid} for rid in state["removed_ids"]]

    # 预先序列化JSON字符串用于模板填充
    frozen_payload_json = json.dumps(frozen_payload, ensure_ascii=False, indent=2) if frozen_payload else '无'
    removed_payload_json = json.dumps(removed_payload, ensure_ascii=False, indent=2) if removed_payload else '无'
    unfrozen_list_json = json.dumps(unfrozen_list, ensure_ascii=False)
    scores_arr_unfrozen_json = json.dumps(scores_arr_unfrozen, ensure_ascii=False)

    user_template = (
        '你是"需求挖掘智能体（ReqExplore）"，资深需求工程师与闭环设计专家。只对**未冻结且未移除**的条目进行优化；\n'
        '冻结条目与已移除条目严禁修改，也不要出现在输出中（系统稍后合并）。\n'
        '\n'
        '【你将收到】\n'
        '- 未冻结条目清单（id, content）\n'
        '- 这些条目的评分（来自上一轮澄清）\n'
        '- （只读）冻结清单\n'
        '- （只读）已移除清单（禁止复提/改写）\n'
        '\n'
        '【动作对照（严格遵循评分）】\n'
        '+2（强采纳）：仅做文字精炼与条理化；**不新增**条目，不改变语义。\n'
        '+1（采纳）：在**不改变意图**前提下补齐验证口径：输入/输出、权限与审计、异常与回退、数据一致性与幂等、\n'
        '             可观测性（日志/指标/追踪）；可进行轻量结构化；如确需支撑，可近邻新增 ≤2 条直接依赖项。\n'
        '0（中性）：保持或泛化表述，将不确定假设前置为"前提：…"，并转换为**可检验**表达；**不新增**条目。\n'
        '-1（不采纳）：在不扩张范围的前提下重写或替代，使之贴合**角色/触发/结果/失败策略/一致性**；可新增 ≤2 条近邻支撑项。\n'
        '-2（强不采纳）：**从清单中剔除该 id**；不得新增替代；且**禁止以 FR/NFR/CON/SUG 任意形式复提或近义改写**。\n'
        '\n'
        '【闭环模板（建议融合到 content 内）】\n'
        '触发：…  输入：…  处理：…  输出：…  异常与回退：…  访问控制与审计：…  一致性与幂等：…  可观测性：…  前提：…\n'
        '\n'
        '【近邻外推（新增边界）】\n'
        '- 仅围绕当前条目的必要上下游步骤与合规/可靠/可观测性支撑；禁止跨域扩张，不得引入新模块/新角色/新数据域。\n'
        '- "必要"判定：若缺失该支撑，将导致条目**不可验证或不可运行**。\n'
        '\n'
        '【术语与一致性】\n'
        '- 术语沿用既有清单与用户原文；单位/阈值/量纲要明确，不要含混词（如"快速""较多"）。\n'
        '- 不发明新概念；必要假设用"前提：…"列出，勿用 TBD/？ 等占位。\n'
        '\n'
        '【编号与冲突】\n'
        '- 仅输出"未冻结 + 新增"条目；**不要**输出任何冻结或已移除条目。\n'
        '- 新增条目编号：在各自序列（FR/NFR/CON/SUG）末尾顺延；避免与现有 id 冲突；保持未冻结 id 的相对顺序稳定。\n'
        '- 不得复提 removed_ids 中任何 id 或其近义改写。\n'
        '\n'
        '【自检清单】\n'
        '- 是否形成"触发→处理→输出→异常/回退→访问控制/审计→一致性/幂等→可观测性→前提"的可验证闭环？\n'
        '- 是否删除模糊词并给出阈值/口径？是否显式列出"前提：…"？\n'
        '- 是否严格遵循分数动作边界（+2/0 不新增；-2 必剔除且禁复提）？\n'
        '- 输出是否为**合法 JSON**，元素仅含 id 与 content？\n'
        '\n'
        '【（只读）冻结清单】：\n'
        '{frozen_payload_json}\n'
        '\n'
        '【（只读）已移除清单（禁复提/改写）】：\n'
        '{removed_payload_json}\n'
        '\n'
        '---\n'
        '\n'
        '未冻结且未移除清单（JSON）：\n'
        '```json\n'
        '{unfrozen_list_json}\n'
        '```\n'
        '\n'
        '评分（仅未冻结且未移除项，JSON）：\n'
        '```json\n'
        '{scores_arr_unfrozen_json}\n'
        '```\n'
        '\n'
        '（只读）冻结清单（请勿输出或修改）：\n'
        '{frozen_payload_json}\n'
        '\n'
        '（只读）已移除清单（请勿复提/改写）：\n'
        '{removed_payload_json}\n'
        '\n'
        '请输出"仅含未冻结及新增 id"的清单（外层用 ```json 围栏；元素仅含 id, content）。'
        '禁止包含任何冻结或已移除 id。'
    )

    user = user_template.format(
        frozen_payload_json=frozen_payload_json,
        removed_payload_json=removed_payload_json,
        unfrozen_list_json=unfrozen_list_json,
        scores_arr_unfrozen_json=scores_arr_unfrozen_json
    )

    messages = [{"role": "user", "content": user}]
    resp = llm.invoke(messages)
    unfrozen_new_list = extract_first_json(resp.content)
    record_llm_interaction(state, agent="ReqExplore", iteration=state["iteration"],
                           messages=messages, raw_output=resp.content, parsed_output=unfrozen_new_list)

    # 以新输出为准，合并冻结与未移除项；对越界新增/重复做去重校正
    new_map: Dict[str, str] = {}
    for it in unfrozen_new_list:
        rid = str(it.get("id"))
        if not rid or rid in frozen_ids_set or rid in removed_ids_set:
            continue  # 模型越界输出，丢弃
        content = str(it.get("content", "")).strip()
        if not content:
            continue
        new_map[rid] = content  # 若重复，以最后一次为准

    prev_ids_order = [str(it["id"]) for it in state["req_list"]]

    merged: List[Dict[str, str]] = []
    for rid in prev_ids_order:
        if rid in removed_ids_set:
            continue  # 已移除，跳过
        if rid in frozen_ids_set:
            content = state["frozen_reqs"].get(
                rid, next((x["content"] for x in state["req_list"] if x["id"] == rid), "")
            )
            merged.append({"id": rid, "content": content})
        else:
            # 未冻结 -> 采用 new_map 中的更新；若不存在，则沿用旧值（模型可能未返回）
            content = new_map.get(
                rid, next((x["content"] for x in state["req_list"] if x["id"] == rid), "")
            )
            merged.append({"id": rid, "content": content})

    # 附加“真正新增”的条目（不在 prev_ids 中，且未移除）
    for rid, content in new_map.items():
        if rid not in prev_ids_order and rid not in removed_ids_set:
            merged.append({"id": rid, "content": content})

    state["req_list"] = merged
    log(f"ReqExplore：完成合并，共 {len(state['req_list'])} 条；冻结 {len(state['frozen_ids'])} 条；已移除 {len(state['removed_ids'])} 条")
    return state


def req_clarify_node(state: GraphState, llm=None) -> GraphState:
    llm = llm or get_llm_for("ReqClarify")
    log(f"ReqClarify：第 {state['iteration']} 轮，对需求逐条评分（排除冻结项）")
    state.setdefault('frozen_ids', [])
    state.setdefault('frozen_reqs', {})
    state.setdefault('removed_ids', [])
    frozen_ids_set = set(state['frozen_ids'])

    unfrozen_list = [item for item in state["req_list"] if item["id"] not in frozen_ids_set]
    if not unfrozen_list:
        log("ReqClarify：所有需求均已冻结，本轮跳过评分")
        state["scores"] = {}
        state["iteration"] += 1
        return state

    # 预先序列化JSON字符串用于模板填充
    unfrozen_list_json = json.dumps(unfrozen_list, ensure_ascii=False)

    user_template = (
        '你是"需求澄清智能体（ReqClarify）"，以**需求方/验收方**立场对"未冻结"的需求逐条评分；'
        '以"基准 SRS"为唯一标尺，锚定术语/量纲/阈值/角色称谓，禁止引入新需求或改写原文。\n'
        '\n'
        '【评分口径】\n'
        '+2（强采纳）：与基准 SRS 一致或更优；语义完整、可测试、边界清晰、无含糊词。\n'
        '+1（采纳）：基本一致，仅缺少少量可验证细节（异常/权限/可观测性等）。\n'
        '0（中性）：信息不足或不确定假设过多，需条件化或补充验证口径后再评估。\n'
        '-1（不采纳）：与角色/范围/触发/结果/失败策略/一致性等存在明显缺口或偏差，应重写或替代。\n'
        '-2（强不采纳）：与目标/约束/合规严重冲突或跨域；**应直接移除，并在后续迭代禁止复提或近义改写**。\n'
        '\n'
        '【一致性约束】\n'
        '• 必须对输入的每一条未冻结需求逐条给出评分；顺序与长度与输入一致；id 一一对应。\n'
        '• 仅输出一个 ```json 围栏包裹的数组：[{"id":"FR-01","score":1,"reason":"..."}]。\n'
        '• reason 简洁可审计（≤50字）。\n'
        '\n'
        '---\n'
        '\n'
        '需评分（未冻结）需求清单：\n'
        '```json\n'
        '{unfrozen_list_json}\n'
        '```\n'
        '\n'
        '基准 SRS：\n'
        '```text\n'
        '{reference_srs}\n'
        '```\n'
        '\n'
        '请输出评分 JSON 数组（使用 ```json``` 包裹）：'
    )

    user = user_template.format(
        unfrozen_list_json=unfrozen_list_json,
        reference_srs=state['reference_srs']
    )

    messages = [{"role": "user", "content": user}]
    resp = llm.invoke(messages)
    evaluations = extract_first_json(resp.content)
    record_llm_interaction(state, agent="ReqClarify", iteration=state["iteration"],
                           messages=messages, raw_output=resp.content, parsed_output=evaluations)

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
        log(f"ReqClarify：标记移除 {add_cnt} 项（-2），累计移除 {len(state['removed_ids'])} 项")

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
            log(f"ReqClarify：本轮最高正向分 = {max_score}，新增冻结 {added} 项；累计冻结 {len(state['frozen_ids'])} 项")
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
        it for it in state['req_list']
        if it['id'] not in set(state.get('removed_ids', []))
    ]

    # 预先序列化JSON字符串用于模板填充
    effective_req_list_json = json.dumps(effective_req_list, ensure_ascii=False)

    user_template = (
        '你是"文档生成智能体（DocGenerate）"。请将优化后的需求清单转化为'
        '遵循 IEEE Std 830-1998 的软件需求规格说明书（SRS），输出介质为 Markdown。\n'
        '\n'
        '【结构与编号（须包含以下章节，必要时可精简无内容的小节但不得更名）：】\n'
        '1. Introduction\n'
        '   1.1 Purpose\n'
        '   1.2 Document Conventions\n'
        '   1.3 Intended Audience and Reading Suggestions\n'
        '   1.4 Project Scope\n'
        '   1.5 References\n'
        '2. Overall Description\n'
        '   2.1 Product Perspective\n'
        '   2.2 Product Functions（高层功能概览，非逐条 FR）\n'
        '   2.3 User Characteristics\n'
        '   2.4 Constraints（高层约束，如法规、平台、组织策略）\n'
        '   2.5 Assumptions and Dependencies\n'
        '3. Specific Requirements\n'
        '   3.1 External Interface Requirements（User / Hardware / Software / Communications）\n'
        '   3.2 Functional Requirements（逐条列出，必须保留原始 id，如 FR-01；'
        '   3.3 Performance Requirements（将 NFR-* 中与性能相关的条目归入，保留原始 id）\n'
        '   3.4 Design Constraints（将 CON-* 或约束性 NFR 归入，保留原始 id）\n'
        '   3.5 Software System Attributes（Reliability、Availability、Security、Maintainability、Portability 等；'
        '映射 NFR-*，保留原始 id）\n'
        '   3.6 Other Requirements（无法归入以上小节但仍必要的条目；保留原始 id）\n'
        '4. Appendices（可选）\n'
        '5. Index（可选）\n'
        '\n'
        '【需求编号规范】\n'
        '• 需求 id 格式：FR-01, FR-02, ...；NFR-01, NFR-02, ...；CON-01, CON-02, ...\n'
        '• 必须保留并原样展示清单中的需求 id，不得修改或重新编号\n'
        '\n'
        '【强制要求】\n'
        '• 将 FR-* 归入 3.2；将 NFR-* 与 CON-* 根据语义分别放入 3.3/3.4/3.5/3.6；若无法判断则置于 3.6，并注明理由。\n'
        '• 语言正式、无二义、可测试；避免使用"可能/大概/TBD"等不确定措辞。\n'
        '\n'
        '---\n'
        '\n'
        '最终需求清单（JSON 数组）：\n'
        '```json\n'
        '{effective_req_list_json}\n'
        '```\n'
        '\n'
        '请输出遵循 IEEE Std 830-1998 基本格式的 Markdown 版本 SRS：'
    )

    user = user_template.format(effective_req_list_json=effective_req_list_json)

    messages = [{"role": "user", "content": user}]

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
    """SimEvaluate：嵌入计算并打印两类相似度：
        A) 生成文档（srs_output） vs 基准 SRS（reference_srs）
        B) 用户输入（user_input） vs 基准 SRS（reference_srs）
        
        使用直接调用 OpenAI embeddings API 的方式，进行 L2 归一化后计算点积（余弦相似度）
    """
    log("SimEvaluate：开始计算文本嵌入相似度（使用直接距离计算方法）")
    if not state.get("srs_output"):
        log("SimEvaluate：未检测到生成文档内容（srs_output 为空），仍将计算 user_input vs reference_srs。")

    try:
        embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

        # --- (A) 生成文档 vs 基准 SRS ---
        if state.get("srs_output"):
            sim_gr = compute_direct_distance(
                state["srs_output"],
                state["reference_srs"],
                embedding_model=embedding_model
            )
            state["embedding_similarity"] = sim_gr
            print(f"====== 嵌入相似度（Generated vs Reference） ======\n"
                  f"Cosine Similarity : {sim_gr:.6f}\n", flush=True)
        else:
            state["embedding_similarity"] = None

        # --- (B) 用户输入 vs 基准 SRS ---
        sim_ur = compute_direct_distance(
            state["user_input"],
            state["reference_srs"],
            embedding_model=embedding_model
        )
        state["embedding_similarity_user_ref"] = sim_ur
        print(f"====== 嵌入相似度（UserInput vs Reference） ======\n"
              f"Cosine Similarity : {sim_ur:.6f}\n", flush=True)

    except Exception as e:
        state["embedding_similarity"] = None
        state["embedding_similarity_user_ref"] = None
        log(f"SimEvaluate：计算嵌入相似度失败：{e}")

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
    graph.add_node("SimEvaluate", sim_evaluate_node)

    graph.set_entry_point("ReqParse")

    if ablation_mode == "no-explore-clarify":
        # 模式：移除 ReqExplore + ReqClarify
        # 流程：ReqParse -> DocGenerate -> SimEvaluate
        log("构建图结构：no-explore-clarify 模式（跳过 ReqExplore 和 ReqClarify）")
        graph.add_edge("ReqParse", "DocGenerate")
    elif ablation_mode == "no-clarify":
        # 模式：移除 ReqClarify
        # 流程：ReqParse -> ReqExplore -> DocGenerate -> SimEvaluate
        log("构建图结构：no-clarify 模式（跳过 ReqClarify，ReqExplore 执行一次）")
        graph.add_node("ReqExplore", req_explore_node)
        graph.add_edge("ReqParse", "ReqExplore")
        graph.add_conditional_edges("ReqExplore", should_continue, {
            "DocGenerate": "DocGenerate"
        })
    else:
        # 默认模式：完整流程
        # 流程：ReqParse -> ReqExplore <-> ReqClarify -> DocGenerate -> SimEvaluate
        log("构建图结构：默认模式（完整流程）")
        graph.add_node("ReqExplore", req_explore_node)
        graph.add_node("ReqClarify", req_clarify_node)
        graph.add_edge("ReqParse", "ReqExplore")
        graph.add_edge("ReqExplore", "ReqClarify")
        graph.add_conditional_edges("ReqClarify", should_continue, {
            "ReqExplore": "ReqExplore",
            "DocGenerate": "DocGenerate"
        })

    graph.add_edge("DocGenerate", "SimEvaluate")
    graph.add_edge("SimEvaluate", END)
    return graph.compile()


class DemoInput(BaseModel):
    user_input: str = Field(..., description="自然语言需求")
    reference_srs: str = Field(..., description="基准SRS（文本）")
    max_iterations: int = 5
    ablation_mode: Optional[str] = Field(default=None, description="消融实验模式：no-clarify 或 no-explore-clarify")


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
        "embedding_similarity": None,
        "embedding_similarity_user_ref": None,
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

        print("\n====== 相似度摘要 ======", flush=True)
        sim_gr = final_state.get("embedding_similarity")
        sim_ur = final_state.get("embedding_similarity_user_ref")
        if sim_gr is not None:
            print(f"Generated vs Reference => Cosine: {sim_gr:.6f}", flush=True)
        else:
            print("Generated vs Reference => 未计算", flush=True)
        if sim_ur is not None:
            print(f"UserInput vs Reference => Cosine: {sim_ur:.6f}", flush=True)
        else:
            print("UserInput vs Reference => 未计算", flush=True)
    
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
    parser.add_argument("-u", "--user-input", type=str, help="用户需求文本文件路径（user_input）")
    parser.add_argument("-r", "--reference-srs", type=str, help="参考 SRS 文本文件路径（reference_srs）")
    # 2) 直接文本（优先级高于文件）
    parser.add_argument("--user-text", type=str, help="用户需求文本内容，直接传入字符串")
    parser.add_argument("--reference-text", type=str, help="参考 SRS 文本内容，直接传入字符串")
    # 3) 迭代轮数
    parser.add_argument("-m", "--max-iterations", type=int, default=5, help="最大迭代轮数，默认 5")
    # 4) 消融实验模式
    parser.add_argument(
        "--ablation-mode",
        type=str,
        choices=["no-clarify", "no-explore-clarify"],
        default=None,
        help="消融实验模式：no-clarify（移除需求澄清智能体）或 no-explore-clarify（移除需求挖掘+澄清智能体）"
    )

    args = parser.parse_args()

    # 参数校验与选择：优先使用直接文本；其次文件；都缺失则使用内置示例
    has_text_pair = bool(args.user_text) and bool(args.reference_text)
    has_file_pair = bool(args.user_input) and bool(args.reference_srs)

    if (args.user_text and not args.reference_text) or (args.reference_text and not args.user_text):
        parser.error("直接文本模式必须同时提供 --user-text 与 --reference-text。")
    if (args.user_input and not args.reference_srs) or (args.reference_srs and not args.user_input):
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
        ablation_mode=args.ablation_mode
    )
    run_demo(demo)


if __name__ == "__main__":
    main()
