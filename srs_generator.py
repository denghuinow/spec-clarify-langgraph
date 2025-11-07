# -*- coding: utf-8 -*-
"""
Multi-Agent SRS Generation with LangGraph
-----------------------------------------
流程：ReqParse -> 并行处理所有原子需求（ReqExplore <-> ReqClarify）-> 汇总 -> DocGenerate -> END

- ReqParse：自然语言 -> 原子需求字符串数组
- ParallelProcessAtomicReqs：并行处理所有原子需求，每个原子需求独立进行 ReqExplore <-> ReqClarify 迭代循环
  - ReqExplore：单条原子需求 -> 细化需求列表（支持迭代优化）
  - ReqClarify：对当前原子需求的细化需求进行评分
- DocGenerate：输出 Markdown（IEEE Std 830-1998 基本格式）

并行度配置：
- 环境变量 PARALLEL_WORKERS：设置并行工作线程数
- 命令行参数 --parallel-workers：覆盖环境变量
- 默认值：3 个工作线程
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
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


def get_parallel_workers(default: Optional[int] = None) -> int:
    """
    获取并行处理的工作线程数。
    优先级：环境变量 PARALLEL_WORKERS > 参数 default > 默认值 3
    
    Args:
        default: 默认值（如果环境变量未设置）
    
    Returns:
        并行工作线程数（至少为 1）
    """
    env_value = os.getenv("PARALLEL_WORKERS")
    if env_value:
        try:
            workers = int(env_value)
            if workers > 0:
                return workers
        except ValueError:
            pass
    
    if default is not None and default > 0:
        return default
    
    return 3  # 默认值


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
    parallel_workers: Optional[int]  # 并行处理的工作线程数
    req_max_scores: Dict[str, int]  # 每个需求的历史最高分 {req_id: max_score}
    req_scores_history: Dict[str, List[int]]  # 每个需求的所有评分历史 {req_id: [score1, score2, ...]}
    req_first_iteration: Dict[str, int]  # 每个需求首次出现的迭代轮次 {req_id: iteration}
    req_last_iteration: Dict[str, int]  # 每个需求最后出现的迭代轮次 {req_id: iteration}
    req_last_clarify_iteration: Dict[str, int]  # 每个需求最后一次被评分的迭代轮次 {req_id: iteration}


# -----------------------------
# 节点实现
# -----------------------------
def process_single_atomic_req(
    atomic_req_index: int,
    atomic_req: str,
    user_input: str,
    reference_srs: str,
    max_iterations: int,
    logs: List[Dict[str, Any]],
    logs_lock: Lock,
    iteration: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, str]]], Dict[int, int], Dict[str, int], Dict[str, List[int]], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    处理单个原子需求的完整流程（ReqExplore <-> ReqClarify 迭代循环）。
    
    Args:
        atomic_req_index: 原子需求索引
        atomic_req: 原子需求文本
        user_input: 用户原始输入
        reference_srs: 参考 SRS
        max_iterations: 最大迭代次数
        logs: 日志列表（用于记录）
        logs_lock: 日志列表的锁（用于线程安全）
        iteration: 当前迭代轮数
    
    Returns:
        (normalized_reqs, explore_history, scores_count, req_max_scores, req_scores_history, 
         req_first_iteration, req_last_iteration, req_last_clarify_iteration):
        - normalized_reqs: 该原子需求处理后的细化需求列表（包含 iteration 和 score_in_iteration 字段）
        - explore_history: 该原子需求的对话历史
        - scores_count: 该原子需求的打分次数
        - req_max_scores: 该原子需求的需求历史最高分
        - req_scores_history: 该原子需求的需求评分历史
        - req_first_iteration: 该原子需求的需求首次出现迭代
        - req_last_iteration: 该原子需求的需求最后出现迭代
        - req_last_clarify_iteration: 该原子需求的需求最后评分迭代
    """
    llm_explore = get_llm_for("ReqExplore")
    llm_clarify = get_llm_for("ReqClarify")
    
    log(f"开始并行处理原子需求 {atomic_req_index + 1}: {atomic_req[:50]}...")
    
    # 初始化该原子需求的状态
    explore_history: List[Dict[str, str]] = []
    scores_count = 0
    normalized_reqs: List[Dict[str, Any]] = []
    scores: Dict[str, int] = {}
    
    # 初始化跟踪数据结构
    req_max_scores: Dict[str, int] = {}
    req_scores_history: Dict[str, List[int]] = {}
    req_first_iteration: Dict[str, int] = {}
    req_last_iteration: Dict[str, int] = {}
    req_last_clarify_iteration: Dict[str, int] = {}
    
    # 构建系统提示词（ReqExplore）
    system_template_explore = """你是"软件工程需求挖掘智能体"，一名资深软件需求工程师与业务闭环设计专家。你的任务是：

1. 输入处理：用户会提供：
   • 【原始需求】：背景、目标、现状、限制等
   • 【原子需求】：一条待细化的功能或规则

2. 输出格式：你将原子需求细化为多条结构化条目，输出一个 JSON 数组，数组每个元素包含：
   {
     "id": "REQ-001",
     "text": "需求描述"
   }

3. 每条"需求描述"应自然流畅、面向产品/研发/测试三方协作，覆盖以下闭环要素（以自然语言表述，非分段）：
   - 功能意图与使用场景
   - 触发条件
   - 处理逻辑
   - 输出或系统反馈
   - 异常处理 / 回退策略
   - 访问控制 / 操作审计
   - 幂等性 / 一致性要求
   - 可观测性（如埋点、日志）
   - 前提条件

4. 每次仅针对一条【原子需求】细化生成多个子条目（每条聚焦单一闭环），避免功能堆叠。
5. 用户会用 JSON 数组评分你的输出，如：[{"id":"REQ-001", "score":1}]，你将根据评分逐条迭代并更新。
6. 评分规则：+2 强采纳，+1 采纳，0 中性建议，-1 不采纳，-2 强不采纳。
7. 不输出接口字段、数据结构等技术实现，仅描述"系统应该做什么、表现为何、如何被感知验证"。
8. 所有输出仅为 JSON 数组，无解释性语言。
"""
    
    # 构建系统提示词（ReqClarify）
    system_template_clarify = """你是"需求澄清智能体"。从验收方视角出发，对需求清单逐条依据"基准 SRS"进行评分。

评分规则为5分制：强采纳 +2；采纳 +1；中性意见 0；不采纳 -1；强不采纳 -2。以基准 SRS 为唯一标尺，锚定术语、量纲、阈值、角色称谓；不得引入新需求、不得改写原文、不得扩展范围。

当输入同时包含：
- 需求清单：代码围栏，类型为 json，包含 [{"id":..., "text":...}] 数组；
- 基准 SRS：代码围栏，类型为 text。

此时进入评分模式，仅输出一个 json 围栏包裹的评分数组，格式如：[{"id":"FR-01","score":1,"reason":"..."}]。

输出要求：
- 与输入顺序完全一致；id 一一对应；
- reason 为中文、≤50字，仅指出审计差异点（术语不一致/阈值冲突/表述含糊/无 SRS 对应等）；
- 不提出建议、不引入修改、不输出任何非 JSON 内容。

若未同时提供需求清单与 SRS，则不进入评分模式，而是提供结构化提示与输入模板。

默认不追问澄清，除非缺少必要输入。输出风格应简洁、严谨、可审计。评分依据"是否与 SRS 保持一致性"为唯一标准。
"""
    
    # 首次 ReqExplore
    initial_message = {
        "role": "user",
        "content": f"""【原始需求】
{user_input}

【原子需求】
{atomic_req}
"""
    }
    explore_history.append(initial_message)
    
    # 迭代循环：ReqExplore <-> ReqClarify
    while scores_count < max_iterations:
        # ReqExplore 阶段
        messages_explore = [{"role": "system", "content": system_template_explore}] + explore_history
        parsed, raw_output = invoke_with_json_retry(llm_explore, messages_explore, max_retries=3)
        
        # 记录交互（线程安全）
        entry: Dict[str, Any] = {
            "iteration": iteration,
            "agent": "ReqExplore",
            "input_messages": messages_explore,
            "raw_output": raw_output if isinstance(raw_output, str) else str(raw_output),
            "parsed_output": parsed,
        }
        with logs_lock:
            logs.append(entry)
        
        # 保存响应到历史
        assistant_message = {"role": "assistant", "content": raw_output}
        explore_history.append(assistant_message)
        
        # 规范化输出
        normalized_reqs = []
        current_iteration = scores_count  # 当前迭代轮次（从0开始）
        if isinstance(parsed, list):
            req_id_counter = 1
            for item in parsed:
                if isinstance(item, dict):
                    original_id = str(item.get("id", ""))
                    if original_id and not original_id.startswith(f"A{atomic_req_index}-"):
                        req_id = f"A{atomic_req_index}-{original_id}"
                    elif not original_id:
                        req_id = f"A{atomic_req_index}-REQ-{req_id_counter:03d}"
                    else:
                        req_id = original_id
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
        
        # 打印生成的细化需求列表
        if normalized_reqs:
            log(f"[ReqExplore] 原子需求 {atomic_req_index + 1} 第 {scores_count + 1} 次迭代生成细化需求：")
            for req in normalized_reqs:
                req_id = req.get("id", "")
                req_text = req.get("text", "")
                # 截断长文本（超过200字符）
                if len(req_text) > 200:
                    req_text_display = req_text[:200] + "..."
                else:
                    req_text_display = req_text
                log(f"  - ID: {req_id}, 文本: {req_text_display}")
        
        if not normalized_reqs:
            log(f"原子需求 {atomic_req_index + 1} 无细化需求，跳过评分")
            break
        
        # ReqClarify 阶段
        req_list_for_scoring = [
            {"id": req.get("id", ""), "text": req.get("text", req.get("content", ""))}
            for req in normalized_reqs
        ]
        req_list_json = json.dumps(req_list_for_scoring, ensure_ascii=False)
        
        user_clarify = f"""【基准 SRS】
```text
{reference_srs}
```
【需求清单】
```json
{req_list_json}
```
"""
        
        messages_clarify = [
            {"role": "system", "content": system_template_clarify},
            {"role": "user", "content": user_clarify}
        ]
        
        evaluations, raw_output_clarify = invoke_with_json_retry(llm_clarify, messages_clarify, max_retries=3)
        
        # 记录交互（线程安全）
        entry_clarify: Dict[str, Any] = {
            "iteration": iteration,
            "agent": "ReqClarify",
            "input_messages": messages_clarify,
            "raw_output": raw_output_clarify if isinstance(raw_output_clarify, str) else str(raw_output_clarify),
            "parsed_output": evaluations,
        }
        with logs_lock:
            logs.append(entry_clarify)
        
        # 规范化评分
        scores = {}
        scored_req_ids = set()  # 记录本次被评分的需求ID
        for item in evaluations:
            rid = str(item.get("id"))
            sc_val = item.get("score")
            try:
                sc = int(sc_val)
            except Exception:
                continue
            if rid:
                scores[rid] = sc
                scored_req_ids.add(rid)
                # 更新评分历史
                if rid not in req_scores_history:
                    req_scores_history[rid] = []
                req_scores_history[rid].append(sc)
                # 更新历史最高分
                if rid not in req_max_scores or sc > req_max_scores[rid]:
                    req_max_scores[rid] = sc
                # 记录最后一次被评分的迭代轮次
                req_last_clarify_iteration[rid] = current_iteration
                # 更新当前迭代中该需求的评分
                for req in normalized_reqs:
                    if req.get("id") == rid:
                        req["score_in_iteration"] = sc
                        break
        
        # 打印评分结果
        if scores:
            log(f"[ReqClarify] 原子需求 {atomic_req_index + 1} 第 {scores_count + 1} 次迭代评分结果：")
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
        
        # 对于在 normalized_reqs 中但未在 evaluations 中出现的需求，
        # 不更新 req_last_clarify_iteration，这样在 aggregate_node 中可以识别为"未被评分"
        # 注意：这些需求可能因为 LLM 输出不完整等原因未被评分
        
        scores_count += 1
        
        # 如果有评分反馈，添加到历史以便下次迭代
        if scores:
            scores_arr = [{"id": rid, "score": sc} for rid, sc in scores.items()]
            scores_arr_json = json.dumps(scores_arr, ensure_ascii=False)
            feedback_message = {
                "role": "user",
                "content": f"""【评分】
{scores_arr_json}
"""
            }
            explore_history.append(feedback_message)
        
        log(f"原子需求 {atomic_req_index + 1} 完成第 {scores_count} 次迭代（共生成 {len(normalized_reqs)} 条细化需求）")
    
    log(f"原子需求 {atomic_req_index + 1} 处理完成，共迭代 {scores_count} 次，生成 {len(normalized_reqs)} 条细化需求")
    
    return (normalized_reqs, {atomic_req_index: explore_history}, {atomic_req_index: scores_count},
            req_max_scores, req_scores_history, req_first_iteration, req_last_iteration,
            req_last_clarify_iteration)


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
    
    current_atomic_req = atomic_reqs_queue[atomic_req_index]
    log(f"ReqExplore：处理原子需求 {atomic_req_index + 1}/{len(atomic_reqs_queue)}: {current_atomic_req[:50]}...")
    
    # 获取或初始化对话历史
    explore_history = state.get("atomic_req_explore_history", {})
    if atomic_req_index not in explore_history:
        explore_history[atomic_req_index] = []
    messages_history = explore_history[atomic_req_index]
    
    # 构建系统提示词
    system_template = """你是「软件工程需求挖掘智能体」，一名资深软件需求工程师与业务闭环设计专家。

【你的目标】
将用户输入的一条【原子需求】拆解为多条可执行、可验证、覆盖闭环的需求描述，服务于「产品 / 研发 / 测试」三方协作，而不卷入具体技术实现细节。

【输入说明】
用户会提供：
1. 【原始需求】（可选）：背景、业务目标、现状问题、约束条件等。
2. 【原子需求】（必选）：一条待细化的功能或规则，语义相对单一，但可能包含隐含意图。

【输出要求】
1. 输出格式：仅输出一个 JSON 数组，不添加任何解释性文本。
2. 数组中每个元素为一条独立闭环需求，结构为：
   {
     "id": "REQ-001",
     "text": "需求描述"
   }
3. "id" 从 REQ-001 开始按顺序递增。
4. 每条 "text" 必须是自然语言的单句或短段落，便于直接用于评审与用例设计，不拆成子条列表。

【单条需求描述应体现的闭环要素】
在不生硬拆分的前提下，每条需求描述需尽可能完整体现以下内容（用连贯自然语言表达，而非罗列）：
- 功能意图与适用场景（该能力解决什么问题，用于何种业务情境）
- 触发条件（由谁、在何种前提或事件下触发）
- 处理逻辑（系统应该如何响应与决策的规则）
- 输出或系统反馈（用户或上下游系统可感知的结果）
- 异常处理 / 回退策略（输入异常、规则冲突、依赖失败时系统应如何处理）
- 访问控制 / 操作审计（谁有权限执行/查看，以及关键操作是否需记录）
- 幂等性 / 一致性要求（重复触发或并发情况下的预期行为）
- 可观测性要求（需要哪些埋点、日志、指标，便于验证与监控）
- 前提条件（依赖的配置、数据、外部系统或业务状态）

注意：
- 每条需求描述应形成一个「自洽闭环」，而非简单拼接上述要素。
- 同一【原子需求】可以拆解为多条子需求，每条各自闭环，避免在单条中堆叠多个不相干流程。
- 不输出接口字段名、数据库结构、表名、具体协议等实现层细节，仅描述「系统应该做什么、表现为何、如何被感知与验证」。

【拆解策略】
1. 每次只针对一条【原子需求】进行拆解，生成多个子条目。
2. 支持多层级挖掘：
   - 在原子需求存在隐含前置规则、风控约束、配置入口、审核流程等时，应主动拆解为二级、三级子需求。
3. 遇到信息缺失时，基于【原始需求】及常见行业实践进行合理补全，并保持表述为「应当」而非「也许」。

【评分与自适应迭代】
用户会基于你上一轮输出给出评分 JSON，例如：
[
  { "id": "REQ-001", "score": 2 },
  { "id": "REQ-002", "score": -1 }
]
评分含义：
- +2：强采纳
- +1：采纳
-  0：中性
- -1：不采纳
- -2：强不采纳

当收到评分后：
1. 仅根据对应 id 对相关需求进行增删改或重写。
2. 总结用户偏好（如：粒度、严谨度、业务语气、保守/激进补全程度），在后续输出中自适应调整。
3. 迭代输出时同样严格遵守「仅输出 JSON 数组」的格式要求。

【全局约束】
- 所有输出均为 JSON 数组，无额外说明文字。
- 语言风格：清晰、具体、业务可读，避免空洞表述与技术实现细节。
- 核心定位：面向真实业务场景的需求工程与闭环设计，而非仅做功能拆分罗列。

"""
    
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
                "content": f"""【评分】
{scores_arr_json}
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
    normalized_reqs: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        req_id_counter = 1
        for item in parsed:
            if isinstance(item, dict):
                # 如果模型返回的 ID 不包含原子需求索引，我们需要添加
                original_id = str(item.get("id", ""))
                if original_id and not original_id.startswith(f"A{atomic_req_index}-"):
                    req_id = f"A{atomic_req_index}-{original_id}"
                elif not original_id:
                    req_id = f"A{atomic_req_index}-REQ-{req_id_counter:03d}"
                else:
                    req_id = original_id
                req_text = str(item.get("text", item.get("content", ""))).strip()
                if req_text:
                    normalized_reqs.append({"id": req_id, "text": req_text, "atomic_req_index": atomic_req_index})
                    req_id_counter += 1
            elif isinstance(item, str):
                # 兼容纯字符串格式，自动分配 ID
                if item.strip():
                    req_id = f"A{atomic_req_index}-REQ-{req_id_counter:03d}"
                    normalized_reqs.append({"id": req_id, "text": item.strip(), "atomic_req_index": atomic_req_index})
                    req_id_counter += 1
    
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
    
    system_template = """你是"需求澄清智能体"。从验收方视角出发，对需求清单逐条依据"基准 SRS"进行评分。

评分规则为5分制：强采纳 +2；采纳 +1；中性意见 0；不采纳 -1；强不采纳 -2。以基准 SRS 为唯一标尺，锚定术语、量纲、阈值、角色称谓；不得引入新需求、不得改写原文、不得扩展范围。

当输入同时包含：
- 基准 SRS：代码围栏，类型为 text。
- 需求清单：代码围栏，类型为 json，包含 [{"id":..., "text":...}] 数组；

此时进入评分模式，仅输出一个 json 围栏包裹的评分数组，格式如：[{"id":"FR-01","score":1,"reason":"..."}]。

输出要求：
- 与输入顺序完全一致；id 一一对应；
- reason 为中文、≤50字，仅指出审计差异点（术语不一致/阈值冲突/表述含糊/无 SRS 对应等）；
- 不提出建议、不引入修改、不输出任何非 JSON 内容。

若未同时提供需求清单与 SRS，则不进入评分模式，而是提供结构化提示与输入模板。

默认不追问澄清，除非缺少必要输入。输出风格应简洁、严谨、可审计。评分依据"是否与 SRS 保持一致性"为唯一标准。
"""
    
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
    state["atomic_req_scores_count"] = scores_count
    
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
        # 移动到下一个原子需求
        state["current_atomic_req_index"] = atomic_req_index + 1
        state["scores"] = {}  # 清空当前评分
        return "NextAtomicReq"
    
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
        # 移动到下一个原子需求
        state["current_atomic_req_index"] = atomic_req_index + 1
        state["scores"] = {}  # 清空当前评分
        return "NextAtomicReq"
    
    log(f"条件判断：当前原子需求已打分 {current_count} 次（上限 {max_scores}），回到 ReqExplore 继续迭代")
    return "ReqExplore"


def should_process_next_atomic_req(state: GraphState) -> str:
    """判断是否处理下一个原子需求"""
    return has_more_atomic_reqs(state)


def parallel_process_atomic_reqs_node(state: GraphState) -> GraphState:
    """
    并行处理所有原子需求。
    每个原子需求独立进行 ReqExplore <-> ReqClarify 迭代循环。
    """
    atomic_reqs_queue = state.get("atomic_reqs_queue", [])
    if not atomic_reqs_queue:
        log("并行处理：无原子需求需要处理")
        return state
    
    log(f"并行处理：开始处理 {len(atomic_reqs_queue)} 个原子需求")
    
    # 获取并行度（从 state 中读取，如果没有则使用默认值）
    parallel_workers = state.get("parallel_workers")
    workers = get_parallel_workers(default=parallel_workers)
    log(f"并行处理：使用 {workers} 个工作线程")
    
    # 准备共享状态（只读部分）
    user_input = state["user_input"]
    reference_srs = state["reference_srs"]
    max_iterations = state.get("max_iterations", 5)
    iteration = state.get("iteration", 0)
    
    # 用于线程安全的状态合并
    all_reqs: List[Dict[str, Any]] = []
    all_explore_history: Dict[int, List[Dict[str, str]]] = {}
    all_scores_count: Dict[int, int] = {}
    all_req_max_scores: Dict[str, int] = {}
    all_req_scores_history: Dict[str, List[int]] = {}
    all_req_first_iteration: Dict[str, int] = {}
    all_req_last_iteration: Dict[str, int] = {}
    all_req_last_clarify_iteration: Dict[str, int] = {}
    logs_list = state.get("logs", [])
    logs_lock = Lock()
    
    # 并行处理所有原子需求
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(
                process_single_atomic_req,
                idx,
                atomic_req,
                user_input,
                reference_srs,
                max_iterations,
                logs_list,
                logs_lock,
                iteration,
            ): idx
            for idx, atomic_req in enumerate(atomic_reqs_queue)
        }
        
        # 收集结果
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                (normalized_reqs, explore_history, scores_count,
                 req_max_scores, req_scores_history, req_first_iteration,
                 req_last_iteration, req_last_clarify_iteration) = future.result()
                
                # 线程安全地合并结果
                with logs_lock:
                    all_reqs.extend(normalized_reqs)
                    all_explore_history.update(explore_history)
                    all_scores_count.update(scores_count)
                    # 合并评分相关数据
                    all_req_max_scores.update(req_max_scores)
                    for req_id, scores in req_scores_history.items():
                        if req_id not in all_req_scores_history:
                            all_req_scores_history[req_id] = []
                        all_req_scores_history[req_id].extend(scores)
                    all_req_first_iteration.update(req_first_iteration)
                    all_req_last_iteration.update(req_last_iteration)
                    all_req_last_clarify_iteration.update(req_last_clarify_iteration)
                
                log(f"并行处理：原子需求 {idx + 1} 处理完成")
            except Exception as e:
                log(f"并行处理：原子需求 {idx + 1} 处理失败: {str(e)}")
                # 即使失败也继续处理其他原子需求
    
    # 更新状态
    state["req_list"] = all_reqs
    state["atomic_req_explore_history"] = all_explore_history
    state["atomic_req_scores_count"] = all_scores_count
    state["req_max_scores"] = all_req_max_scores
    state["req_scores_history"] = all_req_scores_history
    state["req_first_iteration"] = all_req_first_iteration
    state["req_last_iteration"] = all_req_last_iteration
    state["req_last_clarify_iteration"] = all_req_last_clarify_iteration
    state["logs"] = logs_list
    
    log(f"并行处理：所有原子需求处理完成，共生成 {len(all_reqs)} 条细化需求")
    return state


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
        
        # 如果最高分 >= 0，保留最高分对应的版本
        if max_score is not None and max_score >= 0:
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
    log(f"汇总：评分>=0的需求 {sum(1 for score in req_max_scores.values() if score is not None and score >= 0)} 条")
    return state


def build_graph(ablation_mode: Optional[str] = None):
    graph = StateGraph(GraphState)
    
    # 添加所有节点
    graph.add_node("ReqParse", req_parse_node)
    graph.add_node("ParallelProcessAtomicReqs", parallel_process_atomic_reqs_node)
    graph.add_node("Aggregate", aggregate_node)
    graph.add_node("DocGenerate", doc_generate_node)
    
    # 保留旧节点以支持消融实验模式（如果需要串行处理）
    graph.add_node("ReqExplore", req_explore_node)
    graph.add_node("ReqClarify", req_clarify_node)
    
    graph.set_entry_point("ReqParse")
    
    if ablation_mode == "no-explore-clarify":
        # 模式：移除 ReqExplore + ReqClarify
        # 流程：ReqParse -> Aggregate -> DocGenerate -> END
        log("构建图结构：no-explore-clarify 模式（跳过 ReqExplore 和 ReqClarify）")
        graph.add_edge("ReqParse", "Aggregate")
        graph.add_edge("Aggregate", "DocGenerate")
    elif ablation_mode == "no-clarify":
        # 模式：移除 ReqClarify，但使用并行处理（只并行 ReqExplore）
        # 注意：这个模式可能需要特殊处理，暂时使用串行逻辑
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
        # 默认模式：完整流程（并行处理）
        # 流程：ReqParse -> ParallelProcessAtomicReqs -> Aggregate -> DocGenerate -> END
        log("构建图结构：默认模式（完整流程，并行处理所有原子需求）")
        graph.add_edge("ReqParse", "ParallelProcessAtomicReqs")
        graph.add_edge("ParallelProcessAtomicReqs", "Aggregate")
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
    parallel_workers: Optional[int] = Field(
        default=None, description="并行处理的工作线程数（None 表示使用环境变量或默认值）"
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
        "parallel_workers": demo.parallel_workers,
        "req_max_scores": {},
        "req_scores_history": {},
        "req_first_iteration": {},
        "req_last_iteration": {},
        "req_last_clarify_iteration": {},
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
    # 5) 并行处理工作线程数
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="并行处理的工作线程数（默认使用环境变量 PARALLEL_WORKERS 或默认值 3）",
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
        parallel_workers=args.parallel_workers,
    )
    run_demo(demo)


if __name__ == "__main__":
    main()
