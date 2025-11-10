# -*- coding: utf-8 -*-
"""
SRS评分工具
-----------
独立脚本，用于评估SRS文档（使用OpenAI API进行本地评估）

使用方法：
    # 单文件处理
    python srs_evaluation.py --standard-srs standard.md --evaluated-srs evaluated.md
    python srs_evaluation.py --standard-srs standard.md --evaluated-srs evaluated.md --output result.json
    
环境变量：
    OPENAI_API_KEY: OpenAI API密钥（必需）
    OPENAI_EVALUATION_MODEL: 评估使用的模型名称（可选，优先使用，如果未设置则使用 OPENAI_MODEL）
    OPENAI_MODEL: 使用的模型名称（可选，默认: gpt-4o-mini，当 OPENAI_EVALUATION_MODEL 未设置时使用）
    OPENAI_BASE_URL: API基础URL（可选，用于兼容其他OpenAI兼容的API）
    
注意：批量处理请使用 batch_process.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
def read_text_file(file_path: str) -> str:
    """读取 UTF-8 文本文件内容"""
    resolved_path = os.path.expanduser(file_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"文件不存在: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as f:
        return f.read()


def calculate_comprehensive_score(result: Dict[str, Any]) -> Dict[str, float]:
    """
    计算综合评分（两种口径）
    
    Args:
        result: 评估结果字典，应包含 metrics 字段
        
    Returns:
        包含两种综合评分的字典：
        - Comprehensive_Score_Simple: 简单平均（7个指标等权）
        - Comprehensive_Score_Weighted: 加权平均（偏重落地性）
    """
    # 定义7个指标及其权重
    metric_weights = {
        "coverage": 0.20,
        "completeness": 0.20,
        "testability": 0.20,
        "traceability": 0.15,
        "consistency": 0.10,
        "clarity": 0.10,
        "scope_discipline": 0.05
    }
    
    # 从结果中提取 metrics
    metrics = result.get("metrics", {})
    if not metrics:
        return {}
    
    # 提取7个指标值
    metric_values = {}
    missing_metrics = []
    
    for metric_name in metric_weights.keys():
        if metric_name in metrics:
            value = metrics[metric_name]
            if isinstance(value, (int, float)):
                metric_values[metric_name] = float(value)
        else:
            missing_metrics.append(metric_name)
    
    if not metric_values:
        return {}
    
    # 如果有缺失的指标，记录警告但继续计算
    if missing_metrics:
        print(f"警告: 以下指标缺失，将使用可用指标计算: {', '.join(missing_metrics)}", 
              file=sys.stderr, flush=True)
    
    # 计算简单平均（等权）
    simple_avg = sum(metric_values.values()) / len(metric_values)
    
    # 计算加权平均
    weighted_sum = 0.0
    total_weight = 0.0
    
    for metric_name, weight in metric_weights.items():
        if metric_name in metric_values:
            weighted_sum += metric_values[metric_name] * weight
            total_weight += weight
    
    # 如果总权重不为0，计算加权平均；否则使用简单平均
    if total_weight > 0:
        weighted_avg = weighted_sum / total_weight
    else:
        weighted_avg = simple_avg
    
    return {
        "Comprehensive_Score_Simple": round(simple_avg, 4),
        "Comprehensive_Score_Weighted": round(weighted_avg, 4)
    }


def evaluate_srs(
    standard_srs: str,
    evaluated_srs: str,
    model: Optional[str] = None,
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    评估SRS文档（使用OpenAI API）
    
    Args:
        standard_srs: 标准参考SRS文档
        evaluated_srs: 待评估SRS文档
        model: 使用的模型名称（可选，默认从环境变量读取）
        temperature: 模型温度参数（默认: 0.2）
        
    Returns:
        评分结果字典，包含 Functional_Completeness, Interaction_Flow_Similarity, Total_Score, Summary
    """
    # 初始化OpenAI客户端
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 OPENAI_API_KEY 环境变量，请设置后重试")
    
    client_kwargs = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = OpenAI(**client_kwargs)
    # 优先使用 OPENAI_EVALUATION_MODEL，如果未设置则回退到 OPENAI_MODEL
    model = model or os.environ.get("OPENAI_EVALUATION_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    
    user_prompt = f"""You are a strict SRS requirement review scoring assistant.

Task: Using the "reference requirement document" as ground truth, compare it with the "user requirement document", output only structured score JSON, do not output any explanatory text, tables, code blocks, or extra fields.

Scoring Requirements (all are 0~1 floating point numbers, 4 decimal places precision is sufficient):

1. coverage (Coverage)
   Definition: Items in reference requirements that are "fully covered" by user requirements score 1 point, "partially covered" score 0.5 points, "not covered" score 0 points;
   coverage = (fully covered items count + 0.5 × partially covered items count) / total reference items count.

2. completeness (Completeness)
   Dimensions: Whether it covers main primary flows, key exception flows, boundary conditions, data and state, acceptance criteria, compliance/security, etc.
   High score means "covered functions are written in sufficient detail and are usable".

3. consistency (Consistency)
   Dimensions: Whether user requirements are internally self-consistent, whether there are conflicts or self-contradictory descriptions with reference requirements.
   High score means "no or very few obvious conflicts".

4. testability (Testability)
   Dimensions: Whether requirements are verifiable (whether there are clear trigger conditions, expected results, measurable standards, avoiding unverifiable general terms).
   High score means "most items can be used to design test cases".

5. clarity (Clarity)
   Dimensions: Whether there are vague subjective descriptions like "fast", "reasonable", "friendly", "as much as possible", whether there are composite requirements or ambiguities.
   High score means "clear expression, appropriate granularity, few ambiguities".

6. traceability (Traceability)
   Dimensions: Whether user requirements can have clear mapping back to reference requirements (numbering/title/semantics), whether it is easy to build RTM (Requirements Traceability Matrix).
   High score means "most requirements can clearly correspond to sources".

7. scope_discipline (Scope Management)
   Dimensions: Whether it introduces many requirements beyond reference scope causing scope creep; whether there are clear boundaries for new scope.
   High score means "scope converges well, few new points and clearly marked boundaries".

8. by_category (Score by Category)
   - functional: Overall quality of functional requirements (comprehensive performance of the above metrics on the functional requirements subset).
   - non_functional: Quality related to non-functional requirements (performance, security, reliability, availability, etc.).
   - constraints: Coverage and clarity of mandatory items such as constraints/interfaces/compliance.

Output Format (must strictly satisfy):
- Output only one JSON object, prohibit outputting any Markdown, comments, natural language explanations, or code block markers.
- JSON top-level keys only include:
  - "metrics"

Where:
"metrics" : {{
  "coverage": <float>,
  "completeness": <float>,
  "consistency": <float>,
  "testability": <float>,
  "clarity": <float>,
  "traceability": <float>,
  "scope_discipline": <float>,
  "by_category": {{
    "functional": <float>,
    "non_functional": <float>,
    "constraints": <float>
  }}
}}

Constraints:
- Do not output RTM, do not output gaps, do not output rewriting suggestions, do not output Summary, do not add any other fields.
- If information is insufficient, please truthfully reflect uncertainty in the scores (give conservative scores), but still must return all the above fields, prohibit outputting explanatory text.
- Ensure the return value is valid JSON (double quotes, comma positions, boolean/numeric formats are all correct).

[Reference Requirements]
{standard_srs}

[User Requirements]
{evaluated_srs}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )

    content = response.choices[0].message.content.strip()

    # 尝试多种方式解析JSON
    parsed_result = None
    
    # 方法1: 尝试直接解析
    try:
        parsed_result = json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # 方法2: 如果失败，尝试从markdown代码块中提取JSON
    if parsed_result is None:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                parsed_result = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
    
    # 方法3: 如果还是失败，尝试提取第一个{到最后一个}之间的内容
    if parsed_result is None:
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = content[first_brace:last_brace + 1]
            try:
                parsed_result = json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    # 方法4: 尝试查找所有可能的JSON对象
    if parsed_result is None:
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        for match in matches:
            try:
                parsed_result = json.loads(match)
                # 验证是否包含必需的字段
                if all(key in parsed_result for key in ["Functional_Completeness", "Interaction_Flow_Similarity", "Total_Score"]):
                    break
            except json.JSONDecodeError:
                continue
    
    if parsed_result is not None:
        # 验证必需字段（新格式：包含 metrics 字段）
        if "metrics" in parsed_result:
            # 新格式：计算综合评分
            comprehensive_scores = calculate_comprehensive_score(parsed_result)
            if comprehensive_scores:
                parsed_result.update(comprehensive_scores)
        elif not all(key in parsed_result for key in ["Functional_Completeness", "Interaction_Flow_Similarity", "Total_Score"]):
            # 旧格式验证（已弃用，但保留兼容性）
            return {
                "error": "JSON格式不完整，缺少必需字段",
                "raw_output": content[:500] if len(content) > 500 else content
            }
        return parsed_result
    else:
        # 所有解析方法都失败，返回错误信息
        error_preview = content[:500] if len(content) > 500 else content
        return {
            "error": "模型输出的不是有效JSON",
            "raw_output": error_preview,
            "raw_output_length": len(content)
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估SRS文档（使用OpenAI API进行本地评估）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 单文件处理
  %(prog)s --standard-srs standard.md --evaluated-srs evaluated.md
  
  # 单文件处理并保存结果
  %(prog)s --standard-srs standard.md --evaluated-srs evaluated.md --output result.json
  
环境变量：
  OPENAI_API_KEY: OpenAI API密钥（必需）
  OPENAI_EVALUATION_MODEL: 评估使用的模型名称（可选，优先使用，如果未设置则使用 OPENAI_MODEL）
  OPENAI_MODEL: 使用的模型名称（可选，默认: gpt-4o-mini，当 OPENAI_EVALUATION_MODEL 未设置时使用）
  OPENAI_BASE_URL: API基础URL（可选，用于兼容其他OpenAI兼容的API）
  
注意：批量处理请使用 batch_process.py
        """
    )
    
    # 单文件处理参数
    parser.add_argument(
        "--standard-srs", "-s",
        type=str,
        help="标准参考SRS文档文件路径"
    )
    parser.add_argument(
        "--evaluated-srs", "-e",
        type=str,
        help="待评估SRS文档文件路径"
    )
    
    # 直接文本输入
    parser.add_argument(
        "--text-standard",
        type=str,
        help="标准参考SRS文档内容（直接传入）"
    )
    parser.add_argument(
        "--text-evaluated",
        type=str,
        help="待评估SRS文档内容（直接传入）"
    )
    
    # 模型配置
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="使用的模型名称（可选，默认从环境变量 OPENAI_EVALUATION_MODEL 或 OPENAI_MODEL 读取）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="模型温度参数（默认: 0.2）"
    )
    
    # 输出
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出JSON文件路径（可选）"
    )
    
    # 综合评分方法选择
    parser.add_argument(
        "--score-method",
        type=str,
        choices=["simple", "weighted"],
        default="simple",
        help="综合评分显示方式：simple（简单平均，默认）或 weighted（加权平均）。两种评分都会计算并保存。"
    )
    
    args = parser.parse_args()
    
    # 检查必要的环境变量
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("必须设置 OPENAI_API_KEY 环境变量")
    
    # 单文件处理模式
    standard_srs = None
    evaluated_srs = None
    
    # 确定文本来源（优先级：直接文本 > 文件路径）
    if args.text_standard:
        standard_srs = args.text_standard
    elif args.standard_srs:
        standard_srs = read_text_file(args.standard_srs)
    
    if args.text_evaluated:
        evaluated_srs = args.text_evaluated
    elif args.evaluated_srs:
        evaluated_srs = read_text_file(args.evaluated_srs)
    
    if not standard_srs:
        parser.error("必须提供标准参考SRS文档（--standard-srs 或 --text-standard）")
    
    if not evaluated_srs:
        parser.error("必须提供待评估SRS文档（--evaluated-srs 或 --text-evaluated）")
    
    # 调用评分
    try:
        print("正在评估SRS文档...", flush=True)
        result = evaluate_srs(
            standard_srs=standard_srs,
            evaluated_srs=evaluated_srs,
            model=args.model,
            temperature=args.temperature
        )
        
        # 打印结果
        print("\n" + "=" * 60, flush=True)
        print("SRS评分结果", flush=True)
        print("=" * 60, flush=True)
        
        # 显示新格式的指标（如果存在）
        if "metrics" in result:
            metrics = result["metrics"]
            print("\n指标得分:", flush=True)
            metric_names = {
                "coverage": "覆盖率",
                "completeness": "完整度",
                "consistency": "一致性",
                "testability": "可验证性",
                "clarity": "明确性",
                "traceability": "可追溯性",
                "scope_discipline": "范围管理"
            }
            for key, name in metric_names.items():
                if key in metrics:
                    value = metrics[key]
                    print(f"  {name}: {value}", flush=True)
        
        # 显示综合评分
        if "Comprehensive_Score_Simple" in result and "Comprehensive_Score_Weighted" in result:
            print("\n综合评分:", flush=True)
            simple_score = result["Comprehensive_Score_Simple"]
            weighted_score = result["Comprehensive_Score_Weighted"]
            
            if args.score_method == "simple":
                print(f"  简单平均（默认）: {simple_score} ⭐", flush=True)
                print(f"  加权平均: {weighted_score}", flush=True)
            else:
                print(f"  简单平均: {simple_score}", flush=True)
                print(f"  加权平均（默认）: {weighted_score} ⭐", flush=True)
        
        # 显示旧格式的指标（兼容性，已弃用）
        if "Functional_Completeness" in result:
            fc = result["Functional_Completeness"]
            print(f"\n功能完整性得分: {fc.get('score', 'N/A')}", flush=True)
            if "analysis" in fc:
                print(f"  分析: {fc['analysis']}", flush=True)
            elif "reason" in fc:
                print(f"  理由: {fc['reason']}", flush=True)
        
        if "Interaction_Flow_Similarity" in result:
            ifs = result["Interaction_Flow_Similarity"]
            print(f"交互流程相似度得分: {ifs.get('score', 'N/A')}", flush=True)
            if "analysis" in ifs:
                print(f"  分析: {ifs['analysis']}", flush=True)
            elif "reason" in ifs:
                print(f"  理由: {ifs['reason']}", flush=True)
        
        if "Total_Score" in result:
            print(f"总分: {result['Total_Score']}", flush=True)
        
        if "Summary" in result:
            print(f"\n摘要:\n{result['Summary']}", flush=True)
        
        print("=" * 60, flush=True)
        
        # 保存到文件
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}", flush=True)
        else:
            # 如果没有指定输出文件，打印JSON到控制台
            print("\n结果（JSON格式）:", flush=True)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"SRS评分失败: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

