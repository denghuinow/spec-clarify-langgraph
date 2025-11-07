# -*- coding: utf-8 -*-
"""
批量SRS评分脚本
---------------
读取 generated_documents.json，对每个文档调用 srs_evaluation.py 进行评估，
并生成包含 filename、time_cost、srs得分 的 CSV 文件。

使用方法:
    python batch_evaluate_srs.py
    python batch_evaluate_srs.py --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Any, Optional, Tuple

# 导入评估函数
from srs_evaluation import evaluate_srs, read_text_file


def find_standard_srs_file(filename: str, base_dir: Path) -> Optional[Path]:
    """
    根据文件名在基准SRS目录中查找对应的文件
    
    Args:
        filename: 待评估文档的文件名
        base_dir: 基准SRS文件所在的基础目录
        
    Returns:
        找到的文件路径，如果不存在则返回None
    """
    # 基准SRS文件目录
    req_md_dir = base_dir / "req_md"
    
    if not req_md_dir.exists():
        return None
    
    # 直接查找同名文件
    standard_file = req_md_dir / filename
    if standard_file.exists() and standard_file.is_file():
        return standard_file
    
    return None


def process_single_document(
    doc: Dict[str, Any],
    idx: int,
    total: int,
    standard_base_dir: Path,
    verbose: bool,
    print_lock: Lock
) -> Tuple[int, Dict[str, Any]]:
    """
    处理单个文档的评估
    
    Args:
        doc: 文档字典，包含 filename, document, time_cost
        idx: 文档索引（从1开始）
        total: 总文档数
        standard_base_dir: 基准SRS文件的基础目录
        verbose: 是否显示详细进度
        print_lock: 用于线程安全输出的锁
        
    Returns:
        (索引, 结果字典) 或 (索引, None) 如果处理失败
    """
    filename = doc.get("filename", "")
    document_content = doc.get("document", "")
    time_cost = doc.get("time_cost", 0.0)
    
    if not filename:
        if verbose:
            with print_lock:
                print(f"警告 [{idx}/{total}]: 跳过缺少filename的文档", file=sys.stderr, flush=True)
        return (idx, {
            "filename": "N/A",
            "time_cost": time_cost,
            "srs_score": "ERROR: 缺少filename"
        })
    
    if verbose:
        with print_lock:
            print(f"[{idx}/{total}] 处理: {filename}", flush=True)
    
    # 查找基准SRS文件
    standard_file = find_standard_srs_file(filename, standard_base_dir)
    
    if not standard_file:
        if verbose:
            with print_lock:
                print(f"  警告: 未找到基准SRS文件: {filename}", file=sys.stderr, flush=True)
        return (idx, {
            "filename": filename,
            "time_cost": time_cost,
            "srs_score": "ERROR: 基准SRS文件不存在"
        })
    
    # 读取基准SRS文件内容
    try:
        standard_srs_content = read_text_file(str(standard_file))
    except Exception as e:
        if verbose:
            with print_lock:
                print(f"  错误: 读取基准SRS文件失败: {e}", file=sys.stderr, flush=True)
        return (idx, {
            "filename": filename,
            "time_cost": time_cost,
            "srs_score": f"ERROR: 读取基准文件失败 - {str(e)}"
        })
    
    # 调用评估函数
    try:
        if verbose:
            with print_lock:
                print(f"  [{idx}/{total}] 正在评估 {filename}...", flush=True)
        
        eval_start_time = time.time()
        result = evaluate_srs(
            standard_srs=standard_srs_content,
            evaluated_srs=document_content
        )
        eval_time = time.time() - eval_start_time
        
        # 提取SRS得分
        srs_score = None
        if "Comprehensive_Score_Simple" in result:
            srs_score = result["Comprehensive_Score_Simple"]
        elif "Comprehensive_Score_Weighted" in result:
            srs_score = result["Comprehensive_Score_Weighted"]
        elif "error" in result:
            srs_score = f"ERROR: {result.get('error', '评估失败')}"
        else:
            # 尝试从旧格式中提取
            if "Total_Score" in result:
                srs_score = result["Total_Score"]
            else:
                srs_score = "ERROR: 无法提取得分"
        
        if verbose:
            with print_lock:
                if isinstance(srs_score, (int, float)):
                    print(f"  [{idx}/{total}] ✓ {filename}: {srs_score:.4f} (耗时: {eval_time:.2f}秒)", flush=True)
                else:
                    print(f"  [{idx}/{total}] ✗ {filename}: {srs_score} (耗时: {eval_time:.2f}秒)", flush=True)
        
        return (idx, {
            "filename": filename,
            "time_cost": time_cost,
            "srs_score": srs_score if isinstance(srs_score, (int, float)) else str(srs_score)
        })
        
    except Exception as e:
        if verbose:
            with print_lock:
                print(f"  [{idx}/{total}] ✗ 错误: 评估失败 {filename}: {e}", file=sys.stderr, flush=True)
        return (idx, {
            "filename": filename,
            "time_cost": time_cost,
            "srs_score": f"ERROR: {str(e)}"
        })


def process_documents(
    json_file: Path,
    standard_base_dir: Path,
    output_csv: Path,
    verbose: bool = True,
    max_workers: int = 4
) -> None:
    """
    处理所有文档并生成CSV报告（并行处理）
    
    Args:
        json_file: generated_documents.json 文件路径
        standard_base_dir: 基准SRS文件的基础目录（../srs-docs）
        output_csv: 输出CSV文件路径
        verbose: 是否显示详细进度
        max_workers: 并行工作线程数（默认: 4）
    """
    # 读取JSON文件
    if verbose:
        print(f"正在读取 {json_file}...", flush=True)
    
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            documents = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件不存在: {json_file}", file=sys.stderr, flush=True)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    if not isinstance(documents, list):
        print("错误: JSON文件格式不正确，应为数组", file=sys.stderr, flush=True)
        sys.exit(1)
    
    total = len(documents)
    if verbose:
        print(f"找到 {total} 个文档待处理", flush=True)
        print(f"使用 {max_workers} 个并行工作线程\n", flush=True)
    
    # 准备结果存储（按索引排序）
    results_dict = {}
    print_lock = Lock()
    
    # 使用线程池并行处理
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(
                process_single_document,
                doc,
                idx + 1,
                total,
                standard_base_dir,
                verbose,
                print_lock
            ): idx
            for idx, doc in enumerate(documents)
        }
        
        # 收集结果
        completed = 0
        for future in as_completed(future_to_idx):
            try:
                idx, result = future.result()
                results_dict[idx] = result
                completed += 1
                if verbose:
                    with print_lock:
                        print(f"进度: {completed}/{total} 完成", flush=True)
            except Exception as e:
                original_idx = future_to_idx[future]
                if verbose:
                    with print_lock:
                        print(f"  错误: 任务执行异常 [{original_idx + 1}/{total}]: {e}", file=sys.stderr, flush=True)
                results_dict[original_idx + 1] = {
                    "filename": documents[original_idx].get("filename", "N/A"),
                    "time_cost": documents[original_idx].get("time_cost", 0.0),
                    "srs_score": f"ERROR: 任务执行异常 - {str(e)}"
                }
                completed += 1
    
    total_time = time.time() - start_time
    
    # 按索引排序生成CSV行
    csv_rows = [results_dict[i] for i in sorted(results_dict.keys())]
    
    # 统计成功和失败数量
    success_count = sum(1 for row in csv_rows if isinstance(row.get("srs_score"), (int, float)))
    error_count = total - success_count
    
    if verbose:
        print(f"\n总耗时: {total_time:.2f}秒", flush=True)
    
    # 写入CSV文件
    if verbose:
        print(f"\n正在生成CSV文件: {output_csv}...", flush=True)
    
    try:
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["filename", "time_cost", "srs_score"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(csv_rows)
        
        if verbose:
            print(f"CSV文件已生成: {output_csv}", flush=True)
            print(f"\n统计信息:", flush=True)
            print(f"  总计: {total}", flush=True)
            print(f"  成功: {success_count}", flush=True)
            print(f"  失败: {error_count}", flush=True)
    
    except Exception as e:
        print(f"错误: 写入CSV文件失败: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量SRS评分脚本 - 对 generated_documents.json 中的文档进行评估并生成CSV报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认输出文件名
  %(prog)s
  
  # 指定输出CSV文件
  %(prog)s --output results.csv
  
  # 静默模式（不显示详细进度）
  %(prog)s --quiet
  
  # 使用8个并行工作线程
  %(prog)s --workers 8

环境变量:
  OPENAI_API_KEY: OpenAI API密钥（必需）
  OPENAI_MODEL: 使用的模型名称（可选，默认: gpt-4o-mini）
  OPENAI_BASE_URL: API基础URL（可选，用于兼容其他OpenAI兼容的API）
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="generated_documents.json",
        help="输入的JSON文件路径（默认: generated_documents.json）"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="srs_evaluation_results.csv",
        help="输出的CSV文件路径（默认: srs_evaluation_results.csv）"
    )
    
    parser.add_argument(
        "--standard-dir",
        type=str,
        default="../srs-docs",
        help="基准SRS文件的基础目录（默认: ../srs-docs）"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式，不显示详细进度"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="并行工作线程数（默认: 4）"
    )
    
    args = parser.parse_args()
    
    # 检查必要的环境变量
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("必须设置 OPENAI_API_KEY 环境变量")
    
    # 解析路径
    script_dir = Path(__file__).parent.resolve()
    json_file = script_dir / args.input
    output_csv = script_dir / args.output
    standard_base_dir = Path(args.standard_dir).resolve()
    
    # 检查输入文件
    if not json_file.exists():
        parser.error(f"输入文件不存在: {json_file}")
    
    # 检查基准目录
    if not standard_base_dir.exists():
        parser.error(f"基准SRS目录不存在: {standard_base_dir}")
    
    # 处理文档
    try:
        process_documents(
            json_file=json_file,
            standard_base_dir=standard_base_dir,
            output_csv=output_csv,
            verbose=not args.quiet,
            max_workers=args.workers
        )
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n用户中断", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

