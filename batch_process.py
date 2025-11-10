# -*- coding: utf-8 -*-
"""
批量处理脚本：对 summary_ultra_short 目录中的所有文件执行 srs_generator.py
- 记录完整日志
- 保存最终生成的SRS文档

支持的处理模式：
- full: 完整流程（生成+评估，默认）
- generate_only: 只生成SRS（不评估）
- evaluate_only: 只评估（从 output_dir/srs_output 读取已生成的SRS）

使用示例：
  # 完整流程（默认，使用v2）
  python batch_process.py --user-input-dir ./input --reference-srs-dir ./reference
  
  # 使用v1版本
  python batch_process.py --version v1 --user-input-dir ./input --reference-srs-dir ./reference
  
  # 只生成SRS
  python batch_process.py --mode generate_only --user-input-dir ./input --reference-srs-dir ./reference
  
  # 先批量生成，再批量评估
  python batch_process.py --mode generate_only --user-input-dir ./input --reference-srs-dir ./reference --output-dir ./output1
  python batch_process.py --mode evaluate_only --user-input-dir ./input --reference-srs-dir ./reference --output-dir ./output1
"""

import os
import json
import sys
import time
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import pandas as pd

# 动态导入将在 BatchProcessor.__init__ 中根据 version 参数进行


class BatchProcessor:
    """批量处理器"""
    
    def __init__(
        self,
        user_input_dir: str,
        reference_srs_dir: str,
        output_dir: str,
        max_iterations: int = 5,
        max_files: int = None,
        max_workers: int = 1,
        ablation_mode: Optional[str] = None,
        mode: str = "full",
        generated_srs_dir: Optional[str] = None,
        version: str = "v2"
    ):
        """
        初始化批量处理器
        
        Args:
            user_input_dir: 用户需求文本文件目录
            reference_srs_dir: 参考SRS文本文件目录
            output_dir: 输出目录（保存SRS、日志等）
            max_iterations: 最大迭代轮数
            max_files: 最大处理文件数（None表示处理所有文件）
            max_workers: 最大并发线程数（默认: 1，即串行执行）
            ablation_mode: 消融实验模式（no-clarify 或 no-explore-clarify）
            mode: 处理模式，可选值：
                - full: 完整流程（生成+评估，默认）
                - generate_only: 只生成SRS
                - evaluate_only: 只评估（从 output_dir/srs_output 读取已生成的SRS）
            generated_srs_dir: 已生成的SRS目录（可选，evaluate_only 模式下如果不指定则从 output_dir/srs_output 推导）
            version: 使用的生成器版本，可选值：v1 或 v2（默认: v2）
        """
        self.user_input_dir = Path(user_input_dir)
        self.reference_srs_dir = Path(reference_srs_dir)
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.max_files = max_files
        self.max_workers = max_workers
        self.ablation_mode = ablation_mode
        self.mode = mode
        self.version = version
        
        # evaluate_only 模式下，从 output_dir 推导 generated_srs_dir
        if mode == "evaluate_only":
            if generated_srs_dir:
                self.generated_srs_dir = Path(generated_srs_dir)
            else:
                # 从 output_dir / "srs_output" 推导
                self.generated_srs_dir = self.output_dir / "srs_output"
        else:
            self.generated_srs_dir = Path(generated_srs_dir) if generated_srs_dir else None
        
        # 动态导入对应版本的模块
        if version == "v1":
            from srs_generator_v1 import DemoInput, run_demo, log
        elif version == "v2":
            from srs_generator_v2 import DemoInput, run_demo, log
        else:
            raise ValueError(f"不支持的版本: {version}，必须是 'v1' 或 'v2'")
        
        # 保存导入的函数和类供后续使用
        self.DemoInput = DemoInput
        self.run_demo = run_demo
        self.log = log
        
        # 创建输出目录结构（evaluate_only 模式不需要创建，因为目录已存在）
        if self.mode != "evaluate_only":
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.srs_output_dir = self.output_dir / "srs_output"
            self.log_output_dir = self.output_dir / "logs"
            self.srs_output_dir.mkdir(exist_ok=True)
            self.log_output_dir.mkdir(exist_ok=True)
        else:
            # evaluate_only 模式下，这些目录不会被使用，但仍需要初始化以避免属性错误
            self.srs_output_dir = None
            self.log_output_dir = None
        
        # 评估数据列表（需要线程安全）
        self.evaluation_data: List[Dict[str, Any]] = []
        self.evaluation_data_lock = Lock()
        
        # 生成摘要数据列表（需要线程安全）
        self.generator_summary_data: List[Dict[str, Any]] = []
        self.generator_summary_lock = Lock()
        
        # 评估摘要数据列表（需要线程安全）
        self.evaluation_summary_data: List[Dict[str, Any]] = []
        self.evaluation_summary_lock = Lock()
        
        # 进度跟踪（需要线程安全）
        self.completed_count = 0
        self.completed_lock = Lock()
    
    def get_generator_model(self) -> str:
        """获取生成模型名称"""
        return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    
    def get_evaluation_model(self) -> str:
        """获取评估模型名称（优先使用 OPENAI_EVALUATION_MODEL，如果未设置则回退到 OPENAI_MODEL）"""
        return os.environ.get("OPENAI_EVALUATION_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        
    def get_md_files(self) -> List[Path]:
        """获取用户需求目录中的所有 .md 文件"""
        md_files = list(self.user_input_dir.glob("*.md"))
        md_files.sort()
        return md_files
    
    def find_reference_file(self, user_file: Path) -> Path:
        """根据用户需求文件名找到对应的参考SRS文件"""
        ref_file = self.reference_srs_dir / user_file.name
        if not ref_file.exists():
            raise FileNotFoundError(f"参考SRS文件不存在: {ref_file}")
        return ref_file
    
    def find_generated_srs_file(self, user_file: Path) -> Path:
        """根据用户需求文件名找到对应的已生成SRS文件"""
        if self.generated_srs_dir is None:
            raise ValueError("generated_srs_dir 未设置，无法查找已生成的SRS文件")
        file_basename = user_file.stem
        srs_file = self.generated_srs_dir / f"{file_basename}_srs.md"
        if not srs_file.exists():
            raise FileNotFoundError(f"已生成的SRS文件不存在: {srs_file}")
        return srs_file
    
    def _extract_evaluation_scores(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        从评估结果中提取评分数据（只提取综合评分字段）
        
        Args:
            result: 评估结果字典（可能包含 error 字段）
            
        Returns:
            包含评分数据和错误信息的字典
        """
        scores = {
            "简单平均分": None,
            "加权平均分": None,
            "评估错误": None,
            "错误类型": None
        }
        
        # 检查是否包含错误字段
        if "error" in result or "error_type" in result:
            error_type = result.get("error_type") or result.get("error", "unknown_error")
            error_message = result.get("message", "评估失败")
            scores["评估错误"] = error_message
            scores["错误类型"] = error_type
            return scores
        
        # 新格式：从 metrics 字段提取
        if "metrics" in result:
            # 提取综合评分
            if "Comprehensive_Score_Simple" in result:
                scores["简单平均分"] = result["Comprehensive_Score_Simple"]
            if "Comprehensive_Score_Weighted" in result:
                scores["加权平均分"] = result["Comprehensive_Score_Weighted"]
            
            # 如果 metrics 存在但没有综合评分，尝试计算
            if scores["简单平均分"] is None and scores["加权平均分"] is None:
                try:
                    # 尝试导入 calculate_comprehensive_score 函数
                    from srs_evaluation import calculate_comprehensive_score
                    comprehensive_scores = calculate_comprehensive_score(result)
                    if comprehensive_scores:
                        scores["简单平均分"] = comprehensive_scores.get("Comprehensive_Score_Simple")
                        scores["加权平均分"] = comprehensive_scores.get("Comprehensive_Score_Weighted")
                    else:
                        scores["评估错误"] = "无法计算综合评分：metrics 字段不完整"
                        scores["错误类型"] = "calculation_error"
                except Exception as e:
                    scores["评估错误"] = f"计算综合评分时出错: {str(e)}"
                    scores["错误类型"] = "calculation_error"
        
        # 旧格式：兼容性支持（已弃用）
        elif "Functional_Completeness" in result:
            # 旧格式不包含综合评分字段，尝试从旧字段计算
            # 这里可以尝试计算一个简单的平均分，但为了保持一致性，我们标记为旧格式
            scores["评估错误"] = "评估结果使用旧格式，不包含综合评分字段"
            scores["错误类型"] = "old_format"
        
        # 如果既没有 metrics 也没有 Functional_Completeness，可能是格式错误
        else:
            scores["评估错误"] = "评估结果格式不正确：缺少 metrics 或 Functional_Completeness 字段"
            scores["错误类型"] = "format_error"
        
        return scores
    
    def _get_generation_time_from_log(self, file_basename: str) -> Optional[float]:
        """
        从日志文件中读取生成耗时（排除 ReqClarify 的耗时）
        
        Args:
            file_basename: 文件基础名称（不含扩展名）
            
        Returns:
            生成耗时（秒），如果日志文件不存在或字段不存在，返回 None
        """
        # 在 evaluate_only 模式下，日志文件可能在 generated_srs_dir 的父目录的 logs 子目录中
        # 或者在其他可能的日志目录中
        possible_log_dirs = []
        
        if self.generated_srs_dir is not None:
            # 尝试 generated_srs_dir 的父目录下的 logs 目录
            possible_log_dirs.append(self.generated_srs_dir.parent / "logs")
            # 也尝试 generated_srs_dir 的父目录本身（如果日志文件在那里）
            possible_log_dirs.append(self.generated_srs_dir.parent)
        
        # 也尝试 output_dir 下的 logs 目录（如果存在）
        if self.output_dir.exists():
            possible_log_dirs.append(self.output_dir / "logs")
        
        # 尝试查找日志文件
        for log_dir in possible_log_dirs:
            log_file = log_dir / f"{file_basename}_log.json"
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        log_data = json.load(f)
                    
                    # 优先读取 elapsed_time_seconds 字段（已排除 ReqClarify）
                    if "elapsed_time_seconds" in log_data:
                        return log_data["elapsed_time_seconds"]
                    # 如果没有，尝试从 agent_timings 计算
                    elif "agent_timings" in log_data:
                        agent_timings = log_data["agent_timings"]
                        if isinstance(agent_timings, dict):
                            total_time = sum(agent_timings.values())
                            clarify_time = agent_timings.get("ReqClarify", 0.0)
                            return total_time - clarify_time
                    # 向后兼容：如果有 elapsed_time_seconds_total，使用它
                    elif "elapsed_time_seconds_total" in log_data:
                        total_time = log_data["elapsed_time_seconds_total"]
                        # 尝试从 agent_timings 中减去 ReqClarify 的耗时
                        if "agent_timings" in log_data:
                            agent_timings = log_data["agent_timings"]
                            if isinstance(agent_timings, dict):
                                clarify_time = agent_timings.get("ReqClarify", 0.0)
                                return total_time - clarify_time
                        return total_time
                except (json.JSONDecodeError, IOError) as e:
                    self.log(f"警告: 读取日志文件失败 {log_file}: {str(e)}")
                    continue
        
        # 如果找不到日志文件，返回 None
        return None
    
    def evaluate_srs(
        self, 
        standard_srs: Union[str, Path], 
        evaluated_srs: Union[str, Path], 
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        调用SRS评分脚本评估SRS文档（带重试机制）
        
        Args:
            standard_srs: 标准参考SRS文档（可以是文件路径或内容字符串）
            evaluated_srs: 待评审SRS文档（可以是文件路径或内容字符串）
            max_retries: 最大重试次数（默认3次）
            
        Returns:
            评分结果字典（新格式包含 metrics 和综合评分，旧格式包含 Functional_Completeness 等）
            如果调用失败，返回包含错误信息的字典，而不是 None
        """
        script_path = Path(__file__).parent / "srs_evaluation.py"
        
        # 判断是文件路径还是内容字符串
        def is_file_path(value: Union[str, Path]) -> bool:
            """判断值是否为文件路径"""
            if isinstance(value, Path):
                return value.exists() and value.is_file()
            if isinstance(value, str):
                # 首先检查是否为存在的文件路径
                path = Path(value)
                if path.exists() and path.is_file():
                    return True
                # 如果字符串很长（>1000字符）且包含换行符，很可能是内容而不是路径
                if len(value) > 1000 and '\n' in value:
                    return False
                # 如果字符串很短（<500字符）且看起来像路径（包含路径分隔符或文件扩展名），可能是路径
                if len(value) < 500 and ('/' in value or '\\' in value or value.endswith(('.md', '.txt', '.doc'))):
                    # 但需要确保不是相对路径字符串（如 "../path"）
                    if not value.startswith(('.', '/', '\\')) or path.parent.exists():
                        return False
            return False
        
        # 处理标准SRS：如果是文件路径，直接使用；如果是内容，创建临时文件
        standard_path = None
        standard_temp_file = None
        if is_file_path(standard_srs):
            standard_path = Path(standard_srs) if isinstance(standard_srs, str) else standard_srs
        else:
            # 是内容字符串，创建临时文件
            temp_dir = self.output_dir / "temp_eval_files"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / f"standard_{uuid.uuid4().hex}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(str(standard_srs))
            standard_path = temp_file
            standard_temp_file = temp_file
        
        # 处理待评估SRS：如果是文件路径，直接使用；如果是内容，创建临时文件
        evaluated_path = None
        evaluated_temp_file = None
        if is_file_path(evaluated_srs):
            evaluated_path = Path(evaluated_srs) if isinstance(evaluated_srs, str) else evaluated_srs
        else:
            # 是内容字符串，创建临时文件
            temp_dir = self.output_dir / "temp_eval_files"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / f"evaluated_{uuid.uuid4().hex}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(str(evaluated_srs))
            evaluated_path = temp_file
            evaluated_temp_file = temp_file
        
        # 构建命令，使用文件路径参数
        cmd = [
            sys.executable,
            str(script_path),
            "--standard-srs", str(standard_path),
            "--evaluated-srs", str(evaluated_path)
        ]
        
        last_error = None
        
        # 确保临时文件在函数结束时被清理
        try:
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        # 指数退避：1秒、2秒、4秒
                        wait_time = 2 ** (attempt - 1)
                        self.log(f"重试评估（第 {attempt}/{max_retries - 1} 次），等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    
                    if result.returncode != 0:
                        error_info = {
                            "error": "execution_error",
                            "error_type": "execution_error",
                            "returncode": result.returncode,
                            "stderr": result.stderr[:500] if result.stderr else "",
                            "stdout_preview": result.stdout[:500] if result.stdout else "",
                            "message": f"SRS评分脚本执行失败，返回码: {result.returncode}"
                        }
                        last_error = error_info
                        self.log(f"警告: SRS评分脚本执行失败（尝试 {attempt + 1}/{max_retries}）: 返回码={result.returncode}, stderr={result.stderr[:200]}")
                        
                        # 执行失败通常可以重试
                        if attempt < max_retries - 1:
                            continue
                        else:
                            return error_info
                    
                    # 从输出中提取JSON结果
                    output_lines = result.stdout.strip().split('\n')
                    json_start = -1
                    for i, line in enumerate(output_lines):
                        if line.strip().startswith('{') or line.strip().startswith('['):
                            json_start = i
                            break
                    
                    if json_start >= 0:
                        json_text = '\n'.join(output_lines[json_start:])
                        try:
                            parsed_result = json.loads(json_text)
                            # 检查是否包含错误字段
                            if "error" in parsed_result:
                                error_info = {
                                    "error": "parse_error",
                                    "error_type": "parse_error",
                                    "message": parsed_result.get("error", "评估结果包含错误字段"),
                                    "raw_output_preview": parsed_result.get("raw_output", "")[:500]
                                }
                                last_error = error_info
                                self.log(f"警告: 评估结果包含错误（尝试 {attempt + 1}/{max_retries}）: {error_info['message']}")
                                
                                # 解析错误可以重试
                                if attempt < max_retries - 1:
                                    continue
                                else:
                                    return error_info
                            
                            # 成功解析，返回结果
                            if attempt > 0:
                                self.log(f"✓ 评估成功（经过 {attempt} 次重试）")
                            return parsed_result
                        except json.JSONDecodeError as e:
                            error_info = {
                                "error": "parse_error",
                                "error_type": "parse_error",
                                "message": f"JSON解析失败: {str(e)}",
                                "stdout_preview": result.stdout[:500] if result.stdout else "",
                                "json_text_preview": json_text[:500] if json_start >= 0 else ""
                            }
                            last_error = error_info
                            self.log(f"警告: JSON解析失败（尝试 {attempt + 1}/{max_retries}）: {str(e)}")
                            
                            # JSON解析失败可以重试
                            if attempt < max_retries - 1:
                                continue
                            else:
                                return error_info
                    
                    # 如果无法从输出中提取JSON
                    error_info = {
                        "error": "parse_error",
                        "error_type": "parse_error",
                        "message": "无法从输出中提取JSON结果",
                        "stdout_preview": result.stdout[:500] if result.stdout else ""
                    }
                    last_error = error_info
                    self.log(f"警告: 无法从SRS评分脚本输出中解析JSON结果（尝试 {attempt + 1}/{max_retries}）")
                    
                    # 无法提取JSON可以重试
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return error_info
                        
                except subprocess.TimeoutExpired:
                    error_info = {
                        "error": "timeout",
                        "error_type": "timeout",
                        "message": "SRS评分脚本执行超时（600秒）"
                    }
                    last_error = error_info
                    self.log(f"警告: SRS评分脚本执行超时（尝试 {attempt + 1}/{max_retries}）")
                    
                    # 超时可以重试
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return error_info
                        
                except Exception as e:
                    error_info = {
                        "error": "exception",
                        "error_type": "exception",
                        "message": f"SRS评分脚本调用异常: {str(e)}",
                        "exception_type": type(e).__name__
                    }
                    last_error = error_info
                    self.log(f"警告: SRS评分脚本调用失败（尝试 {attempt + 1}/{max_retries}）: {str(e)}")
                    
                    # 异常可以重试
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return error_info
            
            # 所有重试都失败，返回最后一次错误
            if last_error:
                return last_error
            else:
                return {
                    "error": "unknown_error",
                    "error_type": "unknown_error",
                    "message": "评估失败，原因未知"
                }
        finally:
            # 清理临时文件
            if standard_temp_file and standard_temp_file.exists():
                try:
                    standard_temp_file.unlink()
                except Exception as e:
                    self.log(f"警告: 清理临时文件失败 {standard_temp_file}: {str(e)}")
            if evaluated_temp_file and evaluated_temp_file.exists():
                try:
                    evaluated_temp_file.unlink()
                except Exception as e:
                    self.log(f"警告: 清理临时文件失败 {evaluated_temp_file}: {str(e)}")
    
    def process_evaluation_only(self, user_file: Path) -> Dict[str, Any]:
        """
        处理只评估模式：读取已生成的SRS，只执行评估
        
        Returns:
            包含处理结果的字典
        """
        file_basename = user_file.stem
        self.log(f"\n{'='*80}")
        self.log(f"处理文件（只评估）: {file_basename}")
        self.log(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            # 验证文件存在性
            missing_files = []
            if not user_file.exists():
                missing_files.append(f"用户需求文件: {user_file}")
            
            ref_file = self.find_reference_file(user_file)
            if not ref_file.exists():
                missing_files.append(f"参考SRS文件: {ref_file}")
            
            generated_srs_file = self.find_generated_srs_file(user_file)
            if not generated_srs_file.exists():
                missing_files.append(f"已生成SRS文件: {generated_srs_file}")
            
            if missing_files:
                error_msg = f"缺少必要文件: {', '.join(missing_files)}"
                self.log(f"✗ {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # 读取文件
            with open(user_file, "r", encoding="utf-8") as f:
                user_input = f.read()
            with open(ref_file, "r", encoding="utf-8") as f:
                reference_srs = f.read()
            with open(generated_srs_file, "r", encoding="utf-8") as f:
                generated_srs = f.read()
            
            self.log(f"用户需求文件: {user_file}")
            self.log(f"参考SRS文件: {ref_file}")
            self.log(f"已生成SRS文件: {generated_srs_file}")
            
            # 调用SRS评分（直接传递文件路径，避免命令行参数过长）
            srs_evaluation_result = None
            evaluation_start_time = time.time()
            try:
                self.log(f"正在调用SRS评分...")
                srs_evaluation_result = self.evaluate_srs(
                    standard_srs=ref_file,
                    evaluated_srs=generated_srs_file
                )
                if srs_evaluation_result:
                    # 提取评分数据
                    scores = self._extract_evaluation_scores(srs_evaluation_result)
                    # 使用加权平均或简单平均作为总分显示
                    total_score = scores.get("加权平均分") or scores.get("简单平均分") or "N/A"
                    error_info = scores.get("评估错误")
                    if error_info:
                        self.log(f"警告: SRS评分包含错误: {error_info}")
                    else:
                        self.log(f"SRS评分完成: 加权平均={scores.get('加权平均分', 'N/A')}, 简单平均={scores.get('简单平均分', 'N/A')}")
                else:
                    self.log(f"警告: SRS评分调用失败或返回空结果")
            except Exception as e:
                self.log(f"警告: SRS评分调用异常: {str(e)}")
                srs_evaluation_result = {
                    "error": "exception",
                    "error_type": "exception",
                    "message": f"SRS评分调用异常: {str(e)}"
                }
            finally:
                # 计算评估耗时
                evaluation_elapsed_time = time.time() - evaluation_start_time
            
            elapsed_time = time.time() - start_time
            
            # 从日志文件读取生成耗时
            generation_time = self._get_generation_time_from_log(file_basename)
            
            # 收集数据
            evaluation_entry = {
                "文件名": file_basename,
                "迭代轮数": None,
                "srs生成耗时": round(generation_time, 2) if generation_time is not None else None,
            }
            
            # 添加SRS评分数据（包含错误信息）
            if srs_evaluation_result:
                scores = self._extract_evaluation_scores(srs_evaluation_result)
                evaluation_entry.update(scores)
            else:
                evaluation_entry["简单平均分"] = None
                evaluation_entry["加权平均分"] = None
                evaluation_entry["评估错误"] = "评估结果为空"
                evaluation_entry["错误类型"] = "empty_result"
            
            with self.evaluation_data_lock:
                self.evaluation_data.append(evaluation_entry)
            
            # 收集评估摘要数据
            evaluation_summary_entry = {
                "文件": file_basename,
                "使用模型": self.get_evaluation_model(),
                "评估耗时(秒)": round(evaluation_elapsed_time, 2),
                "简单平均分": None,
                "加权平均分": None
            }
            
            if srs_evaluation_result:
                scores = self._extract_evaluation_scores(srs_evaluation_result)
                evaluation_summary_entry["简单平均分"] = scores.get("简单平均分")
                evaluation_summary_entry["加权平均分"] = scores.get("加权平均分")
                if scores.get("评估错误"):
                    evaluation_summary_entry["评估错误"] = scores.get("评估错误")
                    evaluation_summary_entry["错误类型"] = scores.get("错误类型")
            
            with self.evaluation_summary_lock:
                self.evaluation_summary_data.append(evaluation_summary_entry)
            
            self.log(f"✓ 文件 {file_basename} 评估完成，耗时: {evaluation_elapsed_time:.2f} 秒")
            return {
                "status": "success",
                "file": file_basename,
                "evaluation_entry": evaluation_entry
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"处理文件 {file_basename} 时出错: {str(e)}"
            self.log(f"✗ {error_msg}")
            
            # 从日志文件读取生成耗时（如果可能）
            generation_time = self._get_generation_time_from_log(file_basename)
            
            # 判断错误类型
            error_type = "file_not_found" if isinstance(e, FileNotFoundError) else "other_error"
            
            evaluation_entry = {
                "文件名": file_basename,
                "迭代轮数": None,
                "srs生成耗时": round(generation_time, 2) if generation_time is not None else None,
                "简单平均分": None,
                "加权平均分": None,
                "评估错误": str(e),
                "错误类型": error_type,
                "错误": str(e)
            }
            with self.evaluation_data_lock:
                self.evaluation_data.append(evaluation_entry)
            
            # 收集错误情况的评估摘要数据
            evaluation_summary_entry = {
                "文件": file_basename,
                "使用模型": self.get_evaluation_model(),
                "评估耗时(秒)": round(elapsed_time, 2),
                "简单平均分": None,
                "加权平均分": None,
                "评估错误": str(e),
                "错误类型": error_type,
                "错误": str(e)
            }
            
            with self.evaluation_summary_lock:
                self.evaluation_summary_data.append(evaluation_summary_entry)
            
            return {
                "status": "error",
                "file": file_basename,
                "error": str(e),
                "evaluation_entry": evaluation_entry
            }
    
    def process_file(self, user_file: Path) -> Dict[str, Any]:
        """
        处理单个文件（根据mode参数调用不同的处理方法）
        
        Returns:
            包含处理结果的字典
        """
        # 根据模式调用不同的处理方法
        if self.mode == "evaluate_only":
            return self.process_evaluation_only(user_file)
        elif self.mode == "generate_only":
            return self.process_generate_only(user_file)
        else:  # mode == "full" (默认)
            return self.process_full(user_file)
    
    def process_generate_only(self, user_file: Path) -> Dict[str, Any]:
        """
        处理只生成模式：只生成SRS，不评估、不计算相似度
        
        Returns:
            包含处理结果的字典
        """
        file_basename = user_file.stem
        self.log(f"\n{'='*80}")
        self.log(f"处理文件（只生成）: {file_basename}")
        self.log(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            # 读取文件
            ref_file = self.find_reference_file(user_file)
            with open(user_file, "r", encoding="utf-8") as f:
                user_input = f.read()
            with open(ref_file, "r", encoding="utf-8") as f:
                reference_srs = f.read()
            
            self.log(f"用户需求文件: {user_file}")
            self.log(f"参考SRS文件: {ref_file}")
            self.log(f"用户需求长度: {len(user_input)} 字符")
            self.log(f"参考SRS长度: {len(reference_srs)} 字符")
            
            # 创建DemoInput
            demo = self.DemoInput(
                user_input=user_input,
                reference_srs=reference_srs,
                max_iterations=self.max_iterations,
                ablation_mode=self.ablation_mode
            )
            
            # 运行演示（带重试机制）
            max_retries = 5
            final_state = None
            errors: List[str] = []
            
            for attempt in range(max_retries):
                try:
                    final_state = self.run_demo(demo, silent=True)
                    if attempt > 0:
                        self.log(f"文件处理重试成功（第 {attempt + 1} 次尝试）")
                    break
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = f"第 {attempt + 1} 次尝试失败（错误类型: {error_type}）: {str(e)}"
                    errors.append(error_msg)
                    
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        self.log(
                            f"文件处理失败，{error_msg}，等待 {wait_time} 秒后重试（剩余 {max_retries - attempt - 1} 次）"
                        )
                        time.sleep(wait_time)
                    else:
                        self.log(f"文件处理失败，{error_msg}，已达到最大重试次数")
                        all_errors = "\n".join(errors)
                        raise Exception(
                            f"文件处理失败，已重试 {max_retries} 次均失败。\n"
                            f"所有尝试的错误信息：\n{all_errors}"
                        ) from e
            
            if final_state is None:
                raise Exception("文件处理失败：未获得最终状态")
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            
            # 从 final_state 获取 agent_timings，计算排除 ReqClarify 的总耗时
            agent_timings = final_state.get("agent_timings", {})
            elapsed_time_excluding_clarify = elapsed_time
            if agent_timings:
                # 计算所有智能体的总耗时
                total_agent_time = sum(agent_timings.values())
                # 排除 ReqClarify 的耗时
                clarify_time = agent_timings.get("ReqClarify", 0.0)
                elapsed_time_excluding_clarify = total_agent_time - clarify_time
                # 如果计算出的排除耗时小于0或大于总耗时，使用总耗时作为兜底
                if elapsed_time_excluding_clarify < 0:
                    elapsed_time_excluding_clarify = elapsed_time
                elif elapsed_time_excluding_clarify > elapsed_time:
                    elapsed_time_excluding_clarify = elapsed_time - clarify_time
            else:
                # 如果没有 agent_timings，使用总耗时（向后兼容）
                elapsed_time_excluding_clarify = elapsed_time
            
            # 保存SRS文档
            srs_output_file = self.srs_output_dir / f"{file_basename}_srs.md"
            with open(srs_output_file, "w", encoding="utf-8") as f:
                f.write(final_state["srs_output"])
            self.log(f"已保存SRS文档: {srs_output_file}")
            
            # 保存完整日志（JSON格式）
            log_output_file = self.log_output_dir / f"{file_basename}_log.json"
            # 处理版本差异：v1有frozen_ids/removed_ids，v2有banned_ids
            frozen_ids = final_state.get("frozen_ids", [])
            removed_ids = final_state.get("removed_ids", [])
            banned_ids = final_state.get("banned_ids", [])
            log_data = {
                "file": file_basename,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "user_input_file": str(user_file),
                "reference_srs_file": str(ref_file),
                "user_input": user_input,
                "reference_srs": reference_srs,
                "max_iterations": self.max_iterations,
                "ablation_mode": self.ablation_mode,
                "final_iteration": final_state["iteration"],
                "elapsed_time_seconds": elapsed_time_excluding_clarify,  # 使用排除 ReqClarify 的耗时
                "elapsed_time_seconds_total": elapsed_time,  # 保存总耗时
                "agent_timings": agent_timings,  # 保存各节点详细耗时
                "req_list": final_state["req_list"],
                "frozen_ids": frozen_ids,
                "removed_ids": removed_ids,
                "banned_ids": banned_ids,
                "scores": final_state.get("scores", {}),
                "logs": final_state["logs"],
                "srs_output": final_state["srs_output"],
                "srs_evaluation": None
            }
            with open(log_output_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            self.log(f"已保存日志: {log_output_file}")
            
            # 收集生成摘要数据
            # 获取ReqExplore迭代次数（v2使用global_iteration，v1使用iteration）
            if self.version == "v2":
                req_explore_iterations = final_state.get("global_iteration", 0)
            else:
                req_explore_iterations = final_state.get("iteration", 0)
            
            generator_summary_entry = {
                "文件": file_basename,
                "状态": "success",
                "生成耗时(秒)": round(elapsed_time_excluding_clarify, 2),  # 使用排除 ReqClarify 的耗时
                "使用模型": self.get_generator_model(),
                "ReqExplore迭代次数": req_explore_iterations,
                "消融实验模式": self.ablation_mode
            }
            
            with self.generator_summary_lock:
                self.generator_summary_data.append(generator_summary_entry)
            
            self.log(f"✓ 文件 {file_basename} 生成完成，耗时: {elapsed_time_excluding_clarify:.2f} 秒（排除 ReqClarify，总耗时: {elapsed_time:.2f} 秒）")
            return {
                "status": "success",
                "file": file_basename,
                "evaluation_entry": None
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"处理文件 {file_basename} 时出错: {str(e)}"
            self.log(f"✗ {error_msg}")
            
            # 收集错误情况的生成摘要数据
            generator_summary_entry = {
                "文件": file_basename,
                "状态": "error",
                "生成耗时(秒)": round(elapsed_time, 2),  # 错误情况下使用总耗时
                "使用模型": self.get_generator_model(),
                "ReqExplore迭代次数": None,
                "消融实验模式": self.ablation_mode,
                "错误": str(e)
            }
            
            with self.generator_summary_lock:
                self.generator_summary_data.append(generator_summary_entry)
            
            return {
                "status": "error",
                "file": file_basename,
                "error": str(e),
                "evaluation_entry": None
            }
    
    def process_full(self, user_file: Path) -> Dict[str, Any]:
        """
        处理完整流程模式：生成+评估
        
        Returns:
            包含处理结果的字典
        """
        file_basename = user_file.stem
        self.log(f"\n{'='*80}")
        self.log(f"处理文件: {file_basename}")
        self.log(f"{'='*80}\n")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 读取文件
            ref_file = self.find_reference_file(user_file)
            with open(user_file, "r", encoding="utf-8") as f:
                user_input = f.read()
            with open(ref_file, "r", encoding="utf-8") as f:
                reference_srs = f.read()
            
            self.log(f"用户需求文件: {user_file}")
            self.log(f"参考SRS文件: {ref_file}")
            self.log(f"用户需求长度: {len(user_input)} 字符")
            self.log(f"参考SRS长度: {len(reference_srs)} 字符")
            
            # 创建DemoInput
            demo = self.DemoInput(
                user_input=user_input,
                reference_srs=reference_srs,
                max_iterations=self.max_iterations,
                ablation_mode=self.ablation_mode
            )
            
            # 运行演示（带重试机制）
            max_retries = 5
            final_state = None
            errors: List[str] = []
            
            for attempt in range(max_retries):
                try:
                    final_state = self.run_demo(demo, silent=True)
                    if attempt > 0:
                        self.log(f"文件处理重试成功（第 {attempt + 1} 次尝试）")
                    break
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = f"第 {attempt + 1} 次尝试失败（错误类型: {error_type}）: {str(e)}"
                    errors.append(error_msg)
                    
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        self.log(
                            f"文件处理失败，{error_msg}，等待 {wait_time} 秒后重试（剩余 {max_retries - attempt - 1} 次）"
                        )
                        time.sleep(wait_time)
                    else:
                        self.log(f"文件处理失败，{error_msg}，已达到最大重试次数")
                        all_errors = "\n".join(errors)
                        raise Exception(
                            f"文件处理失败，已重试 {max_retries} 次均失败。\n"
                            f"所有尝试的错误信息：\n{all_errors}"
                        ) from e
            
            if final_state is None:
                raise Exception("文件处理失败：未获得最终状态")
            
            # 计算生成耗时
            generation_elapsed_time = time.time() - start_time
            
            # 保存SRS文档
            srs_output_file = self.srs_output_dir / f"{file_basename}_srs.md"
            with open(srs_output_file, "w", encoding="utf-8") as f:
                f.write(final_state["srs_output"])
            self.log(f"已保存SRS文档: {srs_output_file}")
            
            # 收集生成摘要数据
            # 获取ReqExplore迭代次数（v2使用global_iteration，v1使用iteration）
            if self.version == "v2":
                req_explore_iterations = final_state.get("global_iteration", 0)
            else:
                req_explore_iterations = final_state.get("iteration", 0)
            
            generator_summary_entry = {
                "文件": file_basename,
                "状态": "success",
                "生成耗时(秒)": round(generation_elapsed_time, 2),
                "使用模型": self.get_generator_model(),
                "ReqExplore迭代次数": req_explore_iterations,
                "消融实验模式": self.ablation_mode
            }
            
            with self.generator_summary_lock:
                self.generator_summary_data.append(generator_summary_entry)
            
            # 验证生成的SRS文件已成功保存
            if not srs_output_file.exists():
                raise FileNotFoundError(f"生成的SRS文件未成功保存: {srs_output_file}")
            
            # 调用SRS评分（直接传递文件路径，避免命令行参数过长）
            srs_evaluation_result = None
            evaluation_start_time = time.time()
            try:
                self.log(f"正在调用SRS评分...")
                srs_evaluation_result = self.evaluate_srs(
                    standard_srs=ref_file,
                    evaluated_srs=srs_output_file
                )
                if srs_evaluation_result:
                    # 提取评分数据
                    scores = self._extract_evaluation_scores(srs_evaluation_result)
                    # 使用加权平均或简单平均作为总分显示
                    total_score = scores.get("加权平均分") or scores.get("简单平均分") or "N/A"
                    error_info = scores.get("评估错误")
                    if error_info:
                        self.log(f"警告: SRS评分包含错误: {error_info}")
                    else:
                        self.log(f"SRS评分完成: 加权平均={scores.get('加权平均分', 'N/A')}, 简单平均={scores.get('简单平均分', 'N/A')}")
                else:
                    self.log(f"警告: SRS评分调用失败或返回空结果")
                    srs_evaluation_result = {
                        "error": "empty_result",
                        "error_type": "empty_result",
                        "message": "评估结果为空"
                    }
            except Exception as e:
                self.log(f"警告: SRS评分调用异常: {str(e)}")
                srs_evaluation_result = {
                    "error": "exception",
                    "error_type": "exception",
                    "message": f"SRS评分调用异常: {str(e)}"
                }
            finally:
                # 计算评估耗时
                evaluation_elapsed_time = time.time() - evaluation_start_time
                
                # 收集评估摘要数据
                evaluation_summary_entry = {
                    "文件": file_basename,
                    "使用模型": self.get_evaluation_model(),
                    "评估耗时(秒)": round(evaluation_elapsed_time, 2),
                    "简单平均分": None,
                    "加权平均分": None
                }
                
                if srs_evaluation_result:
                    scores = self._extract_evaluation_scores(srs_evaluation_result)
                    evaluation_summary_entry["简单平均分"] = scores.get("简单平均分")
                    evaluation_summary_entry["加权平均分"] = scores.get("加权平均分")
                    if scores.get("评估错误"):
                        evaluation_summary_entry["评估错误"] = scores.get("评估错误")
                        evaluation_summary_entry["错误类型"] = scores.get("错误类型")
                
                with self.evaluation_summary_lock:
                    self.evaluation_summary_data.append(evaluation_summary_entry)
            
            # 保存完整日志（JSON格式）
            log_output_file = self.log_output_dir / f"{file_basename}_log.json"
            # 处理版本差异：v1有frozen_ids/removed_ids，v2有banned_ids
            frozen_ids = final_state.get("frozen_ids", [])
            removed_ids = final_state.get("removed_ids", [])
            banned_ids = final_state.get("banned_ids", [])
            log_data = {
                "file": file_basename,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "user_input_file": str(user_file),
                "reference_srs_file": str(ref_file),
                "user_input": user_input,
                "reference_srs": reference_srs,
                "max_iterations": self.max_iterations,
                "ablation_mode": self.ablation_mode,
                "final_iteration": final_state["iteration"],
                "elapsed_time_seconds": generation_elapsed_time + evaluation_elapsed_time,
                "req_list": final_state["req_list"],
                "frozen_ids": frozen_ids,
                "removed_ids": removed_ids,
                "banned_ids": banned_ids,
                "scores": final_state.get("scores", {}),
                "logs": final_state["logs"],
                "srs_output": final_state["srs_output"],
                "srs_evaluation": srs_evaluation_result
            }
            with open(log_output_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            self.log(f"已保存日志: {log_output_file}")
            
            # 收集评估数据（线程安全）
            evaluation_entry = {
                "文件名": file_basename,
                "迭代轮数": final_state["iteration"],
                "srs生成耗时": round(generation_elapsed_time, 2),
            }
            
            # 添加SRS评分数据（包含错误信息）
            if srs_evaluation_result:
                scores = self._extract_evaluation_scores(srs_evaluation_result)
                evaluation_entry.update(scores)
            else:
                evaluation_entry["简单平均分"] = None
                evaluation_entry["加权平均分"] = None
                evaluation_entry["评估错误"] = "评估结果为空"
                evaluation_entry["错误类型"] = "empty_result"
            with self.evaluation_data_lock:
                self.evaluation_data.append(evaluation_entry)
            
            total_elapsed_time = generation_elapsed_time + evaluation_elapsed_time
            self.log(f"✓ 文件 {file_basename} 处理完成，总耗时: {total_elapsed_time:.2f} 秒（生成: {generation_elapsed_time:.2f}秒, 评估: {evaluation_elapsed_time:.2f}秒）")
            return {
                "status": "success",
                "file": file_basename,
                "evaluation_entry": evaluation_entry
            }
            
        except Exception as e:
            # 计算耗时（即使出错也记录）
            elapsed_time = time.time() - start_time
            error_msg = f"处理文件 {file_basename} 时出错: {str(e)}"
            self.log(f"✗ {error_msg}")
            
            # 收集错误情况的生成摘要数据
            generator_summary_entry = {
                "文件": file_basename,
                "状态": "error",
                "生成耗时(秒)": round(elapsed_time, 2),
                "使用模型": self.get_generator_model(),
                "ReqExplore迭代次数": None,
                "消融实验模式": self.ablation_mode,
                "错误": str(e)
            }
            
            with self.generator_summary_lock:
                self.generator_summary_data.append(generator_summary_entry)
            
            # 判断错误类型
            error_type = "file_not_found" if isinstance(e, FileNotFoundError) else "other_error"
            
            # 记录错误信息（线程安全）
            evaluation_entry = {
                "文件名": file_basename,
                "迭代轮数": None,
                "srs生成耗时": None,
                "简单平均分": None,
                "加权平均分": None,
                "评估错误": str(e),
                "错误类型": error_type,
                "错误": str(e)
            }
            with self.evaluation_data_lock:
                self.evaluation_data.append(evaluation_entry)
            
            return {
                "status": "error",
                "file": file_basename,
                "error": str(e),
                "evaluation_entry": evaluation_entry
            }
    
    def generate_evaluation_table(self) -> None:
        """生成评估数据表格"""
        if not self.evaluation_data:
            self.log("警告: 没有评估数据可生成表格")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.evaluation_data)
        
        # 统一保存到 output_dir
        csv_dir = self.output_dir
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为CSV
        csv_file = csv_dir / "evaluation_table.csv"
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        self.log(f"已保存评估表格(CSV): {csv_file}")
        
        # 打印统计摘要
        self.log("\n" + "="*80)
        self.log("评估数据统计摘要")
        self.log("="*80)
        
        # 计算平均值（仅对数值列）
        numeric_cols = [
            "srs生成耗时", 
            "简单平均分", "加权平均分"
        ]
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    if col == "srs生成耗时":
                        self.log(f"{col}: 平均值={mean_val:.2f}秒, 标准差={std_val:.2f}秒, 样本数={len(values)}, 总耗时={values.sum():.2f}秒")
                    elif col in ["简单平均分", "加权平均分"]:
                        self.log(f"{col}: 平均值={mean_val:.4f}, 标准差={std_val:.4f}, 样本数={len(values)}")
        
        self.log(f"总文件数: {len(df)}")
        success_count = len(df[df["迭代轮数"].notna()])
        error_count = len(df) - success_count
        self.log(f"成功处理: {success_count}, 失败: {error_count}")
        
        # 统计评估成功和失败
        if "简单平均分" in df.columns:
            evaluation_success = len(df[df["简单平均分"].notna()])
            evaluation_failed = len(df) - evaluation_success
            self.log(f"评估成功: {evaluation_success}, 评估失败: {evaluation_failed}")
        
        # 统计错误类型分布
        if "错误类型" in df.columns:
            error_types = df["错误类型"].value_counts()
            if len(error_types) > 0:
                self.log("\n错误类型分布:")
                for error_type, count in error_types.items():
                    if pd.notna(error_type):
                        self.log(f"  {error_type}: {count}")
        
        # 统计评估错误类型分布
        if "评估错误" in df.columns:
            eval_errors = df[df["评估错误"].notna()]
            if len(eval_errors) > 0:
                self.log("\n评估错误统计:")
                eval_error_types = eval_errors["错误类型"].value_counts()
                for error_type, count in eval_error_types.items():
                    if pd.notna(error_type):
                        self.log(f"  {error_type}: {count}")
                
                # 显示最常见的评估错误消息
                if len(eval_errors) > 0:
                    common_errors = eval_errors["评估错误"].value_counts().head(5)
                    if len(common_errors) > 0:
                        self.log("\n最常见的评估错误消息（前5个）:")
                        for error_msg, count in common_errors.items():
                            if pd.notna(error_msg):
                                # 截断过长的错误消息
                                msg_preview = error_msg[:100] + "..." if len(str(error_msg)) > 100 else error_msg
                                self.log(f"  [{count}次] {msg_preview}")
        
        self.log("="*80 + "\n")
        
        # 打印表格预览
        print("\n评估数据表格预览（前10行）:")
        print(df.head(10).to_string())
    
    def run(self) -> None:
        """运行批量处理"""
        self.log(f"\n{'='*80}")
        self.log("批量处理开始")
        self.log(f"处理模式: {self.mode}")
        self.log(f"生成器版本: {self.version}")
        self.log(f"用户需求目录: {self.user_input_dir}")
        self.log(f"参考SRS目录: {self.reference_srs_dir}")
        self.log(f"输出目录: {self.output_dir}")
        if self.generated_srs_dir is not None:
            self.log(f"已生成SRS目录: {self.generated_srs_dir}")
        self.log(f"最大迭代轮数: {self.max_iterations}")
        if self.ablation_mode is not None:
            self.log(f"消融实验模式: {self.ablation_mode}")
        if self.max_files is not None:
            self.log(f"最大处理文件数: {self.max_files}")
        self.log(f"并发线程数: {self.max_workers}")
        self.log(f"{'='*80}\n")
        
        # 获取所有文件
        md_files = self.get_md_files()
        total_files = len(md_files)
        
        # 如果设置了 max_files，只处理前 N 个文件
        if self.max_files is not None and self.max_files > 0:
            md_files = md_files[:self.max_files]
            self.log(f"找到 {total_files} 个文件，将处理前 {len(md_files)} 个文件\n")
        else:
            self.log(f"找到 {total_files} 个文件需要处理\n")
        
        if len(md_files) == 0:
            self.log("警告: 没有找到任何 .md 文件")
            return
        
        # 重置进度计数器
        self.completed_count = 0
        
        # 处理每个文件（支持并发）
        results = []
        total_files_count = len(md_files)
        
        if self.max_workers > 1:
            # 并发执行
            self.log(f"使用 {self.max_workers} 个线程并发处理 {total_files_count} 个文件\n")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_file = {
                    executor.submit(self.process_file, user_file): user_file 
                    for user_file in md_files
                }
                
                # 收集结果
                for future in as_completed(future_to_file):
                    user_file = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # 更新进度（线程安全）
                        with self.completed_lock:
                            self.completed_count += 1
                            self.log(f"进度: {self.completed_count}/{total_files_count} - {result['file']}")
                    except Exception as e:
                        file_basename = user_file.stem
                        error_result = {
                            "status": "error",
                            "file": file_basename,
                            "error": f"执行异常: {str(e)}"
                        }
                        results.append(error_result)
                        
                        with self.completed_lock:
                            self.completed_count += 1
                            self.log(f"进度: {self.completed_count}/{total_files_count} - {file_basename} (错误)")
        else:
            # 串行执行（原有逻辑）
            for i, user_file in enumerate(md_files, 1):
                self.log(f"\n进度: {i}/{total_files_count}")
                result = self.process_file(user_file)
                results.append(result)
        
        # 生成评估表格
        self.log("\n" + "="*80)
        self.log("生成评估数据表格")
        self.log("="*80 + "\n")
        self.generate_evaluation_table()
        
        # 保存生成摘要（在 generate_only 和 full 模式下）
        if self.mode in ["generate_only", "full"]:
            if self.generator_summary_data:
                generator_summary = {
                    "timestamp": datetime.now().isoformat(),
                    "total_files": len(self.generator_summary_data),
                    "success_count": sum(1 for entry in self.generator_summary_data if entry.get("状态") == "success"),
                    "error_count": sum(1 for entry in self.generator_summary_data if entry.get("状态") == "error"),
                    "entries": self.generator_summary_data
                }
                generator_summary_file = self.output_dir / "batch_srs_generator_summary.json"
                with open(generator_summary_file, "w", encoding="utf-8") as f:
                    json.dump(generator_summary, f, ensure_ascii=False, indent=2)
                self.log(f"已保存生成摘要: {generator_summary_file}")
        
        # 保存评估摘要
        if self.evaluation_summary_data:
            evaluation_summary = {
                "timestamp": datetime.now().isoformat(),
                "total_files": len(self.evaluation_summary_data),
                "success_count": sum(1 for entry in self.evaluation_summary_data if entry.get("简单平均分") is not None),
                "error_count": sum(1 for entry in self.evaluation_summary_data if "错误" in entry),
                "entries": self.evaluation_summary_data
            }
            
            # 所有模式都保存到 output_dir
            evaluation_summary_file = self.output_dir / "batch_srs_evaluation_summary.json"
            
            with open(evaluation_summary_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
            self.log(f"已保存评估摘要: {evaluation_summary_file}")
        
        self.log(f"\n{'='*80}")
        self.log("批量处理完成")
        self.log(f"{'='*80}\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="批量处理SRS生成")
    parser.add_argument(
        "--user-input-dir",
        type=str,
        default="../srs-docs/summary_ultra_short",
        help="用户需求文本文件目录（默认: ../srs-docs/summary_ultra_short）"
    )
    parser.add_argument(
        "--reference-srs-dir",
        type=str,
        default="../srs-docs/req_md",
        help="参考SRS文本文件目录（默认: ../srs-docs/req_md）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./batch_output",
        help="输出目录（默认: ./batch_output）"
    )
    parser.add_argument(
        "-m", "--max-iterations",
        type=int,
        default=5,
        help="最大迭代轮数（默认: 5）"
    )
    parser.add_argument(
        "-n", "--max-files",
        type=int,
        default=None,
        help="最大处理文件数（默认: None，处理所有文件）"
    )
    parser.add_argument(
        "-w", "--max-workers",
        type=int,
        default=1,
        help="最大并发线程数（默认: 1，即串行执行）"
    )
    parser.add_argument(
        "--ablation-mode",
        type=str,
        choices=["no-clarify", "no-explore-clarify"],
        default=None,
        help="消融实验模式：no-clarify（移除需求澄清智能体）或 no-explore-clarify（移除需求挖掘+澄清智能体）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "generate_only", "evaluate_only"],
        default="full",
        help="处理模式：full（完整流程，默认）、generate_only（只生成）、evaluate_only（只评估）"
    )
    parser.add_argument(
        "--generated-srs-dir",
        type=str,
        default=None,
        help="已生成的SRS目录（可选，evaluate_only 模式下如果不指定则从 output_dir/srs_output 推导）"
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v2"],
        default="v2",
        help="使用的生成器版本：v1 或 v2（默认: v2）"
    )
    
    args = parser.parse_args()
    
    # 验证 max_workers
    if args.max_workers < 1:
        print(f"错误: 并发线程数必须 >= 1，当前值: {args.max_workers}")
        sys.exit(1)
    
    # 验证模式参数
    # evaluate_only 模式下，从 output_dir / "srs_output" 推导 generated_srs_dir，不需要单独指定
    if args.mode == "evaluate_only":
        # 确定 generated_srs_dir 路径
        if args.generated_srs_dir:
            generated_srs_path = args.generated_srs_dir
        else:
            generated_srs_path = os.path.join(args.output_dir, "srs_output")
        
        # 验证目录是否存在
        if not os.path.isdir(generated_srs_path):
            print(f"错误: 已生成SRS目录不存在: {generated_srs_path}")
            sys.exit(1)
    
    # 检查目录是否存在
    if not os.path.isdir(args.user_input_dir):
        print(f"错误: 用户需求目录不存在: {args.user_input_dir}")
        sys.exit(1)
    if not os.path.isdir(args.reference_srs_dir):
        print(f"错误: 参考SRS目录不存在: {args.reference_srs_dir}")
        sys.exit(1)
    
    # 创建处理器并运行
    processor = BatchProcessor(
        user_input_dir=args.user_input_dir,
        reference_srs_dir=args.reference_srs_dir,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        max_files=args.max_files,
        max_workers=args.max_workers,
        ablation_mode=args.ablation_mode,
        mode=args.mode,
        generated_srs_dir=args.generated_srs_dir,
        version=args.version
    )
    processor.run()


if __name__ == "__main__":
    main()

