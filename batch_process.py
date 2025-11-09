# -*- coding: utf-8 -*-
"""
批量处理脚本：对 summary_ultra_short 目录中的所有文件执行 srs_generator.py
- 记录完整日志
- 保存最终生成的SRS文档
- 聚合相似度数据生成表格

支持的处理模式：
- full: 完整流程（生成+评估+相似度，默认）
- generate_only: 只生成SRS（不评估、不计算相似度）
- evaluate_only: 只评估（需要指定已生成的SRS目录）
- similarity_only: 只计算相似度（需要指定已生成的SRS目录）
- evaluate_and_similarity: 评估+相似度（需要指定已生成的SRS目录）

使用示例：
  # 完整流程（默认）
  python batch_process.py --user-input-dir ./input --reference-srs-dir ./reference
  
  # 只生成SRS
  python batch_process.py --mode generate_only --user-input-dir ./input --reference-srs-dir ./reference
  
  # 先批量生成，再批量评估
  python batch_process.py --mode generate_only --user-input-dir ./input --reference-srs-dir ./reference --output-dir ./output1
  python batch_process.py --mode evaluate_only --user-input-dir ./input --reference-srs-dir ./reference --generated-srs-dir ./output1/srs_output --output-dir ./output2
"""

import os
import json
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import pandas as pd

# 导入 srs_generator.py 中的函数
from srs_generator import DemoInput, run_demo, log


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
        generated_srs_dir: Optional[str] = None
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
                - full: 完整流程（生成+评估+相似度，默认）
                - generate_only: 只生成SRS
                - evaluate_only: 只评估（需要指定 generated_srs_dir）
                - similarity_only: 只计算相似度（需要指定 generated_srs_dir）
                - evaluate_and_similarity: 评估+相似度（需要指定 generated_srs_dir）
            generated_srs_dir: 已生成的SRS目录（评估/相似度模式需要）
        """
        self.user_input_dir = Path(user_input_dir)
        self.reference_srs_dir = Path(reference_srs_dir)
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.max_files = max_files
        self.max_workers = max_workers
        self.ablation_mode = ablation_mode
        self.mode = mode
        self.generated_srs_dir = Path(generated_srs_dir) if generated_srs_dir else None
        
        # 创建输出目录结构
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.srs_output_dir = self.output_dir / "srs_output"
        self.log_output_dir = self.output_dir / "logs"
        self.srs_output_dir.mkdir(exist_ok=True)
        self.log_output_dir.mkdir(exist_ok=True)
        
        # 相似度数据列表（需要线程安全）
        self.similarity_data: List[Dict[str, Any]] = []
        self.similarity_data_lock = Lock()
        
        # 进度跟踪（需要线程安全）
        self.completed_count = 0
        self.completed_lock = Lock()
        
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
    
    def compute_similarity(self, generated_srs: str, user_input: str, reference_srs: str) -> Optional[Dict[str, Any]]:
        """
        计算相似度（通过独立脚本）
        
        Args:
            generated_srs: 生成的SRS文档
            user_input: 用户输入文本
            reference_srs: 参考SRS文档
            
        Returns:
            相似度结果字典，如果调用失败返回None
        """
        try:
            script_path = Path(__file__).parent / "compute_similarity.py"
            cmd = [
                sys.executable,
                str(script_path),
                "--text-generated", generated_srs,
                "--text-user-input", user_input,
                "--text-reference", reference_srs
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                # 从输出中提取JSON结果
                output_lines = result.stdout.strip().split('\n')
                json_start = -1
                for i, line in enumerate(output_lines):
                    if line.strip().startswith('{'):
                        json_start = i
                        break
                
                if json_start >= 0:
                    json_text = '\n'.join(output_lines[json_start:])
                    try:
                        return json.loads(json_text)
                    except json.JSONDecodeError:
                        pass
            return None
        except Exception as e:
            log(f"警告: 相似度计算失败: {str(e)}")
            return None
    
    def _extract_evaluation_scores(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        从评估结果中提取评分数据（支持新旧格式）
        
        Args:
            result: 评估结果字典
            
        Returns:
            包含评分数据的字典
        """
        scores = {
            "功能完整性_得分": None,
            "交互流程相似度_得分": None,
            "SRS总分": None,
            "综合评分_简单平均": None,
            "综合评分_加权平均": None,
        }
        
        # 新格式：从 metrics 字段提取
        if "metrics" in result:
            metrics = result["metrics"]
            # 新格式没有功能完整性和交互流程相似度，这些字段保持为 None
            
            # 提取综合评分
            if "Comprehensive_Score_Simple" in result:
                scores["综合评分_简单平均"] = result["Comprehensive_Score_Simple"]
            if "Comprehensive_Score_Weighted" in result:
                scores["综合评分_加权平均"] = result["Comprehensive_Score_Weighted"]
            
            # 使用加权平均作为 SRS总分（如果没有，使用简单平均）
            if scores["综合评分_加权平均"] is not None:
                scores["SRS总分"] = scores["综合评分_加权平均"]
            elif scores["综合评分_简单平均"] is not None:
                scores["SRS总分"] = scores["综合评分_简单平均"]
        
        # 旧格式：兼容性支持（已弃用）
        elif "Functional_Completeness" in result:
            fc = result.get("Functional_Completeness", {})
            if isinstance(fc, dict):
                scores["功能完整性_得分"] = fc.get("score")
            else:
                scores["功能完整性_得分"] = fc
            
            ifs = result.get("Interaction_Flow_Similarity", {})
            if isinstance(ifs, dict):
                scores["交互流程相似度_得分"] = ifs.get("score")
            else:
                scores["交互流程相似度_得分"] = ifs
            
            scores["SRS总分"] = result.get("Total_Score")
        
        return scores
    
    def evaluate_srs(self, standard_srs: str, evaluated_srs: str) -> Optional[Dict[str, Any]]:
        """
        调用SRS评分脚本评估SRS文档
        
        Args:
            standard_srs: 标准参考SRS文档
            evaluated_srs: 待评审SRS文档
            
        Returns:
            评分结果字典（新格式包含 metrics 和综合评分，旧格式包含 Functional_Completeness 等）
            如果调用失败，返回None
        """
        try:
            # 调用独立的 srs_evaluation.py 脚本
            script_path = Path(__file__).parent / "srs_evaluation.py"
            cmd = [
                sys.executable,
                str(script_path),
                "--text-standard", standard_srs,
                "--text-evaluated", evaluated_srs
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                log(f"警告: SRS评分脚本执行失败: {result.stderr}")
                return None
            
            # 从输出中提取JSON结果
            # 脚本会在最后输出JSON格式的结果
            output_lines = result.stdout.strip().split('\n')
            json_start = -1
            for i, line in enumerate(output_lines):
                if line.strip().startswith('{') or line.strip().startswith('['):
                    json_start = i
                    break
            
            if json_start >= 0:
                json_text = '\n'.join(output_lines[json_start:])
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    # 如果解析失败，尝试从整个输出中提取
                    pass
            
            # 如果无法从输出中提取JSON，返回None
            log(f"警告: 无法从SRS评分脚本输出中解析JSON结果")
            return None
            
        except subprocess.TimeoutExpired:
            log(f"警告: SRS评分脚本执行超时")
            return None
        except Exception as e:
            log(f"警告: SRS评分脚本调用失败: {str(e)}")
            return None
    
    def process_evaluation_only(self, user_file: Path) -> Dict[str, Any]:
        """
        处理只评估模式：读取已生成的SRS，只执行评估
        
        Returns:
            包含处理结果的字典
        """
        file_basename = user_file.stem
        log(f"\n{'='*80}")
        log(f"处理文件（只评估）: {file_basename}")
        log(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            # 读取文件
            ref_file = self.find_reference_file(user_file)
            generated_srs_file = self.find_generated_srs_file(user_file)
            
            with open(user_file, "r", encoding="utf-8") as f:
                user_input = f.read()
            with open(ref_file, "r", encoding="utf-8") as f:
                reference_srs = f.read()
            with open(generated_srs_file, "r", encoding="utf-8") as f:
                generated_srs = f.read()
            
            log(f"用户需求文件: {user_file}")
            log(f"参考SRS文件: {ref_file}")
            log(f"已生成SRS文件: {generated_srs_file}")
            
            # 调用SRS评分
            srs_evaluation_result = None
            try:
                log(f"正在调用SRS评分...")
                srs_evaluation_result = self.evaluate_srs(
                    standard_srs=reference_srs,
                    evaluated_srs=generated_srs
                )
                if srs_evaluation_result:
                    # 提取评分数据
                    scores = self._extract_evaluation_scores(srs_evaluation_result)
                    total_score = scores.get("SRS总分") or srs_evaluation_result.get("Total_Score", "N/A")
                    log(f"SRS评分完成: 总分={total_score}")
                else:
                    log(f"警告: SRS评分调用失败或返回空结果")
            except Exception as e:
                log(f"警告: SRS评分调用异常: {str(e)}")
            
            elapsed_time = time.time() - start_time
            
            # 收集数据
            similarity_entry = {
                "文件": file_basename,
                "迭代轮数": None,
                "需求数量": None,
                "冻结数量": None,
                "移除数量": None,
                "耗时(秒)": round(elapsed_time, 2),
                "生成vs参考_相似度": None,
                "用户vs参考_相似度": None,
            }
            
            # 添加SRS评分数据
            if srs_evaluation_result:
                scores = self._extract_evaluation_scores(srs_evaluation_result)
                similarity_entry.update(scores)
            else:
                similarity_entry["功能完整性_得分"] = None
                similarity_entry["交互流程相似度_得分"] = None
                similarity_entry["SRS总分"] = None
                similarity_entry["综合评分_简单平均"] = None
                similarity_entry["综合评分_加权平均"] = None
            
            with self.similarity_data_lock:
                self.similarity_data.append(similarity_entry)
            
            log(f"✓ 文件 {file_basename} 评估完成，耗时: {elapsed_time:.2f} 秒")
            return {
                "status": "success",
                "file": file_basename,
                "similarity_entry": similarity_entry
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"处理文件 {file_basename} 时出错: {str(e)}"
            log(f"✗ {error_msg}")
            
            similarity_entry = {
                "文件": file_basename,
                "迭代轮数": None,
                "需求数量": None,
                "冻结数量": None,
                "移除数量": None,
                "耗时(秒)": round(elapsed_time, 2),
                "生成vs参考_相似度": None,
                "用户vs参考_相似度": None,
                "功能完整性_得分": None,
                "交互流程相似度_得分": None,
                "SRS总分": None,
                "综合评分_简单平均": None,
                "综合评分_加权平均": None,
                "错误": str(e)
            }
            with self.similarity_data_lock:
                self.similarity_data.append(similarity_entry)
            
            return {
                "status": "error",
                "file": file_basename,
                "error": str(e),
                "similarity_entry": similarity_entry
            }
    
    def process_similarity_only(self, user_file: Path) -> Dict[str, Any]:
        """
        处理只相似度模式：读取已生成的SRS，只执行相似度计算
        
        Returns:
            包含处理结果的字典
        """
        file_basename = user_file.stem
        log(f"\n{'='*80}")
        log(f"处理文件（只相似度）: {file_basename}")
        log(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            # 读取文件
            ref_file = self.find_reference_file(user_file)
            generated_srs_file = self.find_generated_srs_file(user_file)
            
            with open(user_file, "r", encoding="utf-8") as f:
                user_input = f.read()
            with open(ref_file, "r", encoding="utf-8") as f:
                reference_srs = f.read()
            with open(generated_srs_file, "r", encoding="utf-8") as f:
                generated_srs = f.read()
            
            log(f"用户需求文件: {user_file}")
            log(f"参考SRS文件: {ref_file}")
            log(f"已生成SRS文件: {generated_srs_file}")
            
            # 计算相似度
            similarity_result = None
            try:
                log(f"正在计算相似度...")
                similarity_result = self.compute_similarity(
                    generated_srs=generated_srs,
                    user_input=user_input,
                    reference_srs=reference_srs
                )
                if similarity_result:
                    log(f"相似度计算完成")
                else:
                    log(f"警告: 相似度计算失败或返回空结果")
            except Exception as e:
                log(f"警告: 相似度计算异常: {str(e)}")
            
            elapsed_time = time.time() - start_time
            
            # 收集数据
            similarity_entry = {
                "文件": file_basename,
                "迭代轮数": None,
                "需求数量": None,
                "冻结数量": None,
                "移除数量": None,
                "耗时(秒)": round(elapsed_time, 2),
                "生成vs参考_相似度": similarity_result.get("generated_vs_reference") if similarity_result else None,
                "用户vs参考_相似度": similarity_result.get("user_input_vs_reference") if similarity_result else None,
                "功能完整性_得分": None,
                "交互流程相似度_得分": None,
                "SRS总分": None,
                "综合评分_简单平均": None,
                "综合评分_加权平均": None,
            }
            
            with self.similarity_data_lock:
                self.similarity_data.append(similarity_entry)
            
            log(f"✓ 文件 {file_basename} 相似度计算完成，耗时: {elapsed_time:.2f} 秒")
            return {
                "status": "success",
                "file": file_basename,
                "similarity_entry": similarity_entry
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"处理文件 {file_basename} 时出错: {str(e)}"
            log(f"✗ {error_msg}")
            
            similarity_entry = {
                "文件": file_basename,
                "迭代轮数": None,
                "需求数量": None,
                "冻结数量": None,
                "移除数量": None,
                "耗时(秒)": round(elapsed_time, 2),
                "生成vs参考_相似度": None,
                "用户vs参考_相似度": None,
                "功能完整性_得分": None,
                "交互流程相似度_得分": None,
                "SRS总分": None,
                "综合评分_简单平均": None,
                "综合评分_加权平均": None,
                "错误": str(e)
            }
            with self.similarity_data_lock:
                self.similarity_data.append(similarity_entry)
            
            return {
                "status": "error",
                "file": file_basename,
                "error": str(e),
                "similarity_entry": similarity_entry
            }
    
    def process_evaluate_and_similarity(self, user_file: Path) -> Dict[str, Any]:
        """
        处理评估+相似度模式：读取已生成的SRS，执行评估和相似度计算
        
        Returns:
            包含处理结果的字典
        """
        file_basename = user_file.stem
        log(f"\n{'='*80}")
        log(f"处理文件（评估+相似度）: {file_basename}")
        log(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            # 读取文件
            ref_file = self.find_reference_file(user_file)
            generated_srs_file = self.find_generated_srs_file(user_file)
            
            with open(user_file, "r", encoding="utf-8") as f:
                user_input = f.read()
            with open(ref_file, "r", encoding="utf-8") as f:
                reference_srs = f.read()
            with open(generated_srs_file, "r", encoding="utf-8") as f:
                generated_srs = f.read()
            
            log(f"用户需求文件: {user_file}")
            log(f"参考SRS文件: {ref_file}")
            log(f"已生成SRS文件: {generated_srs_file}")
            
            # 调用SRS评分
            srs_evaluation_result = None
            try:
                log(f"正在调用SRS评分...")
                srs_evaluation_result = self.evaluate_srs(
                    standard_srs=reference_srs,
                    evaluated_srs=generated_srs
                )
                if srs_evaluation_result:
                    # 提取评分数据
                    scores = self._extract_evaluation_scores(srs_evaluation_result)
                    total_score = scores.get("SRS总分") or srs_evaluation_result.get("Total_Score", "N/A")
                    log(f"SRS评分完成: 总分={total_score}")
                else:
                    log(f"警告: SRS评分调用失败或返回空结果")
            except Exception as e:
                log(f"警告: SRS评分调用异常: {str(e)}")
            
            # 计算相似度
            similarity_result = None
            try:
                log(f"正在计算相似度...")
                similarity_result = self.compute_similarity(
                    generated_srs=generated_srs,
                    user_input=user_input,
                    reference_srs=reference_srs
                )
                if similarity_result:
                    log(f"相似度计算完成")
                else:
                    log(f"警告: 相似度计算失败或返回空结果")
            except Exception as e:
                log(f"警告: 相似度计算异常: {str(e)}")
            
            elapsed_time = time.time() - start_time
            
            # 收集数据
            similarity_entry = {
                "文件": file_basename,
                "迭代轮数": None,
                "需求数量": None,
                "冻结数量": None,
                "移除数量": None,
                "耗时(秒)": round(elapsed_time, 2),
                "生成vs参考_相似度": similarity_result.get("generated_vs_reference") if similarity_result else None,
                "用户vs参考_相似度": similarity_result.get("user_input_vs_reference") if similarity_result else None,
            }
            
            # 添加SRS评分数据
            if srs_evaluation_result:
                scores = self._extract_evaluation_scores(srs_evaluation_result)
                similarity_entry.update(scores)
            else:
                similarity_entry["功能完整性_得分"] = None
                similarity_entry["交互流程相似度_得分"] = None
                similarity_entry["SRS总分"] = None
                similarity_entry["综合评分_简单平均"] = None
                similarity_entry["综合评分_加权平均"] = None
            
            with self.similarity_data_lock:
                self.similarity_data.append(similarity_entry)
            
            log(f"✓ 文件 {file_basename} 处理完成，耗时: {elapsed_time:.2f} 秒")
            return {
                "status": "success",
                "file": file_basename,
                "similarity_entry": similarity_entry
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"处理文件 {file_basename} 时出错: {str(e)}"
            log(f"✗ {error_msg}")
            
            similarity_entry = {
                "文件": file_basename,
                "迭代轮数": None,
                "需求数量": None,
                "冻结数量": None,
                "移除数量": None,
                "耗时(秒)": round(elapsed_time, 2),
                "生成vs参考_相似度": None,
                "用户vs参考_相似度": None,
                "功能完整性_得分": None,
                "交互流程相似度_得分": None,
                "SRS总分": None,
                "综合评分_简单平均": None,
                "综合评分_加权平均": None,
                "错误": str(e)
            }
            with self.similarity_data_lock:
                self.similarity_data.append(similarity_entry)
            
            return {
                "status": "error",
                "file": file_basename,
                "error": str(e),
                "similarity_entry": similarity_entry
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
        elif self.mode == "similarity_only":
            return self.process_similarity_only(user_file)
        elif self.mode == "evaluate_and_similarity":
            return self.process_evaluate_and_similarity(user_file)
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
        log(f"\n{'='*80}")
        log(f"处理文件（只生成）: {file_basename}")
        log(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            # 读取文件
            ref_file = self.find_reference_file(user_file)
            with open(user_file, "r", encoding="utf-8") as f:
                user_input = f.read()
            with open(ref_file, "r", encoding="utf-8") as f:
                reference_srs = f.read()
            
            log(f"用户需求文件: {user_file}")
            log(f"参考SRS文件: {ref_file}")
            log(f"用户需求长度: {len(user_input)} 字符")
            log(f"参考SRS长度: {len(reference_srs)} 字符")
            
            # 创建DemoInput并运行
            demo = DemoInput(
                user_input=user_input,
                reference_srs=reference_srs,
                max_iterations=self.max_iterations,
                ablation_mode=self.ablation_mode
            )
            
            # 运行演示（silent=True 以减少输出）
            final_state = run_demo(demo, silent=True)
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            
            # 保存SRS文档
            srs_output_file = self.srs_output_dir / f"{file_basename}_srs.md"
            with open(srs_output_file, "w", encoding="utf-8") as f:
                f.write(final_state["srs_output"])
            log(f"已保存SRS文档: {srs_output_file}")
            
            # 保存完整日志（JSON格式）
            log_output_file = self.log_output_dir / f"{file_basename}_log.json"
            log_data = {
                "file": file_basename,
                "timestamp": datetime.now().isoformat(),
                "user_input_file": str(user_file),
                "reference_srs_file": str(ref_file),
                "user_input": user_input,
                "reference_srs": reference_srs,
                "max_iterations": self.max_iterations,
                "ablation_mode": self.ablation_mode,
                "final_iteration": final_state["iteration"],
                "elapsed_time_seconds": elapsed_time,
                "req_list": final_state["req_list"],
                "frozen_ids": final_state.get("frozen_ids", []),
                "removed_ids": final_state.get("removed_ids", []),
                "scores": final_state.get("scores", {}),
                "logs": final_state["logs"],
                "srs_output": final_state["srs_output"],
                "similarity": None,
                "srs_evaluation": None
            }
            with open(log_output_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            log(f"已保存日志: {log_output_file}")
            
            log(f"✓ 文件 {file_basename} 生成完成，耗时: {elapsed_time:.2f} 秒")
            return {
                "status": "success",
                "file": file_basename,
                "similarity_entry": None
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"处理文件 {file_basename} 时出错: {str(e)}"
            log(f"✗ {error_msg}")
            
            return {
                "status": "error",
                "file": file_basename,
                "error": str(e),
                "similarity_entry": None
            }
    
    def process_full(self, user_file: Path) -> Dict[str, Any]:
        """
        处理完整流程模式：生成+评估+相似度（原有逻辑）
        
        Returns:
            包含处理结果的字典
        """
        file_basename = user_file.stem
        log(f"\n{'='*80}")
        log(f"处理文件: {file_basename}")
        log(f"{'='*80}\n")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 读取文件
            ref_file = self.find_reference_file(user_file)
            with open(user_file, "r", encoding="utf-8") as f:
                user_input = f.read()
            with open(ref_file, "r", encoding="utf-8") as f:
                reference_srs = f.read()
            
            log(f"用户需求文件: {user_file}")
            log(f"参考SRS文件: {ref_file}")
            log(f"用户需求长度: {len(user_input)} 字符")
            log(f"参考SRS长度: {len(reference_srs)} 字符")
            
            # 创建DemoInput并运行
            demo = DemoInput(
                user_input=user_input,
                reference_srs=reference_srs,
                max_iterations=self.max_iterations,
                ablation_mode=self.ablation_mode
            )
            
            # 运行演示（silent=True 以减少输出）
            final_state = run_demo(demo, silent=True)
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            
            # 保存SRS文档
            srs_output_file = self.srs_output_dir / f"{file_basename}_srs.md"
            with open(srs_output_file, "w", encoding="utf-8") as f:
                f.write(final_state["srs_output"])
            log(f"已保存SRS文档: {srs_output_file}")
            
            # 调用SRS评分
            srs_evaluation_result = None
            try:
                log(f"正在调用SRS评分...")
                srs_evaluation_result = self.evaluate_srs(
                    standard_srs=reference_srs,
                    evaluated_srs=final_state["srs_output"]
                )
                if srs_evaluation_result:
                    # 提取评分数据
                    scores = self._extract_evaluation_scores(srs_evaluation_result)
                    total_score = scores.get("SRS总分") or srs_evaluation_result.get("Total_Score", "N/A")
                    log(f"SRS评分完成: 总分={total_score}")
                else:
                    log(f"警告: SRS评分调用失败或返回空结果")
            except Exception as e:
                log(f"警告: SRS评分调用异常: {str(e)}")
            
            # 计算相似度
            similarity_result = None
            try:
                log(f"正在计算相似度...")
                similarity_result = self.compute_similarity(
                    generated_srs=final_state["srs_output"],
                    user_input=user_input,
                    reference_srs=reference_srs
                )
                if similarity_result:
                    log(f"相似度计算完成")
                else:
                    log(f"警告: 相似度计算失败或返回空结果")
            except Exception as e:
                log(f"警告: 相似度计算异常: {str(e)}")
            
            # 保存完整日志（JSON格式）
            log_output_file = self.log_output_dir / f"{file_basename}_log.json"
            log_data = {
                "file": file_basename,
                "timestamp": datetime.now().isoformat(),
                "user_input_file": str(user_file),
                "reference_srs_file": str(ref_file),
                "user_input": user_input,
                "reference_srs": reference_srs,
                "max_iterations": self.max_iterations,
                "ablation_mode": self.ablation_mode,
                "final_iteration": final_state["iteration"],
                "elapsed_time_seconds": elapsed_time,
                "req_list": final_state["req_list"],
                "frozen_ids": final_state.get("frozen_ids", []),
                "removed_ids": final_state.get("removed_ids", []),
                "scores": final_state.get("scores", {}),
                "logs": final_state["logs"],
                "srs_output": final_state["srs_output"],
                "similarity": similarity_result,
                "srs_evaluation": srs_evaluation_result
            }
            with open(log_output_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            log(f"已保存日志: {log_output_file}")
            
            # 收集相似度数据（线程安全）
            similarity_entry = {
                "文件": file_basename,
                "迭代轮数": final_state["iteration"],
                "需求数量": len(final_state["req_list"]),
                "冻结数量": len(final_state.get("frozen_ids", [])),
                "移除数量": len(final_state.get("removed_ids", [])),
                "耗时(秒)": round(elapsed_time, 2),
                "生成vs参考_相似度": similarity_result.get("generated_vs_reference") if similarity_result else None,
                "用户vs参考_相似度": similarity_result.get("user_input_vs_reference") if similarity_result else None,
            }
            
            # 添加SRS评分数据
            if srs_evaluation_result:
                scores = self._extract_evaluation_scores(srs_evaluation_result)
                similarity_entry.update(scores)
            else:
                similarity_entry["功能完整性_得分"] = None
                similarity_entry["交互流程相似度_得分"] = None
                similarity_entry["SRS总分"] = None
                similarity_entry["综合评分_简单平均"] = None
                similarity_entry["综合评分_加权平均"] = None
            with self.similarity_data_lock:
                self.similarity_data.append(similarity_entry)
            
            log(f"✓ 文件 {file_basename} 处理完成，耗时: {elapsed_time:.2f} 秒")
            return {
                "status": "success",
                "file": file_basename,
                "similarity_entry": similarity_entry
            }
            
        except Exception as e:
            # 计算耗时（即使出错也记录）
            elapsed_time = time.time() - start_time
            error_msg = f"处理文件 {file_basename} 时出错: {str(e)}"
            log(f"✗ {error_msg}")
            
            # 记录错误信息（线程安全）
            similarity_entry = {
                "文件": file_basename,
                "迭代轮数": None,
                "需求数量": None,
                "冻结数量": None,
                "移除数量": None,
                "耗时(秒)": round(elapsed_time, 2),
                "生成vs参考_相似度": None,
                "用户vs参考_相似度": None,
                "功能完整性_得分": None,
                "交互流程相似度_得分": None,
                "SRS总分": None,
                "综合评分_简单平均": None,
                "综合评分_加权平均": None,
                "错误": str(e)
            }
            with self.similarity_data_lock:
                self.similarity_data.append(similarity_entry)
            
            return {
                "status": "error",
                "file": file_basename,
                "error": str(e),
                "similarity_entry": similarity_entry
            }
    
    def generate_similarity_table(self) -> None:
        """生成相似度数据表格"""
        if not self.similarity_data:
            log("警告: 没有相似度数据可生成表格")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.similarity_data)
        
        # 保存为CSV
        csv_file = self.output_dir / "similarity_table.csv"
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        log(f"已保存相似度表格(CSV): {csv_file}")
        
        # 保存为Excel（如果可用）
        try:
            excel_file = self.output_dir / "similarity_table.xlsx"
            df.to_excel(excel_file, index=False, engine="openpyxl")
            log(f"已保存相似度表格(Excel): {excel_file}")
        except Exception as e:
            log(f"无法保存Excel文件（可能需要安装openpyxl）: {e}")
        
        # 打印统计摘要
        log("\n" + "="*80)
        log("相似度数据统计摘要")
        log("="*80)
        
        # 计算平均值（仅对数值列）
        numeric_cols = [
            "生成vs参考_相似度", "用户vs参考_相似度", "耗时(秒)", 
            "功能完整性_得分", "交互流程相似度_得分", "SRS总分",
            "综合评分_简单平均", "综合评分_加权平均"
        ]
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    if col == "耗时(秒)":
                        log(f"{col}: 平均值={mean_val:.2f}秒, 标准差={std_val:.2f}秒, 样本数={len(values)}, 总耗时={values.sum():.2f}秒")
                    elif col in ["功能完整性_得分", "交互流程相似度_得分", "SRS总分", "综合评分_简单平均", "综合评分_加权平均"]:
                        log(f"{col}: 平均值={mean_val:.4f}, 标准差={std_val:.4f}, 样本数={len(values)}")
                    else:
                        log(f"{col}: 平均值={mean_val:.6f}, 标准差={std_val:.6f}, 样本数={len(values)}")
        
        log(f"总文件数: {len(df)}")
        success_count = len(df[df["迭代轮数"].notna()])
        error_count = len(df) - success_count
        log(f"成功处理: {success_count}, 失败: {error_count}")
        log("="*80 + "\n")
        
        # 打印表格预览
        print("\n相似度数据表格预览（前10行）:")
        print(df.head(10).to_string())
    
    def run(self) -> None:
        """运行批量处理"""
        log(f"\n{'='*80}")
        log("批量处理开始")
        log(f"处理模式: {self.mode}")
        log(f"用户需求目录: {self.user_input_dir}")
        log(f"参考SRS目录: {self.reference_srs_dir}")
        log(f"输出目录: {self.output_dir}")
        if self.generated_srs_dir is not None:
            log(f"已生成SRS目录: {self.generated_srs_dir}")
        log(f"最大迭代轮数: {self.max_iterations}")
        if self.ablation_mode is not None:
            log(f"消融实验模式: {self.ablation_mode}")
        if self.max_files is not None:
            log(f"最大处理文件数: {self.max_files}")
        log(f"并发线程数: {self.max_workers}")
        log(f"{'='*80}\n")
        
        # 获取所有文件
        md_files = self.get_md_files()
        total_files = len(md_files)
        
        # 如果设置了 max_files，只处理前 N 个文件
        if self.max_files is not None and self.max_files > 0:
            md_files = md_files[:self.max_files]
            log(f"找到 {total_files} 个文件，将处理前 {len(md_files)} 个文件\n")
        else:
            log(f"找到 {total_files} 个文件需要处理\n")
        
        if len(md_files) == 0:
            log("警告: 没有找到任何 .md 文件")
            return
        
        # 重置进度计数器
        self.completed_count = 0
        
        # 处理每个文件（支持并发）
        results = []
        total_files_count = len(md_files)
        
        if self.max_workers > 1:
            # 并发执行
            log(f"使用 {self.max_workers} 个线程并发处理 {total_files_count} 个文件\n")
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
                            log(f"进度: {self.completed_count}/{total_files_count} - {result['file']}")
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
                            log(f"进度: {self.completed_count}/{total_files_count} - {file_basename} (错误)")
        else:
            # 串行执行（原有逻辑）
            for i, user_file in enumerate(md_files, 1):
                log(f"\n进度: {i}/{total_files_count}")
                result = self.process_file(user_file)
                results.append(result)
        
        # 生成相似度表格
        log("\n" + "="*80)
        log("生成相似度数据表格")
        log("="*80 + "\n")
        self.generate_similarity_table()
        
        # 保存处理摘要
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_files": total_files,
            "success_count": sum(1 for r in results if r["status"] == "success"),
            "error_count": sum(1 for r in results if r["status"] == "error"),
            "results": results
        }
        summary_file = self.output_dir / "batch_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log(f"已保存处理摘要: {summary_file}")
        
        log(f"\n{'='*80}")
        log("批量处理完成")
        log(f"{'='*80}\n")


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
        choices=["full", "generate_only", "evaluate_only", "similarity_only", "evaluate_and_similarity"],
        default="full",
        help="处理模式：full（完整流程，默认）、generate_only（只生成）、evaluate_only（只评估）、similarity_only（只相似度）、evaluate_and_similarity（评估+相似度）"
    )
    parser.add_argument(
        "--generated-srs-dir",
        type=str,
        default=None,
        help="已生成的SRS目录（评估/相似度模式需要，例如: ./batch_output/srs_output）"
    )
    
    args = parser.parse_args()
    
    # 验证 max_workers
    if args.max_workers < 1:
        print(f"错误: 并发线程数必须 >= 1，当前值: {args.max_workers}")
        sys.exit(1)
    
    # 验证模式参数
    evaluation_modes = ["evaluate_only", "similarity_only", "evaluate_and_similarity"]
    if args.mode in evaluation_modes:
        if not args.generated_srs_dir:
            print(f"错误: 模式 '{args.mode}' 需要指定 --generated-srs-dir 参数")
            sys.exit(1)
        if not os.path.isdir(args.generated_srs_dir):
            print(f"错误: 已生成SRS目录不存在: {args.generated_srs_dir}")
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
        generated_srs_dir=args.generated_srs_dir
    )
    processor.run()


if __name__ == "__main__":
    main()

