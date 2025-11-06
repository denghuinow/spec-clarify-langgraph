# -*- coding: utf-8 -*-
"""
批量处理脚本：对 summary_5_10 目录中的所有文件执行 srs_generator.py
- 记录完整日志
- 保存最终生成的SRS文档
- 聚合相似度数据生成表格
"""

import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
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
        max_workers: int = 1
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
        """
        self.user_input_dir = Path(user_input_dir)
        self.reference_srs_dir = Path(reference_srs_dir)
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.max_files = max_files
        self.max_workers = max_workers
        
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
    
    def process_file(self, user_file: Path) -> Dict[str, Any]:
        """
        处理单个文件
        
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
                max_iterations=self.max_iterations
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
                "max_iterations": self.max_iterations,
                "final_iteration": final_state["iteration"],
                "elapsed_time_seconds": elapsed_time,
                "req_list": final_state["req_list"],
                "frozen_ids": final_state.get("frozen_ids", []),
                "removed_ids": final_state.get("removed_ids", []),
                "scores": final_state.get("scores", {}),
                "logs": final_state["logs"],
                "srs_output": final_state["srs_output"],
                "similarity": {
                    "embedding_similarity": final_state.get("embedding_similarity"),
                    "embedding_similarity_user_ref": final_state.get("embedding_similarity_user_ref"),
                }
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
                "生成vs参考_相似度": final_state.get("embedding_similarity"),
                "用户vs参考_相似度": final_state.get("embedding_similarity_user_ref"),
            }
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
        numeric_cols = ["生成vs参考_相似度", "用户vs参考_相似度", "耗时(秒)"]
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    if col == "耗时(秒)":
                        log(f"{col}: 平均值={mean_val:.2f}秒, 标准差={std_val:.2f}秒, 样本数={len(values)}, 总耗时={values.sum():.2f}秒")
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
        log(f"用户需求目录: {self.user_input_dir}")
        log(f"参考SRS目录: {self.reference_srs_dir}")
        log(f"输出目录: {self.output_dir}")
        log(f"最大迭代轮数: {self.max_iterations}")
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
        default="../srs-docs/summary_5_10",
        help="用户需求文本文件目录（默认: ../srs-docs/summary_5_10）"
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
    
    args = parser.parse_args()
    
    # 验证 max_workers
    if args.max_workers < 1:
        print(f"错误: 并发线程数必须 >= 1，当前值: {args.max_workers}")
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
        max_workers=args.max_workers
    )
    processor.run()


if __name__ == "__main__":
    main()

