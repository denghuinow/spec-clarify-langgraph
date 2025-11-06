# -*- coding: utf-8 -*-
"""
批量比较两个文件夹中同名文件的相似度
----------------------------------
遍历两个文件夹，找到同名文件，计算相似度并生成结果表格

使用方法：
    python batch_compare_folders.py --dir1 folder1 --dir2 folder2
    python batch_compare_folders.py --dir1 folder1 --dir2 folder2 --ext .md
    python batch_compare_folders.py --dir1 folder1 --dir2 folder2 --metric cosine euclidean
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import pandas as pd

# 导入相似度计算函数
from text_similarity import compute_similarity_metrics, compute_direct_distance


def log(message: str) -> None:
    """打印日志消息"""
    print(message, flush=True)


class FolderComparator:
    """文件夹比较器"""
    
    def __init__(
        self,
        dir1: str,
        dir2: str,
        output_dir: str = "./compare_output",
        file_ext: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        max_workers: int = 1,
        embedding_model: Optional[str] = None
    ):
        """
        初始化文件夹比较器
        
        Args:
            dir1: 第一个文件夹路径
            dir2: 第二个文件夹路径
            output_dir: 输出目录（保存结果表格）
            file_ext: 文件扩展名过滤（例如：'.md', '.txt'），None表示所有文件
            metrics: 要计算的相似度指标列表，None表示只计算余弦相似度
            max_workers: 最大并发线程数（默认: 1，即串行执行）
            embedding_model: 嵌入模型名称
        """
        self.dir1 = Path(dir1)
        self.dir2 = Path(dir2)
        self.output_dir = Path(output_dir)
        self.file_ext = file_ext
        self.metrics = metrics or ['cosine']
        self.max_workers = max_workers
        self.embedding_model = embedding_model
        
        # 验证文件夹存在
        if not self.dir1.exists() or not self.dir1.is_dir():
            raise ValueError(f"第一个文件夹不存在或不是目录: {self.dir1}")
        if not self.dir2.exists() or not self.dir2.is_dir():
            raise ValueError(f"第二个文件夹不存在或不是目录: {self.dir2}")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果数据列表（需要线程安全）
        self.comparison_data: List[Dict[str, Any]] = []
        self.comparison_data_lock = Lock()
        
        # 进度跟踪（需要线程安全）
        self.completed_count = 0
        self.completed_lock = Lock()
    
    def get_files(self) -> List[Path]:
        """
        获取第一个文件夹中的所有文件（根据扩展名过滤）
        
        Returns:
            文件路径列表
        """
        if self.file_ext:
            files = list(self.dir1.glob(f"*{self.file_ext}"))
        else:
            # 获取所有文件（不包括子目录）
            files = [f for f in self.dir1.iterdir() if f.is_file()]
        
        files.sort()
        return files
    
    def find_matching_file(self, file1: Path) -> Optional[Path]:
        """
        在第二个文件夹中查找同名文件
        
        Args:
            file1: 第一个文件夹中的文件路径
            
        Returns:
            第二个文件夹中的同名文件路径，如果不存在则返回None
        """
        file2 = self.dir2 / file1.name
        if file2.exists() and file2.is_file():
            return file2
        return None
    
    def compare_files(self, file1: Path, file2: Path) -> Dict[str, Any]:
        """
        比较两个文件的相似度
        
        Args:
            file1: 第一个文件路径
            file2: 第二个文件路径
            
        Returns:
            包含比较结果的字典
        """
        file_name = file1.name
        
        try:
            # 读取文件内容
            with open(file1, "r", encoding="utf-8") as f:
                text1 = f.read()
            with open(file2, "r", encoding="utf-8") as f:
                text2 = f.read()
            
            # 记录文件信息
            len1 = len(text1)
            len2 = len(text2)
            
            # 计算相似度
            start_time = time.time()
            similarity_results = compute_similarity_metrics(
                text1,
                text2,
                embedding_model=self.embedding_model,
                metrics=self.metrics
            )
            elapsed_time = time.time() - start_time
            
            # 构建结果字典
            result = {
                "文件名": file_name,
                "文件1长度": len1,
                "文件2长度": len2,
                "长度差": abs(len1 - len2),
                "耗时(秒)": round(elapsed_time, 2)
            }
            
            # 添加相似度指标
            metric_names = {
                'cosine': '余弦相似度',
                'euclidean': '欧几里得距离',
                'manhattan': '曼哈顿距离',
                'dot': '点积',
                'pearson': '皮尔逊相关系数'
            }
            
            for metric in self.metrics:
                if metric in similarity_results:
                    value = similarity_results[metric]
                    # 使用中文名称
                    metric_name = metric_names.get(metric, metric)
                    result[metric_name] = round(value, 6)
            
            log(f"✓ {file_name}: 余弦相似度={similarity_results.get('cosine', 'N/A'):.6f}, 耗时={elapsed_time:.2f}秒")
            return result
            
        except Exception as e:
            error_msg = f"比较文件 {file_name} 时出错: {str(e)}"
            log(f"✗ {error_msg}")
            return {
                "文件名": file_name,
                "文件1长度": None,
                "文件2长度": None,
                "长度差": None,
                "耗时(秒)": None,
                "错误": str(e)
            }
    
    def process_file(self, file1: Path) -> Dict[str, Any]:
        """
        处理单个文件（查找匹配文件并比较）
        
        Args:
            file1: 第一个文件夹中的文件路径
            
        Returns:
            包含处理结果的字典
        """
        file2 = self.find_matching_file(file1)
        
        if file2 is None:
            log(f"⚠ {file1.name}: 在第二个文件夹中未找到同名文件")
            return {
                "文件名": file1.name,
                "文件1长度": None,
                "文件2长度": None,
                "长度差": None,
                "耗时(秒)": None,
                "错误": "第二个文件夹中未找到同名文件"
            }
        
        result = self.compare_files(file1, file2)
        
        # 线程安全地添加结果
        with self.comparison_data_lock:
            self.comparison_data.append(result)
        
        return result
    
    def generate_comparison_table(self) -> None:
        """生成比较结果表格"""
        if not self.comparison_data:
            log("警告: 没有比较数据可生成表格")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.comparison_data)
        
        # 保存为CSV
        csv_file = self.output_dir / "comparison_table.csv"
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        log(f"已保存比较结果表格(CSV): {csv_file}")
        
        # 保存为Excel（如果可用）
        try:
            excel_file = self.output_dir / "comparison_table.xlsx"
            df.to_excel(excel_file, index=False, engine="openpyxl")
            log(f"已保存比较结果表格(Excel): {excel_file}")
        except Exception as e:
            log(f"无法保存Excel文件（可能需要安装openpyxl）: {e}")
        
        # 打印统计摘要
        log("\n" + "="*80)
        log("比较结果统计摘要")
        log("="*80)
        
        # 计算平均值（仅对数值列）
        numeric_cols = [col for col in df.columns if col not in ["文件名", "错误"]]
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    if col == "耗时(秒)":
                        log(f"{col}: 平均值={mean_val:.2f}秒, 标准差={std_val:.2f}秒, 样本数={len(values)}, 总耗时={values.sum():.2f}秒")
                    elif col == "长度差":
                        log(f"{col}: 平均值={mean_val:.0f}, 标准差={std_val:.0f}, 样本数={len(values)}")
                    else:
                        log(f"{col}: 平均值={mean_val:.6f}, 标准差={std_val:.6f}, 样本数={len(values)}")
        
        log(f"总文件数: {len(df)}")
        success_count = len(df[df["文件1长度"].notna()])
        error_count = len(df) - success_count
        log(f"成功比较: {success_count}, 失败: {error_count}")
        log("="*80 + "\n")
        
        # 打印表格预览
        print("\n比较结果表格预览（前10行）:")
        print(df.head(10).to_string())
    
    def run(self) -> None:
        """运行批量比较"""
        log(f"\n{'='*80}")
        log("批量文件夹比较开始")
        log(f"第一个文件夹: {self.dir1}")
        log(f"第二个文件夹: {self.dir2}")
        log(f"输出目录: {self.output_dir}")
        if self.file_ext:
            log(f"文件扩展名过滤: {self.file_ext}")
        log(f"相似度指标: {', '.join(self.metrics)}")
        log(f"并发线程数: {self.max_workers}")
        log(f"{'='*80}\n")
        
        # 获取所有文件
        files = self.get_files()
        total_files = len(files)
        
        if total_files == 0:
            log("警告: 第一个文件夹中没有找到符合条件的文件")
            return
        
        log(f"找到 {total_files} 个文件需要比较\n")
        
        # 重置进度计数器
        self.completed_count = 0
        
        # 处理每个文件（支持并发）
        results = []
        
        if self.max_workers > 1:
            # 并发执行
            log(f"使用 {self.max_workers} 个线程并发处理 {total_files} 个文件\n")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_file = {
                    executor.submit(self.process_file, file1): file1 
                    for file1 in files
                }
                
                # 收集结果
                for future in as_completed(future_to_file):
                    file1 = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # 更新进度（线程安全）
                        with self.completed_lock:
                            self.completed_count += 1
                            log(f"进度: {self.completed_count}/{total_files} - {result['文件名']}")
                    except Exception as e:
                        error_result = {
                            "文件名": file1.name,
                            "错误": f"执行异常: {str(e)}"
                        }
                        results.append(error_result)
                        
                        with self.completed_lock:
                            self.completed_count += 1
                            log(f"进度: {self.completed_count}/{total_files} - {file1.name} (错误)")
        else:
            # 串行执行
            for i, file1 in enumerate(files, 1):
                log(f"\n进度: {i}/{total_files}")
                result = self.process_file(file1)
                results.append(result)
        
        # 生成比较结果表格
        log("\n" + "="*80)
        log("生成比较结果表格")
        log("="*80 + "\n")
        self.generate_comparison_table()
        
        log(f"\n{'='*80}")
        log("批量文件夹比较完成")
        log(f"{'='*80}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量比较两个文件夹中同名文件的相似度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  %(prog)s --dir1 folder1 --dir2 folder2
  %(prog)s --dir1 folder1 --dir2 folder2 --ext .md
  %(prog)s --dir1 folder1 --dir2 folder2 --metric cosine euclidean
  %(prog)s --dir1 folder1 --dir2 folder2 --metric all --workers 4
  
相似度指标说明：
  cosine: 余弦相似度（范围: -1 到 1，越大越相似）
  euclidean: 欧几里得距离（范围: 0 到 ∞，越小越相似）
  manhattan: 曼哈顿距离（范围: 0 到 ∞，越小越相似）
  dot: 点积（值越大通常越相似）
  pearson: 皮尔逊相关系数（范围: -1 到 1，越大越相似）
  all: 计算所有指标
        """
    )
    
    parser.add_argument(
        "--dir1", "-d1",
        type=str,
        required=True,
        help="第一个文件夹路径"
    )
    parser.add_argument(
        "--dir2", "-d2",
        type=str,
        required=True,
        help="第二个文件夹路径"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./compare_output",
        help="输出目录（默认: ./compare_output）"
    )
    parser.add_argument(
        "--ext", "-e",
        type=str,
        default=None,
        help="文件扩展名过滤（例如：.md, .txt），默认不过滤"
    )
    parser.add_argument(
        "--metric", "-m",
        type=str,
        nargs="+",
        choices=['cosine', 'euclidean', 'manhattan', 'dot', 'pearson', 'all'],
        default=['cosine'],
        help="相似度计算指标（可多选，默认: cosine）"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="最大并发线程数（默认: 1，即串行执行）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="嵌入模型名称（覆盖环境变量 OPENAI_EMBEDDING_MODEL）"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.workers < 1:
        print(f"错误: 并发线程数必须 >= 1，当前值: {args.workers}")
        sys.exit(1)
    
    # 确定要计算的指标
    metrics_to_compute = args.metric
    if 'all' in metrics_to_compute:
        metrics_to_compute = ['cosine', 'euclidean', 'manhattan', 'dot', 'pearson']
    
    # 创建比较器并运行
    try:
        comparator = FolderComparator(
            dir1=args.dir1,
            dir2=args.dir2,
            output_dir=args.output_dir,
            file_ext=args.ext,
            metrics=metrics_to_compute,
            max_workers=args.workers,
            embedding_model=args.model
        )
        comparator.run()
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


