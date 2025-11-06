# -*- coding: utf-8 -*-
"""
文本相似度计算工具
------------------
独立脚本，用于比较两个文本文件的相似度（使用 OpenAI 嵌入向量）

支持的相似度算法：
    - cosine: 余弦相似度（默认）
    - euclidean: 欧几里得距离
    - manhattan: 曼哈顿距离
    - dot: 点积
    - pearson: 皮尔逊相关系数

使用方法：
    python text_similarity.py file1.txt file2.txt
    python text_similarity.py --file1 file1.txt --file2 file2.txt
    python text_similarity.py --text1 "文本内容1" --text2 "文本内容2"
    python text_similarity.py file1.txt file2.txt --metric cosine euclidean
    python text_similarity.py file1.txt file2.txt --metric all
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from openai import OpenAI

# 加载 .env 文件
from dotenv import load_dotenv

load_dotenv()


def read_text_file(file_path: str) -> str:
    """读取 UTF-8 文本文件内容"""
    resolved_path = os.path.expanduser(file_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"文件不存在: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as f:
        return f.read()


def get_embeddings(text_a: str, text_b: str, embedding_model: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    获取两个文本的嵌入向量
    
    Args:
        text_a: 第一个文本
        text_b: 第二个文本
        embedding_model: 嵌入模型名称，默认使用 OPENAI_EMBEDDING_MODEL 环境变量或 "text-embedding-3-large"
    
    Returns:
        包含两个嵌入向量的元组 (embedding_a, embedding_b)
    """
    # 初始化OpenAI客户端
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 环境变量未设置")
    
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
    
    embedding_model = embedding_model or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    
    # 将两个文本作为单独的输入获取嵌入向量
    texts = [text_a, text_b]
    
    print(f"正在使用模型 '{embedding_model}' 计算嵌入向量...", flush=True)
    response = client.embeddings.create(
        model=embedding_model,
        input=texts
    )
    
    # 提取嵌入向量
    embedding_a = np.array(response.data[0].embedding)
    embedding_b = np.array(response.data[1].embedding)
    
    return embedding_a, embedding_b


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray, normalized: bool = False) -> float:
    """
    计算余弦相似度
    
    Args:
        embedding_a: 第一个嵌入向量
        embedding_b: 第二个嵌入向量
        normalized: 是否已归一化，默认 False（函数内部会进行归一化）
    
    Returns:
        余弦相似度分数（范围：-1 到 1，接近 1 表示更相似）
    """
    if not normalized:
        embedding_a = embedding_a / np.linalg.norm(embedding_a)
        embedding_b = embedding_b / np.linalg.norm(embedding_b)
    
    similarity = float(np.dot(embedding_a, embedding_b))
    return similarity


def euclidean_distance(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    计算欧几里得距离（L2距离）
    
    Args:
        embedding_a: 第一个嵌入向量
        embedding_b: 第二个嵌入向量
    
    Returns:
        欧几里得距离（值越小表示越相似，0 表示完全相同）
    """
    distance = float(np.linalg.norm(embedding_a - embedding_b))
    return distance


def manhattan_distance(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    计算曼哈顿距离（L1距离）
    
    Args:
        embedding_a: 第一个嵌入向量
        embedding_b: 第二个嵌入向量
    
    Returns:
        曼哈顿距离（值越小表示越相似，0 表示完全相同）
    """
    distance = float(np.sum(np.abs(embedding_a - embedding_b)))
    return distance


def dot_product(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    计算点积
    
    Args:
        embedding_a: 第一个嵌入向量
        embedding_b: 第二个嵌入向量
    
    Returns:
        点积值（值越大通常表示越相似，但取决于向量大小）
    """
    product = float(np.dot(embedding_a, embedding_b))
    return product


def pearson_correlation(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    计算皮尔逊相关系数
    
    Args:
        embedding_a: 第一个嵌入向量
        embedding_b: 第二个嵌入向量
    
    Returns:
        皮尔逊相关系数（范围：-1 到 1，接近 1 表示更相似）
    """
    # 计算均值
    mean_a = np.mean(embedding_a)
    mean_b = np.mean(embedding_b)
    
    # 中心化
    centered_a = embedding_a - mean_a
    centered_b = embedding_b - mean_b
    
    # 计算相关系数
    numerator = np.sum(centered_a * centered_b)
    denominator = np.sqrt(np.sum(centered_a ** 2) * np.sum(centered_b ** 2))
    
    if denominator == 0:
        return 0.0
    
    correlation = float(numerator / denominator)
    return correlation


def compute_similarity_metrics(
    text_a: str, 
    text_b: str, 
    embedding_model: str | None = None,
    metrics: list[str] | None = None
) -> dict[str, float]:
    """
    计算多种相似度指标
    
    Args:
        text_a: 第一个文本
        text_b: 第二个文本
        embedding_model: 嵌入模型名称
        metrics: 要计算的指标列表，如果为 None 则计算所有指标
                可选值: 'cosine', 'euclidean', 'manhattan', 'dot', 'pearson'
    
    Returns:
        包含各种相似度指标的字典
    """
    # 获取嵌入向量
    embedding_a, embedding_b = get_embeddings(text_a, text_b, embedding_model)
    
    # 默认计算所有指标
    if metrics is None:
        metrics = ['cosine', 'euclidean', 'manhattan', 'dot', 'pearson']
    
    results: dict[str, float] = {}
    
    # 计算各种相似度指标
    if 'cosine' in metrics:
        results['cosine'] = cosine_similarity(embedding_a, embedding_b)
    
    if 'euclidean' in metrics:
        results['euclidean'] = euclidean_distance(embedding_a, embedding_b)
    
    if 'manhattan' in metrics:
        results['manhattan'] = manhattan_distance(embedding_a, embedding_b)
    
    if 'dot' in metrics:
        results['dot'] = dot_product(embedding_a, embedding_b)
    
    if 'pearson' in metrics:
        results['pearson'] = pearson_correlation(embedding_a, embedding_b)
    
    return results


def compute_direct_distance(text_a: str, text_b: str, embedding_model: str | None = None) -> float:
    """
    直接比较两个文本的向量距离，不进行切块（保持向后兼容）
    参考实现：使用 OpenAI 客户端直接调用 embeddings API，进行 L2 归一化后计算点积
    返回：余弦相似度（范围：-1 到 1，通常接近 1 表示更相似）
    
    Args:
        text_a: 第一个文本
        text_b: 第二个文本
        embedding_model: 嵌入模型名称，默认使用 OPENAI_EMBEDDING_MODEL 环境变量或 "text-embedding-3-large"
    
    Returns:
        余弦相似度分数（float）
    """
    embedding_a, embedding_b = get_embeddings(text_a, text_b, embedding_model)
    return cosine_similarity(embedding_a, embedding_b)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比较两个文本文件的相似度（使用 OpenAI 嵌入向量）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  %(prog)s file1.txt file2.txt
  %(prog)s --file1 file1.txt --file2 file2.txt
  %(prog)s --text1 "文本1" --text2 "文本2"
  %(prog)s file1.txt file2.txt --metric cosine euclidean
  %(prog)s file1.txt file2.txt --metric all
  
相似度算法说明：
  cosine: 余弦相似度（范围: -1 到 1，越大越相似）
  euclidean: 欧几里得距离（范围: 0 到 ∞，越小越相似）
  manhattan: 曼哈顿距离（范围: 0 到 ∞，越小越相似）
  dot: 点积（值越大通常越相似）
  pearson: 皮尔逊相关系数（范围: -1 到 1，越大越相似）
  
环境变量：
  OPENAI_API_KEY: OpenAI API 密钥（必需）
  OPENAI_BASE_URL: OpenAI API 基础 URL（可选，用于私有部署）
  OPENAI_EMBEDDING_MODEL: 嵌入模型名称（可选，默认: text-embedding-3-large）
        """
    )
    
    # 位置参数（兼容性）
    parser.add_argument(
        "files",
        nargs="*",
        help="两个文本文件路径（位置参数，例如: file1.txt file2.txt）"
    )
    
    # 命名参数
    parser.add_argument(
        "--file1", "-f1",
        type=str,
        help="第一个文本文件路径"
    )
    parser.add_argument(
        "--file2", "-f2",
        type=str,
        help="第二个文本文件路径"
    )
    parser.add_argument(
        "--text1", "-t1",
        type=str,
        help="第一个文本内容（直接传入）"
    )
    parser.add_argument(
        "--text2", "-t2",
        type=str,
        help="第二个文本内容（直接传入）"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="嵌入模型名称（覆盖环境变量 OPENAI_EMBEDDING_MODEL）"
    )
    parser.add_argument(
        "--metric", "-M",
        type=str,
        nargs="+",
        choices=['cosine', 'euclidean', 'manhattan', 'dot', 'pearson', 'all'],
        default=['cosine'],
        help="相似度计算指标（可多选，默认: cosine）\n"
             "  cosine: 余弦相似度（范围: -1 到 1，越大越相似）\n"
             "  euclidean: 欧几里得距离（范围: 0 到 ∞，越小越相似）\n"
             "  manhattan: 曼哈顿距离（范围: 0 到 ∞，越小越相似）\n"
             "  dot: 点积（值越大通常越相似）\n"
             "  pearson: 皮尔逊相关系数（范围: -1 到 1，越大越相似）\n"
             "  all: 计算所有指标"
    )
    
    args = parser.parse_args()
    
    # 确定文本来源
    text_a: str | None = None
    text_b: str | None = None
    
    # 优先级：直接文本 > 文件参数 > 位置参数
    if args.text1 and args.text2:
        text_a = args.text1
        text_b = args.text2
        print("使用命令行直接文本作为输入", flush=True)
    elif args.file1 and args.file2:
        text_a = read_text_file(args.file1)
        text_b = read_text_file(args.file2)
        print(f"读取文件: {args.file1} 和 {args.file2}", flush=True)
    elif len(args.files) == 2:
        text_a = read_text_file(args.files[0])
        text_b = read_text_file(args.files[1])
        print(f"读取文件: {args.files[0]} 和 {args.files[1]}", flush=True)
    else:
        parser.error(
            "请提供两个文本输入。可以使用：\n"
            "  - 位置参数: %(prog)s file1.txt file2.txt\n"
            "  - 命名参数: %(prog)s --file1 file1.txt --file2 file2.txt\n"
            "  - 直接文本: %(prog)s --text1 \"文本1\" --text2 \"文本2\""
        )
    
    if not text_a or not text_b:
        parser.error("无法获取有效的文本输入")
    
    # 打印文本长度信息
    len_a = len(text_a)
    len_b = len(text_b)
    print(f"\n文本长度信息:", flush=True)
    print(f"  文本1: {len_a:,} 字符", flush=True)
    print(f"  文本2: {len_b:,} 字符", flush=True)
    print(f"  长度差: {abs(len_a - len_b):,} 字符", flush=True)
    
    # 确定要计算的指标
    metrics_to_compute = args.metric
    if 'all' in metrics_to_compute:
        metrics_to_compute = ['cosine', 'euclidean', 'manhattan', 'dot', 'pearson']
    
    # 计算相似度
    try:
        results = compute_similarity_metrics(
            text_a, 
            text_b, 
            embedding_model=args.model,
            metrics=metrics_to_compute
        )
        
        print("\n" + "=" * 60, flush=True)
        print("文本相似度计算结果", flush=True)
        print("=" * 60, flush=True)
        
        # 显示各种指标
        metric_names = {
            'cosine': '余弦相似度',
            'euclidean': '欧几里得距离',
            'manhattan': '曼哈顿距离',
            'dot': '点积',
            'pearson': '皮尔逊相关系数'
        }
        
        metric_descriptions = {
            'cosine': '范围: -1 到 1，越大越相似',
            'euclidean': '范围: 0 到 ∞，越小越相似',
            'manhattan': '范围: 0 到 ∞，越小越相似',
            'dot': '值越大通常越相似（取决于向量大小）',
            'pearson': '范围: -1 到 1，越大越相似'
        }
        
        for metric_key in metrics_to_compute:
            if metric_key in results:
                value = results[metric_key]
                name = metric_names.get(metric_key, metric_key)
                desc = metric_descriptions.get(metric_key, '')
                print(f"{name}: {value:.6f}  {desc}", flush=True)
        
        print("=" * 60, flush=True)
        
        # 返回退出码（0 表示成功）
        sys.exit(0)
        
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"计算相似度时发生错误: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

