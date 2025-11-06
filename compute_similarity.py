# -*- coding: utf-8 -*-
"""
文本相似度计算工具（SRS专用）
----------------------------
独立脚本，用于计算SRS文档之间的相似度（使用 OpenAI 嵌入向量）

支持的计算类型：
    - 生成SRS vs 参考SRS
    - 用户输入 vs 参考SRS

使用方法：
    # 单文件处理
    python compute_similarity.py --generated-srs generated.md --reference-srs reference.md
    python compute_similarity.py --user-input user.txt --reference-srs reference.md
    python compute_similarity.py --generated-srs generated.md --reference-srs reference.md --output result.json
    
注意：批量处理请使用 batch_process.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple

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


def get_embeddings(text_a: str, text_b: str, embedding_model: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
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


def compute_direct_distance(text_a: str, text_b: str, embedding_model: Optional[str] = None) -> float:
    """
    直接比较两个文本的向量距离，不进行切块
    使用 OpenAI 客户端直接调用 embeddings API，进行 L2 归一化后计算点积
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


def compute_single_similarity(
    generated_srs: Optional[str] = None,
    user_input: Optional[str] = None,
    reference_srs: str = None,
    embedding_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    计算单个相似度结果
    
    Args:
        generated_srs: 生成的SRS文档（可选）
        user_input: 用户输入文本（可选）
        reference_srs: 参考SRS文档（必需）
        embedding_model: 嵌入模型名称
        
    Returns:
        包含相似度结果的字典
    """
    result = {
        "generated_vs_reference": None,
        "user_input_vs_reference": None
    }
    
    if not reference_srs:
        raise ValueError("reference_srs 是必需的")
    
    # 计算生成SRS vs 参考SRS
    if generated_srs:
        try:
            sim_gr = compute_direct_distance(
                generated_srs,
                reference_srs,
                embedding_model=embedding_model
            )
            result["generated_vs_reference"] = sim_gr
        except Exception as e:
            result["generated_vs_reference_error"] = str(e)
    
    # 计算用户输入 vs 参考SRS
    if user_input:
        try:
            sim_ur = compute_direct_distance(
                user_input,
                reference_srs,
                embedding_model=embedding_model
            )
            result["user_input_vs_reference"] = sim_ur
        except Exception as e:
            result["user_input_vs_reference_error"] = str(e)
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算SRS文档之间的相似度（使用 OpenAI 嵌入向量）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 单文件处理：生成SRS vs 参考SRS
  %(prog)s --generated-srs generated.md --reference-srs reference.md
  
  # 单文件处理：用户输入 vs 参考SRS
  %(prog)s --user-input user.txt --reference-srs reference.md
  
  # 单文件处理：同时计算两种相似度
  %(prog)s --generated-srs generated.md --user-input user.txt --reference-srs reference.md --output result.json
  
环境变量：
  OPENAI_API_KEY: OpenAI API 密钥（必需）
  OPENAI_BASE_URL: OpenAI API 基础 URL（可选，用于私有部署）
  OPENAI_EMBEDDING_MODEL: 嵌入模型名称（可选，默认: text-embedding-3-large）
  
注意：批量处理请使用 batch_process.py
        """
    )
    
    # 单文件处理参数
    parser.add_argument(
        "--generated-srs", "-g",
        type=str,
        help="生成的SRS文档文件路径"
    )
    parser.add_argument(
        "--user-input", "-u",
        type=str,
        help="用户输入文本文件路径"
    )
    parser.add_argument(
        "--reference-srs", "-r",
        type=str,
        help="参考SRS文档文件路径"
    )
    
    # 直接文本输入
    parser.add_argument(
        "--text-generated",
        type=str,
        help="生成的SRS文档内容（直接传入）"
    )
    parser.add_argument(
        "--text-user-input",
        type=str,
        help="用户输入文本内容（直接传入）"
    )
    parser.add_argument(
        "--text-reference",
        type=str,
        help="参考SRS文档内容（直接传入）"
    )
    
    # 其他参数
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="嵌入模型名称（覆盖环境变量 OPENAI_EMBEDDING_MODEL）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出JSON文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    # 单文件处理模式
    generated_srs = None
    user_input = None
    reference_srs = None
    
    # 确定文本来源（优先级：直接文本 > 文件路径）
    if args.text_generated:
        generated_srs = args.text_generated
    elif args.generated_srs:
        generated_srs = read_text_file(args.generated_srs)
    
    if args.text_user_input:
        user_input = args.text_user_input
    elif args.user_input:
        user_input = read_text_file(args.user_input)
    
    if args.text_reference:
        reference_srs = args.text_reference
    elif args.reference_srs:
        reference_srs = read_text_file(args.reference_srs)
    
    if not reference_srs:
        parser.error("必须提供参考SRS文档（--reference-srs 或 --text-reference）")
    
    if not generated_srs and not user_input:
        parser.error("必须提供至少一个：生成的SRS（--generated-srs）或用户输入（--user-input）")
    
    # 计算相似度
    try:
        result = compute_single_similarity(
            generated_srs=generated_srs,
            user_input=user_input,
            reference_srs=reference_srs,
            embedding_model=args.model
        )
        
        # 打印结果
        print("\n" + "=" * 60, flush=True)
        print("相似度计算结果", flush=True)
        print("=" * 60, flush=True)
        
        if result.get("generated_vs_reference") is not None:
            print(f"生成SRS vs 参考SRS: {result['generated_vs_reference']:.6f}", flush=True)
        elif "generated_vs_reference_error" in result:
            print(f"生成SRS vs 参考SRS: 计算失败 - {result['generated_vs_reference_error']}", flush=True)
        
        if result.get("user_input_vs_reference") is not None:
            print(f"用户输入 vs 参考SRS: {result['user_input_vs_reference']:.6f}", flush=True)
        elif "user_input_vs_reference_error" in result:
            print(f"用户输入 vs 参考SRS: 计算失败 - {result['user_input_vs_reference_error']}", flush=True)
        
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
        print(f"计算相似度时发生错误: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

