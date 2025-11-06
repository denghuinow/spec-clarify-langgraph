# 批量处理说明

## 功能说明

`batch_process.py` 脚本用于批量处理 `summary_5_10` 目录中的所有文件（通过调用 `srs_generator.py`），执行以下操作：

1. **记录完整日志**：为每个文件保存完整的执行日志（JSON格式），包括：
   - LLM交互记录
   - 需求清单变化
   - 评分记录
   - 相似度计算结果

2. **保存SRS文档**：为每个文件保存最终生成的SRS文档（Markdown格式）

3. **聚合相似度数据**：收集所有文件的相似度数据，生成CSV和Excel表格

## 使用方法

### 基本用法

```bash
uv run python batch_process.py
```

默认参数：
- `--user-input-dir`: `../srs-docs/summary_5_10`
- `--reference-srs-dir`: `../srs-docs/req_md`
- `--output-dir`: `./batch_output`
- `--max-iterations`: `5`

### 自定义参数

```bash
uv run python batch_process.py \
    --user-input-dir /path/to/user/inputs \
    --reference-srs-dir /path/to/reference/srs \
    --output-dir /path/to/output \
    --max-iterations 5
```

## 输出结构

```
batch_output/
├── srs_output/          # 生成的SRS文档
│   ├── 0000 - cctns.pdf.md_srs.md
│   ├── 0000 - gamma j.pdf.md_srs.md
│   └── ...
├── logs/                # 完整执行日志（JSON格式）
│   ├── 0000 - cctns.pdf.md_log.json
│   ├── 0000 - gamma j.pdf.md_log.json
│   └── ...
├── similarity_table.csv # 相似度数据表格（CSV格式）
├── similarity_table.xlsx # 相似度数据表格（Excel格式）
└── batch_summary.json   # 批量处理摘要
```

## 输出文件说明

### 1. SRS文档 (`srs_output/`)
每个文件对应一个生成的SRS文档，文件名格式：`{原文件名}_srs.md`

### 2. 日志文件 (`logs/`)
每个文件对应一个完整的执行日志，包含：
- 文件信息
- 时间戳
- 需求清单（req_list）
- 评分记录（scores）
- 冻结和移除的需求ID
- LLM交互记录（完整的输入输出）
- 相似度计算结果

### 3. 相似度表格 (`similarity_table.csv` / `similarity_table.xlsx`)
包含以下列：
- 文件：文件名
- 迭代轮数：实际执行的迭代次数
- 需求数量：最终需求清单中的需求数量
- 冻结数量：被冻结的需求数量
- 移除数量：被移除的需求数量
- 生成文档维度：生成的SRS文档的嵌入向量维度
- 参考文档维度：参考SRS文档的嵌入向量维度
- 用户输入维度：用户输入的嵌入向量维度
- 生成vs参考_相似度：生成文档与参考文档的余弦相似度
- 生成vs参考_距离：生成文档与参考文档的距离（1 - 相似度）
- 用户vs参考_相似度：用户输入与参考文档的余弦相似度
- 用户vs参考_距离：用户输入与参考文档的距离（1 - 相似度）

### 4. 处理摘要 (`batch_summary.json`)
包含：
- 处理时间戳
- 总文件数
- 成功/失败数量
- 每个文件的处理结果

## 注意事项

1. **处理时间**：由于需要调用LLM API，批量处理可能需要较长时间。建议在后台运行或使用screen/tmux。

2. **API成本**：每个文件都会进行多次LLM调用，请注意API使用成本。

3. **错误处理**：如果某个文件处理失败，脚本会继续处理其他文件，并在最终摘要中记录错误信息。

4. **输出目录**：如果输出目录已存在，新文件会覆盖旧文件。建议使用不同的输出目录或先备份。

## 查看进度

脚本会在控制台输出处理进度和日志信息。如果脚本在后台运行，可以通过以下方式查看：

```bash
# 查看输出目录中的文件数量
ls -l batch_output/srs_output/ | wc -l
ls -l batch_output/logs/ | wc -l

# 查看最新的日志文件
ls -lt batch_output/logs/ | head -5

# 查看处理摘要（处理完成后）
cat batch_output/batch_summary.json
```

## 示例

处理完成后，可以查看相似度表格：

```bash
# 查看CSV表格
head batch_output/similarity_table.csv

# 或在Python中分析
import pandas as pd
df = pd.read_csv('batch_output/similarity_table.csv')
print(df.describe())
```

