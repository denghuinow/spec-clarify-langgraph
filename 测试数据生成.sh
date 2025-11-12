uv run batch_process.py --output-dir ./output_i5 --version v2 --max-iterations 5 -w 80
uv run batch_process.py --output-dir ./output_i4 --version v2 --max-iterations 4 -w 80
uv run batch_process.py --output-dir ./output_i3 --version v2 --max-iterations 3 -w 80
uv run batch_process.py --output-dir ./output_i2 --version v2 --max-iterations 2 -w 80
uv run batch_process.py --output-dir ./output_i1 --version v2 --max-iterations 1 -w 80


uv run batch_process.py --output-dir ./output_no-clarify --version v2 --max-iterations 1 -w 80 --ablation-mode  no-clarify

uv run batch_process.py --output-dir ./output_no-explore-clarify --version v2 --max-iterations 1 -w 80 --ablation-mode  no-explore-clarify