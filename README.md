# kaggle-1
交易预测

## Qwen3.5-0.8B 端侧小型化（PyTorch，4-bit NF4）

新增脚本：`compress_qwen_4bit.py`

### 依赖安装

```bash
pip install torch transformers accelerate bitsandbytes sentencepiece
```

### 运行方式

```bash
python compress_qwen_4bit.py \
  --model-dir /absolute/path/to/Qwen3.5-0.8B \
  --out-dir ./qwen3_5_0_8b_4bit \
  --max-new-tokens 128
```

说明：
- 使用 `load_in_4bit=True` + NF4（bitsandbytes）进行低比特量化；
- 使用 `torch.compile(mode="reduce-overhead")` 做图级优化（若当前环境支持）；
- 默认 `max_new_tokens=128`，4GB 显存建议控制在 64~256。
