# Qwen3-8B 检查点转换说明

## 环境

- Python 环境：`/share/project/lx/envs/new_megatron`
- 转换脚本目录：`toolkits/distributed_checkpoints_convertor/`（以下命令均在此目录下执行）

## HuggingFace → Megatron

```bash
export PYTHONPATH=/share/project/jiangxin/projects/Megatron-FLM-2
export MODEL_PARALLEL_ARGS="--tensor-model-parallel-size 2 --pipeline-model-parallel-size 2"
bash scripts/qwen3/run_8xH20.sh \
    8B \
    /share/project/lx/models/Qwen3-8B \
    /share/project/jiangxin/models/ckpt_converted/Qwen3-8B-megatron-tp2-pp2 \
    false \
    true \
    bf16
```

## Megatron → HuggingFace

```bash
export PYTHONPATH=/share/project/jiangxin/projects/Megatron-FLM-2
export MODEL_PARALLEL_ARGS="--tensor-model-parallel-size 2 --pipeline-model-parallel-size 2"
bash scripts/qwen3/run_8xH20.sh \
    8B \
    /share/project/lx/projects/Megatron-FLM-2/resource/outputs/ckpts/exp_load_and_save_2 \
    /share/project/lx/models/Qwen3-8B-to-mgt-lx-train-0lr-convert-back-0423 \
    true \
    true \
    bf16 \
    /share/project/lx/models/Qwen3-8B
```

> **注意**：mcore2hf 时 `LOAD_DIR` 必须传 ckpt 的**父目录**（即包含 `latest_checkpointed_iteration.txt` 的目录），不能直接传 `iter_XXXXXXX` 子目录。否则 `load_checkpoint` 找不到 tracker 文件，会从随机初始化转换，导致参数错误。

## 参数说明

`run_8xH20.sh` 位置参数顺序：

| 位置 | 参数 | 说明 |
|------|------|------|
| $1 | MODEL_SIZE | 模型规格，如 `8B` |
| $2 | LOAD_DIR | 源检查点路径 |
| $3 | SAVE_DIR | 目标保存路径 |
| $4 | MG2HF | `true` = mcore→HF，`false` = HF→mcore |
| $5 | USE_CUDA | 是否使用 GPU 加速转换 |
| $6 | PR | 精度，`bf16` 或 `fp16` |
| $7 | HF_DIR | （仅 mcore2hf）原始 HF 模型路径，用于复制 tokenizer/config |
