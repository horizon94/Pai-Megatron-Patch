"""
Compare model parameters between two Megatron torch_dist checkpoints.

Usage:
    python compare_megatron_ckpts.py <path_a> <path_b>

Example:
    python compare_megatron_ckpts.py \
        /share/project/jiangxin/models/ckpt_converted/Qwen3-8B-megatron-tp2-pp2/release \
        /share/project/lx/projects/Megatron-FLM-2/resource/outputs/ckpts/exp_load_and_save_2/iter_0000001
"""

import sys
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def get_tensor_keys(path):
    reader = FileSystemReader(path)
    metadata = reader.read_metadata()
    return {k for k, v in metadata.state_dict_metadata.items()
            if isinstance(v, TensorStorageMetadata)}


def build_state_dict(path, keys):
    reader = FileSystemReader(path)
    metadata = reader.read_metadata()
    sd = {}
    for k in keys:
        size = metadata.state_dict_metadata[k].size
        sd[k] = torch.zeros(size)
    dcp.load(state_dict=sd, storage_reader=FileSystemReader(path))
    return sd


def compare(path_a, path_b):
    keys_a = get_tensor_keys(path_a)
    keys_b = get_tensor_keys(path_b)

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    common = keys_a & keys_b

    print(f"Tensor keys in A: {len(keys_a)}")
    print(f"Tensor keys in B: {len(keys_b)}")
    print(f"Common keys:      {len(common)}")
    if only_a:
        print(f"Only in A: {only_a}")
    if only_b:
        print(f"Only in B: {only_b}")

    print(f"\nLoading A: {path_a}")
    sd_a = build_state_dict(path_a, common)
    print(f"Loading B: {path_b}")
    sd_b = build_state_dict(path_b, common)

    print("\n--- Parameter comparison ---")
    max_diff_overall = 0.0
    for k in sorted(common):
        ta = sd_a[k].float()
        tb = sd_b[k].float()
        max_diff = (ta - tb).abs().max().item()
        mean_diff = (ta - tb).abs().mean().item()
        max_diff_overall = max(max_diff_overall, max_diff)
        status = "✓ same   " if max_diff < 1e-3 else "✗ CHANGED"
        print(f"  [{status}] {k}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    print(f"\nGlobal max abs diff: {max_diff_overall:.6f}")
    if max_diff_overall < 1e-3:
        print("Conclusion: parameters are IDENTICAL.")
    else:
        print("Conclusion: parameters DIFFER.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
