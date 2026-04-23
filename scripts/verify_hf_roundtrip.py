import argparse
import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


GENERATION_PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "请续写这句话：人工智能正在改变",
]


LOSS_CASES = [
    {
        "prompt": "The capital of France is",
        "completion": " Paris.",
    },
    {
        "prompt": "Python is a programming language that",
        "completion": " emphasizes readability and simplicity.",
    },
    {
        "prompt": "请续写这句话：人工智能正在改变",
        "completion": "我们的工作方式和日常生活。",
    },
]


@dataclass
class TensorDiffSummary:
    max_abs_diff: float = 0.0
    mismatched_tensors: int = 0
    compared_tensors: int = 0
    compared_elements: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate HF -> Megatron -> HF round-trip equivalence for local HF model directories."
    )
    parser.add_argument("--original", required=True, help="Path to the original HuggingFace model directory")
    parser.add_argument("--candidate", required=True, help="Path to the converted-back HuggingFace model directory")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference checks (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Torch dtype for inference checks",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Max new tokens for generation checks",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for tensor/logit/loss comparison",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for tensor/logit/loss comparison",
    )
    parser.add_argument(
        "--skip-state-dict",
        action="store_true",
        help="Skip direct checkpoint tensor comparison",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip generation and loss comparison",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for deterministic inference checks")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic PyTorch settings where supported",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional shared attention implementation passed to both models, e.g. eager/sdpa/flash_attention_2",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def configure_determinism(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True)


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    index = load_json(model_dir / "model.safetensors.index.json")
    return index["weight_map"]


def compare_state_dicts(original_dir: Path, candidate_dir: Path, atol: float, rtol: float) -> None:
    original_map = load_weight_map(original_dir)
    candidate_map = load_weight_map(candidate_dir)

    original_keys = set(original_map)
    candidate_keys = set(candidate_map)
    if original_keys != candidate_keys:
        missing = sorted(original_keys - candidate_keys)
        extra = sorted(candidate_keys - original_keys)
        raise AssertionError(
            f"State dict keys differ. Missing in candidate: {missing[:10]}, extra in candidate: {extra[:10]}"
        )

    orig_handles = {
        shard: safe_open(str(original_dir / shard), framework="pt", device="cpu")
        for shard in sorted(set(original_map.values()))
    }
    cand_handles = {
        shard: safe_open(str(candidate_dir / shard), framework="pt", device="cpu")
        for shard in sorted(set(candidate_map.values()))
    }

    summary = TensorDiffSummary()
    worst_tensor = None

    for key in sorted(original_keys):
        orig_tensor = orig_handles[original_map[key]].get_tensor(key)
        cand_tensor = cand_handles[candidate_map[key]].get_tensor(key)

        if orig_tensor.shape != cand_tensor.shape:
            raise AssertionError(f"Shape mismatch for {key}: {orig_tensor.shape} vs {cand_tensor.shape}")
        if orig_tensor.dtype != cand_tensor.dtype:
            raise AssertionError(f"Dtype mismatch for {key}: {orig_tensor.dtype} vs {cand_tensor.dtype}")

        diff = (orig_tensor.float() - cand_tensor.float()).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
        allclose = torch.allclose(orig_tensor.float(), cand_tensor.float(), atol=atol, rtol=rtol)

        summary.compared_tensors += 1
        summary.compared_elements += diff.numel()

        if max_abs_diff > summary.max_abs_diff:
            summary.max_abs_diff = max_abs_diff
            worst_tensor = key

        if not allclose:
            summary.mismatched_tensors += 1
            print(f"[state-dict] mismatch: {key}, max_abs_diff={max_abs_diff:.8g}")

    print(
        f"[state-dict] compared {summary.compared_tensors} tensors / {summary.compared_elements} elements, "
        f"mismatched_tensors={summary.mismatched_tensors}, max_abs_diff={summary.max_abs_diff:.8g}, "
        f"worst_tensor={worst_tensor}"
    )

    if summary.mismatched_tensors:
        raise AssertionError("Direct checkpoint tensor comparison failed")


def load_model_and_tokenizer(model_dir: Path, device: str, dtype_name: str, attn_implementation: str | None):
    torch_dtype = dtype_from_name(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True, "local_files_only": True}
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    else:
        model_kwargs["torch_dtype"] = "auto"
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    model.eval()
    model.to(device)
    return model, tokenizer


@torch.inference_mode()
def run_generation_checks(
    model_dir: Path,
    device: str,
    dtype_name: str,
    max_new_tokens: int,
    attn_implementation: str | None,
) -> Dict[str, Dict]:
    model, tokenizer = load_model_and_tokenizer(model_dir, device, dtype_name, attn_implementation)
    results = {}

    for prompt in GENERATION_PROMPTS:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = outputs[0, encoded["input_ids"].shape[1]:]
        results[prompt] = {
            "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
            "token_ids": new_tokens.detach().cpu().tolist(),
        }

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return results


@torch.inference_mode()
def run_loss_checks(model_dir: Path, device: str, dtype_name: str, attn_implementation: str | None) -> Dict[str, Dict]:
    model, tokenizer = load_model_and_tokenizer(model_dir, device, dtype_name, attn_implementation)
    results = {}

    for case in LOSS_CASES:
        prompt = case["prompt"]
        completion = case["completion"]
        full_text = prompt + completion

        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        full_ids = full_ids.to(device)

        attention_mask = torch.ones_like(full_ids, device=device)
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].float()
        labels = full_ids[:, 1:]

        completion_start = prompt_ids.shape[1] - 1
        completion_logits = logits[:, completion_start:, :]
        completion_labels = labels[:, completion_start:]

        token_losses = F.cross_entropy(
            completion_logits.reshape(-1, completion_logits.shape[-1]),
            completion_labels.reshape(-1),
            reduction="none",
        ).reshape(completion_labels.shape)

        mean_loss = float(token_losses.mean().item())
        results[f"{prompt} >>> {completion}"] = {
            "mean_completion_loss": mean_loss,
            "num_completion_tokens": int(completion_labels.numel()),
            "completion_logits": completion_logits.detach().cpu(),
        }

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return results


def compare_generation_results(original: Dict[str, Dict], candidate: Dict[str, Dict]) -> None:
    for prompt in GENERATION_PROMPTS:
        original_tokens = original[prompt]["token_ids"]
        candidate_tokens = candidate[prompt]["token_ids"]
        same = original_tokens == candidate_tokens
        print(f"[generation] prompt={prompt!r}")
        print(f"  original : {original[prompt]['text']!r}")
        print(f"  candidate: {candidate[prompt]['text']!r}")
        print(f"  token_match={same}")
        if not same:
            raise AssertionError(f"Generation token mismatch for prompt: {prompt!r}")


def compare_loss_results(original: Dict[str, Dict[str, float]], candidate: Dict[str, Dict[str, float]], atol: float, rtol: float) -> None:
    for name in original:
        orig_loss = original[name]["mean_completion_loss"]
        cand_loss = candidate[name]["mean_completion_loss"]
        orig_logits = original[name]["completion_logits"]
        cand_logits = candidate[name]["completion_logits"]
        diff = abs(orig_loss - cand_loss)
        same = torch.isclose(torch.tensor(orig_loss), torch.tensor(cand_loss), atol=atol, rtol=rtol).item()
        logits_same = torch.allclose(orig_logits, cand_logits, atol=atol, rtol=rtol)
        logits_diff = float((orig_logits - cand_logits).abs().max().item()) if orig_logits.numel() > 0 else 0.0
        print(
            f"[loss] case={name!r}, original={orig_loss:.8f}, candidate={cand_loss:.8f}, diff={diff:.8g}, "
            f"loss_match={same}, logits_match={logits_same}, logits_max_abs_diff={logits_diff:.8g}"
        )
        if not same:
            raise AssertionError(f"Loss mismatch for case: {name!r}")
        if not logits_same:
            raise AssertionError(f"Logits mismatch for case: {name!r}")


def main() -> None:
    args = parse_args()
    original_dir = Path(args.original).resolve()
    candidate_dir = Path(args.candidate).resolve()

    configure_determinism(args.seed, args.deterministic)

    if not args.skip_state_dict:
        compare_state_dicts(original_dir, candidate_dir, atol=args.atol, rtol=args.rtol)

    if not args.skip_inference:
        original_generation = run_generation_checks(
            original_dir,
            device=args.device,
            dtype_name=args.dtype,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
        )
        candidate_generation = run_generation_checks(
            candidate_dir,
            device=args.device,
            dtype_name=args.dtype,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
        )
        compare_generation_results(original_generation, candidate_generation)

        original_loss = run_loss_checks(
            original_dir,
            device=args.device,
            dtype_name=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        candidate_loss = run_loss_checks(
            candidate_dir,
            device=args.device,
            dtype_name=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        compare_loss_results(original_loss, candidate_loss, atol=args.atol, rtol=args.rtol)

    print("All requested checks passed.")


if __name__ == "__main__":
    main()
