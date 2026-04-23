# AGENTS.md

## Purpose
- This repo is a patch-oriented training workspace around Megatron, not a single installable Python package. The real execution flow lives in shell launchers and model/toolkit READMEs, not in a root `pyproject.toml` or `Makefile`.

## Repo map
- `megatron_patch/`: reusable patch layer over Megatron; shared arguments, training helpers, tokenizers, model code, generation, and targeted backend fixes.
- `verl_patch/`: local VERL integration code for RL training.
- `examples/`: the main runnable front door. Each model family has its own README plus launcher scripts.
- `toolkits/`: standalone utilities for data preprocessing, checkpoint conversion, and auto-configuration.
- `backends/`: required git submodules. `backends/megatron/` contains multiple pinned Megatron snapshots; `backends/rl/` contains ChatLearn and verl.
- `docker/`: base container definitions only; there is no repo-level wrapper to build or run them.

## Canonical workflow sources
- Clone with submodules: `git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git`.
- Start from the nearest model README under `examples/*/README*.md` for training, RL, evaluation, and environment setup. Root `README.md` is mostly an index, not the operational source of truth.
- Start from `toolkits/*/README.md` plus adjacent shell wrappers for preprocessing and checkpoint conversion.
- When docs and scripts differ, trust the shell wrapper or Python entrypoint.

## Agent guardrails
- Do not invent root-level commands like `make test`, `pytest`, `ruff`, `mypy`, `pre-commit`, or GitHub Actions workflows. This checkout does not define a verified repo-wide lint/test/typecheck/CI entrypoint.
- Treat `examples/*.sh` as the real launch surface. Most flows are `bash` wrappers around `torchrun` plus heavy environment wiring.
- Do not normalize backend paths across examples. Launcher scripts pin different Megatron snapshots via `PYTHONPATH` and the version matters. Verify the exact backend in the script you are editing.
  - Examples: `examples/qwen3/run_mcore_qwen3.sh` uses `backends/megatron/Megatron-LM-250624`; `examples/qwen3_next/run_mcore_qwen3.sh` uses `Megatron-LM-250908`; `toolkits/auto_configurator/run_auto_config.sh` uses `PAI-Megatron-LM-240718`.
- RL flows are separate from plain Megatron training. ChatLearn and VERL launchers pull in `backends/rl/ChatLearn` or `backends/rl/verl` and may reuse entrypoints across model folders.
- Many commands in `examples/` and `toolkits/` are large GPU jobs, not lightweight validation. Prefer focused script inspection over running training/conversion unless the user explicitly asks.

## Repo-specific gotchas
- Distributed checkpoint conversion lives under `toolkits/distributed_checkpoints_convertor/`. It expects training checkpoints saved with `--ckpt-format torch_dist`; some older example scripts still use plain `torch`, so verify before assuming compatibility.
- The distributed convertor README says tokenizer/config files are not copied automatically; check the model-specific wrapper script because some wrappers manually copy JSON/tokenizer artifacts into `SAVE_DIR`.
- Version-specific backend fixes are documented under `megatron_patch/fixes/`. In particular, `optimizer_offloading/README.md` and `yarn_args/README.md` both target `Megatron-LM-250328`, not all Megatron versions.
- Shared training behavior is centralized in `megatron_patch/template/helper.py`; check it before changing dataset handling, sequence packing, or loss behavior across examples.

## Validation
- For command changes, verify the nearest README, shell launcher, and Python entrypoint together.
- For code changes, validate against the exact backend version that the touched launcher exports in `PYTHONPATH`.
- Keep root-level guidance short; model-specific setup belongs in the local `examples/*/README*.md` or toolkit docs, not here.
