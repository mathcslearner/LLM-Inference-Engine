"""Shared determinism and serialization helpers."""

import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# Pinned hub sources. Regeneration is byte-identical because revisions are
# commit hashes, never branch names.
QWEN2_REPO = "Qwen/Qwen2-0.5B-Instruct"
QWEN2_REVISION = "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
# Ungated byte-exact mirror of meta-llama/Meta-Llama-3-8B-Instruct (the
# upstream repo is gated); ships the upstream LICENSE, which we commit
# alongside the tokenizer per the Llama 3 Community License notice terms.
LLAMA3_REPO = "NousResearch/Meta-Llama-3-8B-Instruct"
LLAMA3_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
# Ungated byte-exact mirror of meta-llama/Llama-3.1-8B-Instruct, for the
# config-parser goldens (M4-T03): the 3.1 config carries the llama3-type
# rope_scaling block that 3.0 configs lack. Ships the upstream LICENSE.
LLAMA31_REPO = "NousResearch/Meta-Llama-3.1-8B-Instruct"
LLAMA31_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"


def dump_json(obj, path: pathlib.Path) -> None:
    """Deterministic JSON: sorted keys, fixed separators, trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def save_safetensors(tensors: dict, path: pathlib.Path) -> None:
    """Deterministic safetensors: header entries in sorted-name order."""
    import safetensors.torch

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {name: tensors[name].contiguous() for name in sorted(tensors)}
    safetensors.torch.save_file(ordered, str(path))


def setup_torch_determinism(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def generator_versions() -> dict:
    import safetensors
    import tokenizers
    import torch
    import transformers

    return {
        "safetensors": safetensors.__version__,
        "tokenizers": tokenizers.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def check_size_budget(path: pathlib.Path, limit_bytes: int, label: str) -> None:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    if total > limit_bytes:
        raise SystemExit(
            f"{label}: {total} bytes exceeds the {limit_bytes}-byte budget "
            f"(tests/fixtures/README.md)"
        )
    print(f"  {label}: {total / 1e6:.2f} MB (budget {limit_bytes / 1e6:.0f} MB)")
