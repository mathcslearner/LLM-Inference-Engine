"""Real config.json fixtures for the M4-T03 config-parser goldens.

Byte-copies the unmodified config.json of one Llama-3-family and one
Qwen2-family checkpoint at pinned revisions. The Llama side is deliberately
Llama-3.1: its `rope_scaling` block ("rope_type": "llama3") exercises the
presence side and every RopeScaling field, while the Qwen2 config exercises
absence plus the omitted-`attention_bias` per-arch default. The Llama 3.1
Community License permits redistribution with notice, so the upstream
LICENSE is byte-copied alongside (same discipline as tokenizers/llama3/).
"""

import shutil

from . import common

# (family dir, repo, revision, files to byte-copy)
SOURCES = [
    ("llama3", common.LLAMA31_REPO, common.LLAMA31_REVISION, ("config.json", "LICENSE")),
    ("qwen2", common.QWEN2_REPO, common.QWEN2_REVISION, ("config.json",)),
]


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "model-configs", help="real config.json files for config-parser goldens"
    )
    p.add_argument("--out", default=str(common.DEFAULT_FIXTURES_DIR))
    p.set_defaults(func=run)


def run(args) -> None:
    import pathlib

    from huggingface_hub import hf_hub_download

    for family, repo, revision, filenames in SOURCES:
        out = pathlib.Path(args.out) / "models" / "configs" / family
        out.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            cached = hf_hub_download(repo, filename, revision=revision)
            shutil.copyfile(cached, out / filename)  # byte-copy, never re-serialized
            print(f"  wrote {out / filename}")
