"""qwen2-weight-names fixture: real-checkpoint name/shape inventory, no bytes.

Fetches safetensors header metadata (HTTP range requests — the weights are
never downloaded) plus the config fields M4-T06's mapping tests need
(QKV biases, tied lm_head) from a pinned Qwen2 revision.
"""

import json

from . import common


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "qwen2-names", help="Qwen2 checkpoint weight-name/shape inventory"
    )
    p.add_argument("--out", default=str(common.DEFAULT_FIXTURES_DIR))
    p.set_defaults(func=run)


def run(args) -> None:
    import pathlib

    from huggingface_hub import get_safetensors_metadata, hf_hub_download

    meta = get_safetensors_metadata(common.QWEN2_REPO, revision=common.QWEN2_REVISION)
    tensors = {}
    for file_meta in meta.files_metadata.values():
        for name, info in file_meta.tensors.items():
            assert name not in tensors, f"duplicate tensor across shards: {name}"
            tensors[name] = {"dtype": info.dtype, "shape": list(info.shape)}

    config_path = hf_hub_download(
        common.QWEN2_REPO, "config.json", revision=common.QWEN2_REVISION
    )
    config = json.loads(pathlib.Path(config_path).read_text(encoding="utf-8"))
    config_fields = {
        key: config[key]
        for key in (
            "architectures",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "tie_word_embeddings",
        )
    }

    out = pathlib.Path(args.out) / "models" / "qwen2-weight-names.json"
    common.dump_json(
        {
            "repo": common.QWEN2_REPO,
            "revision": common.QWEN2_REVISION,
            "config": config_fields,
            "tensors": tensors,
        },
        out,
    )
    print(f"  wrote {out} ({len(tensors)} tensors)")
