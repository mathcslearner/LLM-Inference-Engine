"""tiny-llama fixture: random-weight 2-layer Llama checkpoint + activations.

Emits tests/fixtures/models/tiny-llama/ per docs/design/model-loading.md §7.1:
config.json, model.safetensors, a tensor-identical 2-shard copy, and
expected/ activation goldens (fp32 forward on a fixed input) + meta.json.
"""

from . import common

SEED = 20260808
HIDDEN = 64
INTERMEDIATE = 176
LAYERS = 2
HEADS = 4
KV_HEADS = 2
VOCAB = 512
MAX_POS = 128
ROPE_THETA = 10000.0
RMS_EPS = 1e-5

# Fixed prompt for the activation goldens (arbitrary ids < VOCAB, fixed
# forever — changing them invalidates every downstream golden comparison).
INPUT_IDS = [1, 57, 300, 4, 128, 499, 42, 42, 7, 260, 15, 511, 0, 33, 210, 5]

# The committed config.json, hand-authored (config.to_dict() would embed a
# transformers_version stamp). LlamaConfig is built from this same dict so
# the checkpoint and config can never drift apart.
CONFIG_JSON = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": HIDDEN,
    "intermediate_size": INTERMEDIATE,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS,
    "vocab_size": VOCAB,
    "max_position_embeddings": MAX_POS,
    "rope_theta": ROPE_THETA,
    "rms_norm_eps": RMS_EPS,
    "hidden_act": "silu",
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "tiny-llama", help="random-weight tiny Llama checkpoint + activation goldens"
    )
    p.add_argument("--out", default=str(common.DEFAULT_FIXTURES_DIR))
    p.set_defaults(func=run)


def _build_model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    common.setup_torch_determinism(SEED)
    keys = {k: v for k, v in CONFIG_JSON.items() if k not in ("architectures", "model_type")}
    config = LlamaConfig(**keys)
    model = LlamaForCausalLM(config)
    model.eval()
    return model.to(torch.bfloat16)


def _shard(state_dict: dict) -> tuple[dict, dict]:
    """Deterministic 2-way split: sorted names, cut at half the total bytes."""
    names = sorted(state_dict)
    sizes = {n: state_dict[n].numel() * state_dict[n].element_size() for n in names}
    total = sum(sizes.values())
    shards = {1: {}, 2: {}}
    weight_map = {}
    running = 0
    for name in names:
        shard_id = 1 if running < total // 2 else 2
        running += sizes[name]
        shards[shard_id][name] = state_dict[name]
        weight_map[name] = f"model-{shard_id:05d}-of-00002.safetensors"
    assert shards[1] and shards[2], "degenerate shard split"
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    return shards, index


def _record_activations(model, out_dir) -> None:
    import torch

    acts = {}
    hooks = []

    def grab(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            acts[name] = tensor.detach().clone()

        return hook

    fp32 = model.float()
    hooks.append(fp32.model.embed_tokens.register_forward_hook(grab("embeddings")))
    for i, layer in enumerate(fp32.model.layers):
        hooks.append(layer.register_forward_hook(grab(f"layers.{i}")))
    hooks.append(fp32.model.norm.register_forward_hook(grab("final_norm")))

    input_ids = torch.tensor([INPUT_IDS], dtype=torch.int64)
    with torch.no_grad():
        output = fp32(input_ids=input_ids)
    for h in hooks:
        h.remove()

    acts["logits"] = output.logits.detach().clone()
    acts["input_ids"] = input_ids.to(torch.int32)
    expected = sorted([f"layers.{i}" for i in range(LAYERS)] + ["embeddings", "final_norm", "logits", "input_ids"])
    assert sorted(acts) == expected, f"activation capture mismatch: {sorted(acts)}"
    common.save_safetensors(acts, out_dir / "activations.safetensors")

    common.dump_json(
        {
            "description": "fp32 forward of tiny-llama (bf16 weights cast to fp32) "
            "on input_ids; tensors: input_ids [1,seq] i32, embeddings/layers.{i}/"
            "final_norm [1,seq,hidden] f32, logits [1,seq,vocab] f32",
            "input_ids": INPUT_IDS,
            "seed": SEED,
            "regenerate": "tools/.venv/bin/python -m gen_fixtures tiny-llama",
            "generator_versions": common.generator_versions(),
        },
        out_dir / "meta.json",
    )


def run(args) -> None:
    import pathlib

    out = pathlib.Path(args.out) / "models" / "tiny-llama"
    model = _build_model()
    state_dict = {k: v for k, v in model.state_dict().items()}

    common.dump_json(CONFIG_JSON, out / "config.json")
    common.save_safetensors(state_dict, out / "model.safetensors")

    shards, index = _shard(state_dict)
    common.dump_json(index, out / "sharded" / "model.safetensors.index.json")
    for shard_id, tensors in shards.items():
        common.save_safetensors(
            tensors, out / "sharded" / f"model-{shard_id:05d}-of-00002.safetensors"
        )

    _record_activations(model, out / "expected")
    common.check_size_budget(out, 5_000_000, "models/tiny-llama")
    print(f"  wrote {out}")
