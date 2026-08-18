"""tiny-qwen2 fixture: random-weight 2-layer Qwen2 checkpoint + activations.

The M5-T10 companion to tiny-llama: a tiny `Qwen2ForCausalLM` that exercises the
Qwen-family wiring on the *shared* M5 modules (docs/design/model-execution.md
§4.1, §12). The differences from tiny-llama that matter to the engine:

  * **q/k/v projection biases** (`attention_bias` — Qwen2's per-arch default;
    deliberately OMITTED from config.json so the config parser's default path is
    what supplies it), o_proj bias-free.
  * **`head_dim` decoupled from `hidden_size / num_heads`** (24 ≠ 64/4 = 16), the
    config §3.1 flags but tiny-llama never covers.
  * **tied embeddings** — `lm_head.weight` is the embed table; safetensors refuses
    to serialize the shared storage, so the checkpoint omits it (exactly like a
    real Qwen2-0.5B checkpoint) and the loader's tied-alias path supplies it.
  * Qwen2 config defaults: `rope_theta` 1e6, `rms_norm_eps` 1e-6, no BOS.

Emits tests/fixtures/models/tiny-qwen2/: config.json, model.safetensors, and
expected/ activation goldens (fp32 forward on a fixed input) + meta.json. No
sharded copy — sharding is an M4 concern already covered by tiny-llama.
"""

from . import common

SEED = 20260818
HIDDEN = 64
INTERMEDIATE = 176
LAYERS = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 24  # decoupled: != HIDDEN // HEADS (== 16)
VOCAB = 512
MAX_POS = 128
ROPE_THETA = 1000000.0  # Qwen2 default (vs tiny-llama's 1e4)
RMS_EPS = 1e-6  # Qwen2 default (vs tiny-llama's 1e-5)

# Fixed prompt for the activation goldens (arbitrary ids < VOCAB, fixed forever —
# changing them invalidates every downstream golden comparison). No BOS: Qwen's
# tokenizer inserts none, so the id stream need not start with a BOS id.
INPUT_IDS = [17, 402, 5, 233, 88, 300, 1, 461, 39, 128, 274, 6, 500, 42, 213, 90]

# The committed config.json, hand-authored (config.to_dict() would embed a
# transformers_version stamp). Qwen2Config is built from this same dict so the
# checkpoint and config can never drift apart. `attention_bias` is intentionally
# ABSENT: the C++ parser defaults it to true for Qwen2, and this fixture proves
# that default end-to-end. `bos_token_id` is absent (Qwen has no BOS).
CONFIG_JSON = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": HIDDEN,
    "intermediate_size": INTERMEDIATE,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "vocab_size": VOCAB,
    "max_position_embeddings": MAX_POS,
    "rope_theta": ROPE_THETA,
    "rms_norm_eps": RMS_EPS,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
    "use_sliding_window": False,
    "max_window_layers": LAYERS,
    "torch_dtype": "bfloat16",
    "eos_token_id": 3,
}


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "tiny-qwen2", help="random-weight tiny Qwen2 checkpoint + activation goldens"
    )
    p.add_argument("--out", default=str(common.DEFAULT_FIXTURES_DIR))
    p.set_defaults(func=run)


def _build_model():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    common.setup_torch_determinism(SEED)
    keys = {k: v for k, v in CONFIG_JSON.items() if k not in ("architectures", "model_type")}
    config = Qwen2Config(**keys)
    model = Qwen2ForCausalLM(config)

    # HF's _init_weights zero-inits every Linear bias, so a freshly built Qwen2
    # has all-zero q/k/v biases — which would make the bias path invisible in the
    # golden (dropping the biases wouldn't change a thing). Fill them with fixed
    # small-magnitude noise so the golden genuinely exercises bias addition and
    # the "biases are load-bearing" test is meaningful. A dedicated generator
    # keeps this independent of the model-init RNG stream.
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for layer in model.model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                         layer.self_attn.v_proj):
                proj.bias.copy_(torch.randn(proj.bias.shape, generator=gen) * 0.5)

    model.eval()
    return model.to(torch.bfloat16)


def _state_dict_for_save(model) -> dict:
    """State dict minus the tied lm_head (shared storage safetensors refuses)."""
    state_dict = {k: v for k, v in model.state_dict().items()}
    # tie_word_embeddings=True aliases lm_head.weight onto embed_tokens.weight;
    # they share memory, so save_file would raise. Drop it — the loader
    # reconstitutes lm_head from the embed table (weight_map tied-alias path).
    if "lm_head.weight" in state_dict:
        embed = state_dict["model.embed_tokens.weight"]
        lm_head = state_dict["lm_head.weight"]
        assert lm_head.data_ptr() == embed.data_ptr(), "lm_head not tied to embed"
        del state_dict["lm_head.weight"]
    return state_dict


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
    expected = sorted(
        [f"layers.{i}" for i in range(LAYERS)]
        + ["embeddings", "final_norm", "logits", "input_ids"]
    )
    assert sorted(acts) == expected, f"activation capture mismatch: {sorted(acts)}"
    common.save_safetensors(acts, out_dir / "activations.safetensors")

    common.dump_json(
        {
            "description": "fp32 forward of tiny-qwen2 (bf16 weights cast to fp32) "
            "on input_ids; tensors: input_ids [1,seq] i32, embeddings/layers.{i}/"
            "final_norm [1,seq,hidden] f32, logits [1,seq,vocab] f32. Qwen2: q/k/v "
            "biases, head_dim=24 (decoupled), tied embeddings, rope_theta=1e6.",
            "input_ids": INPUT_IDS,
            "seed": SEED,
            "regenerate": "tools/.venv/bin/python -m gen_fixtures tiny-qwen2",
            "generator_versions": common.generator_versions(),
        },
        out_dir / "meta.json",
    )


def run(args) -> None:
    import pathlib

    out = pathlib.Path(args.out) / "models" / "tiny-qwen2"
    model = _build_model()

    common.dump_json(CONFIG_JSON, out / "config.json")
    common.save_safetensors(_state_dict_for_save(model), out / "model.safetensors")

    _record_activations(model, out / "expected")
    common.check_size_budget(out, 5_000_000, "models/tiny-qwen2")
    print(f"  wrote {out}")
