"""Causal-attention goldens: per-layer attention I/O + intermediates (M5-T05).

Emits tests/fixtures/models/tiny-llama/expected/attention.safetensors — HF's own
LlamaAttention output, the oracle for cpu::attention and the `Attention` module
(docs/design/model-execution.md §4.2, §3.1-3.2, §6.3, §12).

The tiny-llama fp32 model (bf16 checkpoint cast to fp32 — the same effective
weights the C++ loader widens per element) is driven at the *attention-module*
level: a chosen fp32 hidden state `x [T, E]` is fed straight into
`model.model.layers[i].self_attn`, with an explicit additive causal mask and the
rotary cos/sin at the given absolute positions. A DynamicCache pre-populated with
`past_k/past_v [Hkv, P, d]` produces the prefill-continuation (P>0) cases; an
empty cache produces prefill-from-empty (P=0). GQA is inherent — tiny-llama has
H=4 query heads / Hkv=2 kv heads.

For each case we store both:
  * the module golden: x [T,E] -> out [T,E] (post o_proj), validating the whole
    `Attention::forward`.
  * the op golden: q_rot [T,H,d] (post-RoPE queries, token-major), k_all/v_all
    [Hkv, P+T, d] (the accumulated head-major cache view), ctx [T,H,d] (the
    attention output before o_proj) — validating `cpu::attention` alone, so a
    regression localizes to the op vs the projections/RoPE/cache wiring.

These intermediates are captured from HF's real eager attention (a registered
capture wrapper delegating to `eager_attention_forward`), so they are HF's
ground truth, not a re-derivation. HF computes attention in fp32 here; the
reference reduces in its own ascending order, so agreement is Class T within the
tolerances stated in attention_test.cpp.

Cases (layer, P, T):
  l{0,1}_prefill_empty     : P=0, T=8, positions 0..7
  l{0,1}_prefill_continue  : P=5, T=6, positions 5..10
  l0_decode                : P=7, T=1, position 7

Both layers appear because their weights differ; the decode case exercises the
T==1 (GEMV-shape) path. This is its own subcommand (tiny-llama-attention)
writing its own file, so the committed ops/rope bytes (M5-T02/T03/T04) are
untouched.
"""

import pathlib

from . import common, tiny_llama

SEED = 20260819

# (name, layer_idx, P, T). positions = range(P, P+T).
CASES = [
    ("l0_prefill_empty", 0, 0, 8),
    ("l1_prefill_empty", 1, 0, 8),
    ("l0_prefill_continue", 0, 5, 6),
    ("l1_prefill_continue", 1, 5, 6),
    ("l0_decode", 0, 7, 1),
]


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "tiny-llama-attention",
        help="causal-attention goldens for the CPU reference (M5-T05)",
    )
    p.add_argument("--out", default=str(common.DEFAULT_FIXTURES_DIR))
    p.set_defaults(func=run)


def _capture_interface():
    """Register a capture wrapper over HF eager attention; return its key + dict.

    The wrapper delegates to the real eager_attention_forward and records the
    query/key/value it receives (pre-repeat_kv, so key/value are [B, Hkv, L, d])
    and the attention output ([B, T, H, d] after its internal transpose).
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import eager_attention_forward

    captured = {}

    def capture_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        out, weights = eager_attention_forward(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )
        captured["q"] = query.detach().clone()
        captured["k"] = key.detach().clone()
        captured["v"] = value.detach().clone()
        captured["ctx"] = out.detach().clone()
        return out, weights

    ALL_ATTENTION_FUNCTIONS.register("capture_eager", capture_eager)
    return "capture_eager", captured


def _run_case(model, captured, layer_idx, past, x, positions):
    """Drive one layer's self_attn; return (out, q_rot, k_all, v_all, ctx)."""
    import torch
    from transformers.cache_utils import DynamicCache

    cfg = model.config
    p_len = 0 if past is None else past[0].shape[1]
    t_len = x.shape[0]
    total = p_len + t_len

    cache = DynamicCache(config=cfg)
    if past is not None:
        past_k, past_v = past
        cache.update(past_k.unsqueeze(0), past_v.unsqueeze(0), layer_idx)

    carrier = torch.zeros(1, 1, 1, dtype=torch.float32)
    pos = torch.tensor([positions], dtype=torch.int64)
    cos, sin = model.model.rotary_emb(carrier, pos)

    # Additive causal mask [1, 1, T, L]: new query t (cache position P+t) attends
    # key positions [0, P+t] inclusive; disallowed entries get finfo.min.
    finfo_min = torch.finfo(torch.float32).min
    mask = torch.zeros(1, 1, t_len, total, dtype=torch.float32)
    for t in range(t_len):
        for s in range(total):
            if s > p_len + t:
                mask[0, 0, t, s] = finfo_min

    attn = model.model.layers[layer_idx].self_attn
    with torch.no_grad():
        out, _ = attn(
            x.unsqueeze(0),
            position_embeddings=(cos, sin),
            attention_mask=mask,
            past_key_values=cache,
        )

    q_rot = captured["q"][0].transpose(0, 1).contiguous()  # [T, H, d]
    k_all = captured["k"][0].contiguous()  # [Hkv, L, d]
    v_all = captured["v"][0].contiguous()  # [Hkv, L, d]
    ctx = captured["ctx"][0].contiguous()  # [T, H, d]
    return out[0].contiguous(), q_rot, k_all, v_all, ctx


def _record_attention(out_dir) -> None:
    import torch

    common.setup_torch_determinism(SEED)
    model = tiny_llama._build_model().float()
    model.eval()
    cfg = model.config
    key, captured = _capture_interface()
    cfg._attn_implementation = key

    d = cfg.head_dim
    hkv = cfg.num_key_value_heads
    e_dim = cfg.hidden_size

    gen = torch.Generator().manual_seed(SEED)
    tensors = {}
    manifest = []
    for name, layer_idx, p_len, t_len in CASES:
        positions = list(range(p_len, p_len + t_len))
        x = torch.randn(t_len, e_dim, generator=gen)
        past = None
        if p_len > 0:
            past_k = torch.randn(hkv, p_len, d, generator=gen)
            past_v = torch.randn(hkv, p_len, d, generator=gen)
            past = (past_k, past_v)

        out, q_rot, k_all, v_all, ctx = _run_case(
            model, captured, layer_idx, past, x, positions
        )

        tensors[f"{name}.x"] = x.contiguous()
        tensors[f"{name}.positions"] = torch.tensor(positions, dtype=torch.int32)
        if past is not None:
            tensors[f"{name}.past_k"] = past[0].contiguous()
            tensors[f"{name}.past_v"] = past[1].contiguous()
        tensors[f"{name}.q_rot"] = q_rot
        tensors[f"{name}.k_all"] = k_all
        tensors[f"{name}.v_all"] = v_all
        tensors[f"{name}.ctx"] = ctx
        tensors[f"{name}.out"] = out
        manifest.append(
            {"name": name, "layer": layer_idx, "P": p_len, "T": t_len, "positions": positions}
        )

    common.save_safetensors(tensors, out_dir / "attention.safetensors")
    common.dump_json(
        {
            "description": "HF LlamaAttention goldens (M5-T05). Per case (layer, "
            "P, T): x [T,E], positions [T] i32, past_k/past_v [Hkv,P,d] (P>0 "
            "only), q_rot [T,H,d] (post-RoPE queries), k_all/v_all [Hkv,P+T,d] "
            "(accumulated head-major cache), ctx [T,H,d] (attention output pre "
            "o_proj), out [T,E] (post o_proj) — all f32. GQA H=4/Hkv=2. "
            "Intermediates captured from HF eager attention (fp32); the reference "
            "reduces in its own order, so agreement is Class T (tolerances in "
            "attention_test.cpp).",
            "seed": SEED,
            "scale": float(model.model.layers[0].self_attn.scaling),
            "dims": {
                "hidden_size": e_dim,
                "head_dim": d,
                "num_heads": cfg.num_attention_heads,
                "num_kv_heads": hkv,
            },
            "cases": manifest,
            "regenerate": "tools/.venv/bin/python -m gen_fixtures tiny-llama-attention",
            "generator_versions": common.generator_versions(),
        },
        out_dir / "attention_meta.json",
    )


def run(args) -> None:
    out = pathlib.Path(args.out) / "models" / "tiny-llama"
    _record_attention(out / "expected")
    common.check_size_budget(out, 5_000_000, "models/tiny-llama")
    print(f"  wrote {out / 'expected' / 'attention.safetensors'}")
