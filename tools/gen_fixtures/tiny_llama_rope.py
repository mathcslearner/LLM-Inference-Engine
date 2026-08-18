"""RoPE goldens: cos/sin tables + apply outputs + scaled inv_freq (M5-T04).

Emits tests/fixtures/models/tiny-llama/expected/rope.safetensors — HF's own
rotary-embedding output, the oracle for cpu::rope_apply and the `Rope` module
(docs/design/model-execution.md §7, §12):

  * tiny_table: cos/sin tables [max_pos, d/2] for the tiny-llama config
    (theta 1e4, head_dim 16, no scaling), positions 0..127. Only the first
    half (d/2) is stored — the half-rotation reuses cos[p,j] for both elements
    of a pair.
  * tiny_sparse / tiny_contig: apply_rotary_pos_emb inputs/outputs in our
    token-major [T, H, d] layout. tiny_sparse exercises the ticket's positions
    {0, 1, large=127} and GQA (H=4 query heads, Hkv=2 kv heads); tiny_contig
    is the ordinary contiguous-prefill path (positions 0..7).
  * llama3: the Llama-3.1 config's "llama3" rope_scaling (committed
    tests/fixtures/models/configs/llama3/config.json) — its scaled inv_freq
    [d/2] validates the _compute_llama3_parameters formula tightly, plus
    cos/sin at a few positions (larger positions carry a documented tolerance
    for fp32-cosf vs fp64-cos range reduction).
  * linear: a synthetic "linear" rope_scaling (factor 4) on the tiny dims —
    scaled inv_freq + cos/sin, the only golden for the linear path.

All cos/sin ground truth is HF fp32 (LlamaRotaryEmbedding forces float32
internally); the reference forms the angle in fp64 and stores fp32, so
agreement is Class T within the tolerances stated in rope_test.cpp.

This is its own subcommand (tiny-llama-rope) writing its own file, so the
committed ops.safetensors bytes (M5-T02/T03) are untouched.
"""

import json
import pathlib

from . import common, tiny_llama

SEED = 20260818

# tiny-llama RoPE dims (derived from tiny_llama.CONFIG_JSON).
TINY_HEADS = tiny_llama.HEADS        # 4 query heads
TINY_KV_HEADS = tiny_llama.KV_HEADS  # 2 kv heads
TINY_MAX_POS = tiny_llama.MAX_POS    # 128

SPARSE_POSITIONS = [0, 1, 127]       # ticket's {0, 1, large}; 127 = max_pos-1
CONTIG_POSITIONS = list(range(8))    # ordinary contiguous prefill
LLAMA3_POSITIONS = [0, 1, 512, 2048, 8192]
LINEAR_POSITIONS = [0, 1, 127]
LINEAR_FACTOR = 4.0


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "tiny-llama-rope",
        help="RoPE cos/sin + apply goldens for the CPU reference (M5-T04)",
    )
    p.add_argument("--out", default=str(common.DEFAULT_FIXTURES_DIR))
    p.set_defaults(func=run)


def _rope_for(config):
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    return LlamaRotaryEmbedding(config)


def _tiny_config():
    from transformers import LlamaConfig

    keys = {
        k: v
        for k, v in tiny_llama.CONFIG_JSON.items()
        if k not in ("architectures", "model_type")
    }
    return LlamaConfig(**keys)


def _linear_config():
    from transformers import LlamaConfig

    keys = {
        k: v
        for k, v in tiny_llama.CONFIG_JSON.items()
        if k not in ("architectures", "model_type")
    }
    keys["rope_scaling"] = {"rope_type": "linear", "factor": LINEAR_FACTOR}
    return LlamaConfig(**keys)


def _llama3_config():
    from transformers import LlamaConfig

    cfg_path = (
        common.REPO_ROOT
        / "tests"
        / "fixtures"
        / "models"
        / "configs"
        / "llama3"
        / "config.json"
    )
    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    return LlamaConfig(**cfg_dict)


def _cos_sin_half(rope, positions, half):
    """HF cos/sin at `positions`, first half [len(positions), d/2] fp32."""
    import torch

    carrier = torch.zeros(1, 1, 1, dtype=torch.float32)
    pos = torch.tensor([positions], dtype=torch.int64)
    with torch.no_grad():
        cos, sin = rope(carrier, pos)  # [1, P, head_dim]
    return cos[0, :, :half].contiguous(), sin[0, :, :half].contiguous()


def _apply(rope, q_tok, k_tok, positions):
    """Rotate token-major q[T,H,d]/k[T,Hkv,d] via HF, return token-major outs."""
    import torch
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    carrier = torch.zeros(1, 1, 1, dtype=torch.float32)
    pos = torch.tensor([positions], dtype=torch.int64)
    with torch.no_grad():
        cos, sin = rope(carrier, pos)  # [1, T, head_dim]
        # HF apply expects [B, heads, T, d]; we store [T, heads, d].
        q_hf = q_tok.permute(1, 0, 2).unsqueeze(0)
        k_hf = k_tok.permute(1, 0, 2).unsqueeze(0)
        q_out, k_out = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)
    return q_out[0].permute(1, 0, 2).contiguous(), k_out[0].permute(1, 0, 2).contiguous()


def _record_rope(out_dir) -> None:
    import torch

    common.setup_torch_determinism(SEED)
    gen = torch.Generator().manual_seed(SEED)

    tiny_cfg = _tiny_config()
    d = tiny_cfg.head_dim
    half = d // 2
    tiny_rope = _rope_for(tiny_cfg)

    tensors = {}

    # --- tiny_table: full 0..max_pos-1 half tables ---
    cos_tab, sin_tab = _cos_sin_half(tiny_rope, list(range(TINY_MAX_POS)), half)
    tensors["tiny_table.cos"] = cos_tab
    tensors["tiny_table.sin"] = sin_tab

    def randn(*shape):
        return torch.randn(*shape, generator=gen)

    # --- tiny_sparse: positions {0,1,127}, GQA heads ---
    q_s = randn(len(SPARSE_POSITIONS), TINY_HEADS, d)
    k_s = randn(len(SPARSE_POSITIONS), TINY_KV_HEADS, d)
    q_so, k_so = _apply(tiny_rope, q_s, k_s, SPARSE_POSITIONS)
    tensors["tiny_sparse.positions"] = torch.tensor(SPARSE_POSITIONS, dtype=torch.int32)
    tensors["tiny_sparse.q"] = q_s.contiguous()
    tensors["tiny_sparse.k"] = k_s.contiguous()
    tensors["tiny_sparse.q_out"] = q_so
    tensors["tiny_sparse.k_out"] = k_so

    # --- tiny_contig: contiguous positions 0..7 ---
    q_c = randn(len(CONTIG_POSITIONS), TINY_HEADS, d)
    k_c = randn(len(CONTIG_POSITIONS), TINY_KV_HEADS, d)
    q_co, k_co = _apply(tiny_rope, q_c, k_c, CONTIG_POSITIONS)
    tensors["tiny_contig.positions"] = torch.tensor(CONTIG_POSITIONS, dtype=torch.int32)
    tensors["tiny_contig.q"] = q_c.contiguous()
    tensors["tiny_contig.k"] = k_c.contiguous()
    tensors["tiny_contig.q_out"] = q_co
    tensors["tiny_contig.k_out"] = k_co

    # --- llama3: scaled inv_freq + cos/sin at a few positions ---
    l3_cfg = _llama3_config()
    l3_half = l3_cfg.head_dim // 2
    l3_rope = _rope_for(l3_cfg)
    tensors["llama3.inv_freq"] = l3_rope.inv_freq.detach().float().contiguous()
    l3_cos, l3_sin = _cos_sin_half(l3_rope, LLAMA3_POSITIONS, l3_half)
    tensors["llama3.positions"] = torch.tensor(LLAMA3_POSITIONS, dtype=torch.int32)
    tensors["llama3.cos"] = l3_cos
    tensors["llama3.sin"] = l3_sin

    # --- linear: scaled inv_freq + cos/sin ---
    lin_cfg = _linear_config()
    lin_rope = _rope_for(lin_cfg)
    tensors["linear.inv_freq"] = lin_rope.inv_freq.detach().float().contiguous()
    lin_cos, lin_sin = _cos_sin_half(lin_rope, LINEAR_POSITIONS, half)
    tensors["linear.positions"] = torch.tensor(LINEAR_POSITIONS, dtype=torch.int32)
    tensors["linear.cos"] = lin_cos
    tensors["linear.sin"] = lin_sin

    common.save_safetensors(tensors, out_dir / "rope.safetensors")
    common.dump_json(
        {
            "description": "HF RoPE goldens (M5-T04). tiny_table.cos/sin [max_pos, "
            "d/2] f32. tiny_sparse/tiny_contig.{positions [T] i32, q [T,H,d], k "
            "[T,Hkv,d], q_out, k_out} f32 = apply_rotary_pos_emb (half-rotation). "
            "llama3.{inv_freq [d/2], positions [P] i32, cos/sin [P,d/2]} from the "
            "committed Llama-3.1 llama3 rope_scaling. linear.{inv_freq, positions, "
            "cos, sin} from a synthetic linear rope_scaling (factor "
            f"{LINEAR_FACTOR}). Reference forms angles in fp64, stores fp32 — "
            "agreement is Class T (tolerances in rope_test.cpp).",
            "seed": SEED,
            "tiny": {
                "head_dim": d,
                "theta": tiny_llama.ROPE_THETA,
                "max_pos": TINY_MAX_POS,
                "heads": TINY_HEADS,
                "kv_heads": TINY_KV_HEADS,
                "sparse_positions": SPARSE_POSITIONS,
                "contig_positions": CONTIG_POSITIONS,
            },
            "llama3": {
                "head_dim": l3_cfg.head_dim,
                "theta": float(l3_cfg.rope_parameters["rope_theta"]),
                "rope_scaling": {
                    "rope_type": "llama3",
                    "factor": float(l3_cfg.rope_parameters["factor"]),
                    "low_freq_factor": float(l3_cfg.rope_parameters["low_freq_factor"]),
                    "high_freq_factor": float(l3_cfg.rope_parameters["high_freq_factor"]),
                    "original_max_position_embeddings": int(
                        l3_cfg.rope_parameters["original_max_position_embeddings"]
                    ),
                },
                "positions": LLAMA3_POSITIONS,
            },
            "linear": {
                "head_dim": d,
                "theta": tiny_llama.ROPE_THETA,
                "factor": LINEAR_FACTOR,
                "positions": LINEAR_POSITIONS,
            },
            "regenerate": "tools/.venv/bin/python -m gen_fixtures tiny-llama-rope",
            "generator_versions": common.generator_versions(),
        },
        out_dir / "rope_meta.json",
    )


def run(args) -> None:
    out = pathlib.Path(args.out) / "models" / "tiny-llama"
    _record_rope(out / "expected")
    common.check_size_budget(out, 5_000_000, "models/tiny-llama")
    print(f"  wrote {out / 'expected' / 'rope.safetensors'}")
