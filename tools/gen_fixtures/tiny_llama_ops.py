"""tiny-llama op-level goldens: GEMM cases (M5-T02).

Emits tests/fixtures/models/tiny-llama/expected/ops.safetensors — reference
inputs/outputs for cpu::gemm across shapes (k==1, skinny, wide, odd tails) and
weight/bias storage dtypes (f32/f16/bf16), including the model's own real bf16
projection weights on the fixture's dims. Ground truth is fp32
`A @ B.float().T (+ bias.float())`, exactly what the scalar reference computes
(convert-on-the-fly, fp32 accumulation; docs/design/model-execution.md §3.3,
§12). M5-T03/T04 append RMSNorm/softmax/SiLU/RoPE/embedding goldens to
ops.safetensors / rope.safetensors under the same subcommand family.

Determinism: a single seeded generator, tensors drawn in sorted-case order, so
regeneration is byte-identical (tools/regen_fixtures.sh --verify).
"""

from . import common, tiny_llama

SEED = 20260817


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "tiny-llama-ops",
        help="op-level goldens (GEMM cases) for the CPU reference (M5-T02)",
    )
    p.add_argument("--out", default=str(common.DEFAULT_FIXTURES_DIR))
    p.set_defaults(func=run)


def _real_weights():
    """The tiny-llama checkpoint's real bf16 projection weights, by name."""
    model = tiny_llama._build_model()  # bf16, same seed as the committed ckpt
    sd = model.state_dict()
    return {
        "q_proj": sd["model.layers.0.self_attn.q_proj.weight"],   # [64, 64]
        "gate_proj": sd["model.layers.0.mlp.gate_proj.weight"],   # [176, 64]
        "down_proj": sd["model.layers.0.mlp.down_proj.weight"],   # [64, 176]
        "lm_head": sd["lm_head.weight"],                          # [512, 64]
    }


def _build_cases(gen):
    """Returns an ordered list of (name, A, B, bias|None) case tuples.

    A is always fp32; B and bias carry the storage dtype under test. Shapes are
    (M, K) x (N, K) -> (M, N), the NT form cpu::gemm implements.
    """
    import torch

    real = _real_weights()

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen).to(dtype)

    # Order is fixed (this list): the generator is drawn in exactly this
    # sequence, so the committed bytes are stable.
    cases = []

    # --- Real-weight cases: the actual model ops on the fixture's dims. ---
    cases.append(("q_proj_prefill", randn(16, 64), real["q_proj"], None))
    cases.append(("gate_proj_wide", randn(16, 64), real["gate_proj"], None))
    cases.append(("down_proj", randn(16, 176), real["down_proj"], None))
    cases.append(("lm_head_wide", randn(4, 64), real["lm_head"], None))
    cases.append(("decode_gemv", randn(1, 64), real["q_proj"], None))

    # --- Synthetic shape-stress cases (f32 weights). ---
    cases.append(("skinny_tall", randn(257, 8), randn(3, 8), None))       # M >> N
    cases.append(("k_one", randn(5, 1), randn(7, 1), None))              # K == 1
    cases.append(
        ("odd_tails", randn(65, 129), randn(33, 129), randn(33))        # non-tile-multiples + f32 bias
    )

    # --- Storage-dtype cases: exercise the widening paths directly. ---
    cases.append(("fp16_weight", randn(8, 64), randn(32, 64, dtype=torch.float16), None))
    cases.append(
        ("bf16_bias", randn(8, 64), randn(32, 64, dtype=torch.bfloat16),
         randn(32, dtype=torch.bfloat16))                                # Qwen2-style bf16 bias
    )
    return cases


def _record_ops(out_dir) -> None:
    import torch

    common.setup_torch_determinism(SEED)
    gen = torch.Generator().manual_seed(SEED)

    cases = _build_cases(gen)
    tensors = {}
    manifest = []
    for name, a, b, bias in cases:
        c = a.float() @ b.float().t()
        if bias is not None:
            c = c + bias.float()
        # .clone() so no two entries alias storage (decode_gemv and
        # q_proj_prefill reuse the same real q_proj weight) — safetensors
        # rejects shared memory.
        tensors[f"{name}.a"] = a.contiguous().clone()
        tensors[f"{name}.b"] = b.contiguous().clone()
        tensors[f"{name}.c"] = c.contiguous().clone()
        entry = {
            "name": name,
            "m": a.shape[0],
            "k": a.shape[1],
            "n": b.shape[0],
            "weight_dtype": str(b.dtype).removeprefix("torch."),
            "bias": bias is not None,
        }
        if bias is not None:
            tensors[f"{name}.bias"] = bias.contiguous().clone()
            entry["bias_dtype"] = str(bias.dtype).removeprefix("torch.")
        manifest.append(entry)

    common.save_safetensors(tensors, out_dir / "ops.safetensors")
    common.dump_json(
        {
            "description": "cpu::gemm goldens (M5-T02): per case <name>.a [M,K] f32, "
            "<name>.b [N,K] weight-dtype, optional <name>.bias [N] bias-dtype, "
            "<name>.c [M,N] f32 = a @ b.float().T (+ bias.float()). Ground truth "
            "is fp32; the reference converts f16/bf16 weights on the fly.",
            "seed": SEED,
            "cases": manifest,
            "regenerate": "tools/.venv/bin/python -m gen_fixtures tiny-llama-ops",
            "generator_versions": common.generator_versions(),
        },
        out_dir / "ops_meta.json",
    )


def run(args) -> None:
    import pathlib

    out = pathlib.Path(args.out) / "models" / "tiny-llama"
    _record_ops(out / "expected")
    # Shares tiny-llama's 5 MB per-model budget; the op goldens are a few
    # hundred KB. The assertion covers the whole model dir.
    common.check_size_budget(out, 5_000_000, "models/tiny-llama")
    print(f"  wrote {out / 'expected' / 'ops.safetensors'}")
