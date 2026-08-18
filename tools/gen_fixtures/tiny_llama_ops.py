"""tiny-llama op-level goldens: GEMM + norm/activation cases (M5-T02, M5-T03).

Emits tests/fixtures/models/tiny-llama/expected/ops.safetensors — reference
inputs/outputs for the CPU scalar ops:

  * cpu::gemm (M5-T02): across shapes (k==1, skinny, wide, odd tails) and
    weight/bias storage dtypes (f32/f16/bf16), including the model's own real
    bf16 projection weights. Ground truth `A @ B.float().T (+ bias.float())`.
  * cpu::rmsnorm (M5-T03): f32 and bf16 inputs, varying eps. Ground truth
    `xf * rsqrt(mean(xf²) + eps) * weight.float()` — the *pure fp32 forward*
    (no HF `.to(input_dtype)` round-trip), exactly what the reference computes.
  * cpu::softmax (M5-T03): typical, large-magnitude (±1e4, stability), and
    causal (-inf masked) rows. Ground truth `torch.softmax(x, -1)`.
  * cpu::silu_mul / cpu::add (M5-T03): SwiGLU activation and residual add.
    Ground truth `F.silu(gate) * up` and `a + b`.

All ground truth is fp32 (convert-on-the-fly, fp32 accumulation;
docs/design/model-execution.md §3.3, §12). M5-T04 appends RoPE/embedding
goldens under the same subcommand family.

Determinism: a single seeded generator; the GEMM cases are drawn first (their
bytes are pinned by the committed M5-T02 goldens), then the M5-T03 cases in a
fixed order, so regeneration is byte-identical (tools/regen_fixtures.sh
--verify).
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


def _build_rmsnorm_cases(gen):
    """Returns an ordered list of (name, x, weight, eps) for cpu::rmsnorm.

    x carries the input storage dtype under test (f32 or bf16); weight matches.
    Drawn from `gen` in this exact order — call AFTER _build_cases so the GEMM
    draws (hence bytes) are unchanged.
    """
    import torch

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen).to(dtype)

    cases = []
    cases.append(("rmsnorm_f32", randn(16, 64), randn(64), 1e-5))
    cases.append(
        (
            "rmsnorm_bf16",  # the "RMSNorm on bf16 input" acceptance case
            randn(16, 64, dtype=torch.bfloat16),
            randn(64, dtype=torch.bfloat16),
            1e-5,
        )
    )
    cases.append(("rmsnorm_eps", randn(8, 64), randn(64), 0.5))  # eps dominates
    return cases


def _rmsnorm_ref(x, weight, eps):
    """The pure fp32 RMSNorm the reference computes (no bf16 round-trip)."""
    import torch

    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return xf * torch.rsqrt(var + eps) * weight.float()


def _build_softmax_cases(gen):
    """Returns an ordered list of (name, x) for cpu::softmax (last-dim)."""
    import torch

    def randn(*shape):
        return torch.randn(*shape, generator=gen)

    cases = []
    cases.append(("softmax_typical", randn(8, 37)))
    cases.append(("softmax_large", randn(4, 512) * 1e4))  # stability: exp overflow
    causal = randn(8, 8)
    mask = torch.triu(torch.ones(8, 8), diagonal=1).bool()  # strict upper tri
    cases.append(("softmax_causal", causal.masked_fill(mask, float("-inf"))))
    return cases


def _build_elementwise_cases(gen):
    """Returns an ordered list of (name, kind, t0, t1); kind in {silu_mul,add}."""
    import torch

    def randn(*shape):
        return torch.randn(*shape, generator=gen)

    cases = []
    cases.append(("silu_mul", "silu_mul", randn(16, 176), randn(16, 176)))
    # up == ones, so silu_mul reduces to pure SiLU(gate) — a true SiLU golden.
    cases.append(("silu_mul_unit", "silu_mul", randn(16, 176), torch.ones(16, 176)))
    # ±100 gate: exp(-gate) saturates; the reference must stay finite (no NaN).
    cases.append(("silu_mul_large", "silu_mul", randn(4, 176) * 100.0, randn(4, 176)))
    cases.append(("add_basic", "add", randn(16, 64), randn(16, 64)))
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

    # --- M5-T03 norm/activation cases, drawn AFTER the GEMM cases. ---
    rmsnorm_manifest = []
    for name, x, weight, eps in _build_rmsnorm_cases(gen):
        y = _rmsnorm_ref(x, weight, eps)
        tensors[f"{name}.x"] = x.contiguous().clone()
        tensors[f"{name}.weight"] = weight.contiguous().clone()
        tensors[f"{name}.y"] = y.contiguous().clone()
        rmsnorm_manifest.append(
            {
                "name": name,
                "rows": x.shape[0],
                "e": x.shape[1],
                "eps": eps,
                "x_dtype": str(x.dtype).removeprefix("torch."),
                "weight_dtype": str(weight.dtype).removeprefix("torch."),
            }
        )

    softmax_manifest = []
    for name, x in _build_softmax_cases(gen):
        y = torch.softmax(x, dim=-1)
        tensors[f"{name}.x"] = x.contiguous().clone()
        tensors[f"{name}.y"] = y.contiguous().clone()
        softmax_manifest.append({"name": name, "rows": x.shape[0], "n": x.shape[1]})

    elementwise_manifest = []
    for name, kind, t0, t1 in _build_elementwise_cases(gen):
        if kind == "silu_mul":
            y = torch.nn.functional.silu(t0) * t1
            tensors[f"{name}.gate"] = t0.contiguous().clone()
            tensors[f"{name}.up"] = t1.contiguous().clone()
        else:  # add
            y = t0 + t1
            tensors[f"{name}.a"] = t0.contiguous().clone()
            tensors[f"{name}.b"] = t1.contiguous().clone()
        tensors[f"{name}.y"] = y.contiguous().clone()
        elementwise_manifest.append(
            {"name": name, "kind": kind, "rows": t0.shape[0], "cols": t0.shape[1]}
        )

    common.save_safetensors(tensors, out_dir / "ops.safetensors")
    common.dump_json(
        {
            "description": "CPU reference-op goldens (M5-T02/T03). GEMM (cases): "
            "<name>.a [M,K] f32, <name>.b [N,K] weight-dtype, optional <name>.bias "
            "[N], <name>.c [M,N] f32 = a @ b.float().T (+ bias.float()). RMSNorm "
            "(rmsnorm_cases): <name>.x [R,E] x-dtype, <name>.weight [E], <name>.y "
            "[R,E] f32 = xf * rsqrt(mean(xf^2)+eps) * weight.float(). Softmax "
            "(softmax_cases): <name>.x [R,N] f32, <name>.y [R,N] f32 = "
            "softmax(x,-1). Elementwise (elementwise_cases): silu_mul <name>.gate/"
            ".up/.y = F.silu(gate)*up; add <name>.a/.b/.y = a+b. All ground truth "
            "is fp32; the reference converts f16/bf16 storage on the fly.",
            "seed": SEED,
            "cases": manifest,
            "rmsnorm_cases": rmsnorm_manifest,
            "softmax_cases": softmax_manifest,
            "elementwise_cases": elementwise_manifest,
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
